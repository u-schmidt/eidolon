"""Unit tests for PromptQualityReader — the aggregation layer behind the
dashboard's prompt/pipeline-quality page. Uses MockStorage (facts) plus a
simple in-memory fake for TraceReader (time-window filtering itself is
already covered by tests/core/trace/test_*_trace_reader.py); these tests
only exercise the fallback/flip-flop/tally/churn aggregation math."""

from __future__ import annotations

from datetime import UTC, datetime

from core.models import Fact
from tests.mocks import MockStorage
from tools.dashboard.prompt_quality_reader import PromptQualityReader

_NOW = datetime(2026, 7, 27, 18, 0, 0, tzinfo=UTC)


class FakeTraceReader:
    def __init__(self, events_by_type: dict[str, list[dict]] | None = None) -> None:
        self._events_by_type = events_by_type or {}

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def events_for_request(self, request_id: str) -> list[dict]:
        return []

    def events_of_type_since(self, event_type: str, since: datetime) -> list[dict]:
        return self._events_by_type.get(event_type, [])


def _detection_event(response: str, result_type: str) -> dict:
    return {"payload": {"response": response, "result_type": result_type}}


def _relabel_event(topic_id: int, label: str | None) -> dict:
    return {"payload": {"topic_id": topic_id, "label": label}}


def _extraction_event(items: list[dict]) -> dict:
    return {"payload": {"items": items}}


def _consolidation_event(upserts: list[dict]) -> dict:
    return {"payload": {"upserts": upserts}}


def _correction_event(category: str, key: str, action: str) -> dict:
    return {"payload": {"peer_id": "u1", "category": category, "key": key, "value": "x", "action": action}}


class TestContinueFallback:
    def test_classifies_literal_fallback_and_new_topic(self):
        events = [
            _detection_event("CONTINUE", "CONTINUE"),
            _detection_event("continue", "CONTINUE"),  # literal, case-insensitive
            _detection_event("uh, sure, let's keep going", "CONTINUE"),  # fallback
            _detection_event("NEW_TOPIC:Trip planning", "NEW_TOPIC"),
        ]
        reader = PromptQualityReader(FakeTraceReader({"topic_detection": events}), MockStorage(), [])

        stats = reader.summary(_NOW).continue_fallback

        assert stats.total_calls == 4
        assert stats.literal_continue == 2
        assert stats.fallback_continue == 1
        assert stats.new_topic == 1


class TestRelabelFlipFlops:
    def test_topic_with_multiple_distinct_labels_is_flagged(self):
        events = [
            _relabel_event(1, "Weekend trip"),
            _relabel_event(1, "Grocery list"),
            _relabel_event(1, "Weekend trip"),
        ]
        reader = PromptQualityReader(FakeTraceReader({"topic_relabel": events}), MockStorage(), [])

        flip_flops = reader.summary(_NOW).relabel_flip_flops

        assert len(flip_flops) == 1
        assert flip_flops[0].topic_id == 1
        assert flip_flops[0].distinct_labels == 2
        assert flip_flops[0].relabel_count == 3
        assert flip_flops[0].latest_label == "Weekend trip"

    def test_topic_with_stable_label_is_excluded(self):
        events = [_relabel_event(2, "Stable label"), _relabel_event(2, "Stable label")]
        reader = PromptQualityReader(FakeTraceReader({"topic_relabel": events}), MockStorage(), [])

        assert reader.summary(_NOW).relabel_flip_flops == []

    def test_failed_relabel_call_with_no_label_is_ignored(self):
        events = [_relabel_event(3, None), _relabel_event(3, "First real label")]
        reader = PromptQualityReader(FakeTraceReader({"topic_relabel": events}), MockStorage(), [])

        assert reader.summary(_NOW).relabel_flip_flops == []


class TestTypeCounts:
    def test_tallies_item_and_attribution_types_independently(self):
        events = [
            _extraction_event([
                {"content": "a", "item_type": "fact", "attribution_type": "self_statement"},
                {"content": "b", "item_type": "fact", "attribution_type": "self_statement"},
                {"content": "c", "item_type": "essence", "attribution_type": "third_party"},
            ]),
        ]
        reader = PromptQualityReader(FakeTraceReader({"extraction": events}), MockStorage(), [])

        summary = reader.summary(_NOW)

        item_counts = {c.type_name: c.count for c in summary.item_type_counts}
        assert item_counts == {"fact": 2, "essence": 1}
        attribution_counts = {c.type_name: c.count for c in summary.attribution_type_counts}
        assert attribution_counts == {"self_statement": 2, "third_party": 1}


class TestFactCategoryOutliers:
    def test_flags_facts_outside_configured_categories(self):
        storage = MockStorage()
        storage.facts.append(Fact(peer_id="u1", category="basic", key="name", value="Alex", source="user"))
        storage.facts.append(Fact(peer_id="u1", category="typo_category", key="thing", value="x", source="ai"))
        reader = PromptQualityReader(FakeTraceReader(), storage, ["basic", "work"])

        outliers = reader.summary(_NOW).fact_category_outliers

        assert len(outliers) == 1
        assert outliers[0].category == "typo_category"

    def test_ignores_selected_window(self):
        # get_all_facts() is not time-filtered — verified indirectly by
        # not passing any topic_detection/etc events at all and still
        # getting the outlier back.
        storage = MockStorage()
        storage.facts.append(Fact(peer_id="u1", category="bad", key="k", value="v", source="ai"))
        reader = PromptQualityReader(FakeTraceReader(), storage, ["basic"])

        assert len(reader.summary(_NOW).fact_category_outliers) == 1


class TestFactChurn:
    def test_repeated_upserts_flagged_with_distinct_value_count(self):
        events = [
            _consolidation_event([{"category": "Work", "key": "Role", "value": "engineer"}]),
            _consolidation_event([{"category": "work", "key": "role", "value": "senior engineer"}]),
            _consolidation_event([{"category": "basic", "key": "name", "value": "Alex"}]),  # only once
        ]
        reader = PromptQualityReader(FakeTraceReader({"fact_consolidation": events}), MockStorage(), [])

        churn = reader.summary(_NOW).fact_churn

        assert len(churn) == 1
        assert churn[0].category == "work"
        assert churn[0].key == "role"
        assert churn[0].upsert_count == 2
        assert churn[0].distinct_values == 2

    def test_repeated_upserts_with_same_value_still_counted_but_distinct_values_one(self):
        events = [
            _consolidation_event([{"category": "basic", "key": "name", "value": "Alex"}]),
            _consolidation_event([{"category": "basic", "key": "name", "value": "Alex"}]),
        ]
        reader = PromptQualityReader(FakeTraceReader({"fact_consolidation": events}), MockStorage(), [])

        churn = reader.summary(_NOW).fact_churn

        assert len(churn) == 1
        assert churn[0].upsert_count == 2
        assert churn[0].distinct_values == 1


class TestFactCorrection:
    def test_counts_actions_and_ranks_top_corrected(self):
        events = [
            _correction_event("basic", "name", "upsert"),
            _correction_event("basic", "name", "upsert"),
            _correction_event("work", "role", "delete"),
        ]
        reader = PromptQualityReader(FakeTraceReader({"fact_correction": events}), MockStorage(), [])

        stats = reader.summary(_NOW).fact_correction

        assert stats.total_corrections == 3
        assert stats.upserts == 2
        assert stats.deletes == 1
        assert stats.top_corrected[0].category == "basic"
        assert stats.top_corrected[0].key == "name"
        assert stats.top_corrected[0].correction_count == 2


class TestEmptyWindow:
    def test_no_events_returns_empty_summary(self):
        reader = PromptQualityReader(FakeTraceReader(), MockStorage(), ["basic"])

        summary = reader.summary(_NOW)

        assert summary.total_topic_detection_calls == 0
        assert summary.continue_fallback.total_calls == 0
        assert summary.relabel_flip_flops == []
        assert summary.item_type_counts == []
        assert summary.attribution_type_counts == []
        assert summary.fact_category_outliers == []
        assert summary.fact_churn == []
        assert summary.fact_correction.total_corrections == 0
