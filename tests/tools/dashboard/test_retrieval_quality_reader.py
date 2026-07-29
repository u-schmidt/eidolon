"""Unit tests for RetrievalQualityReader — the aggregation layer behind
the dashboard's retrieval-quality page. Uses MockStorage (label
resolution) plus a simple in-memory fake for TraceReader (time-window
filtering itself is already covered by
tests/core/trace/test_*_trace_reader.py); these tests only exercise the
tally/averaging/near-miss/injected-fact aggregation math."""

from __future__ import annotations

from datetime import UTC, datetime

from core.models import MemoryItem
from tests.mocks import MockStorage
from tools.dashboard.retrieval_quality_reader import RetrievalQualityReader

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


def _candidate(candidate_type: str, candidate_id: int, final_score: float, included: bool, **extra) -> dict:
    return {
        "candidate_type": candidate_type,
        "candidate_id": candidate_id,
        "semantic_raw": 0.5,
        "semantic_weighted": 0.2,
        "temporal_raw": 1.0,
        "temporal_weighted": 0.1,
        "final_score": final_score,
        "included": included,
        **extra,
    }


def _scoring_event(top_candidates: list[dict]) -> dict:
    return {"payload": {"candidate_type": "topic", "top_candidates": top_candidates}}


def _context_event(facts: list[dict]) -> dict:
    return {"payload": {"facts": facts}}


class TestTopRetrievedAndNeverIncluded:
    def test_tallies_included_and_scored_counts(self):
        storage = MockStorage()
        topic_a = storage.create_topic("Topic A", workspace_id="u1")
        topic_b = storage.create_topic("Topic B", workspace_id="u1")
        events = [
            _scoring_event([
                _candidate("topic", topic_a, 0.5, included=True),
                _candidate("topic", topic_b, 0.3, included=False),
            ]),
            _scoring_event([
                _candidate("topic", topic_a, 0.6, included=True),
            ]),
        ]
        reader = RetrievalQualityReader(
            FakeTraceReader({"retrieval_scoring_summary": events}), storage,
        )

        summary = reader.summary(_NOW)

        assert summary.total_events == 2
        assert len(summary.top_retrieved) == 1
        assert summary.top_retrieved[0].candidate_id == topic_a
        assert summary.top_retrieved[0].included_count == 2
        assert summary.top_retrieved[0].scored_count == 2
        assert summary.top_retrieved[0].label == "Topic A"

        assert len(summary.never_included) == 1
        assert summary.never_included[0].candidate_id == topic_b
        assert summary.never_included[0].scored_count == 1
        assert summary.never_included[0].included_count == 0


class TestScoreComponents:
    def test_averages_shares_and_none_when_field_absent(self):
        storage = MockStorage()
        events = [
            _scoring_event([
                _candidate(
                    "topic", 1, 0.5, included=True,
                    tag_overlap_weighted=0.15, person_weighted=0.05,
                ),
            ]),
            _scoring_event([
                _candidate("memory_item", 2, 0.4, included=True),
            ]),
        ]
        reader = RetrievalQualityReader(FakeTraceReader({"retrieval_scoring_summary": events}), storage)

        summary = reader.summary(_NOW)
        by_type = {c.candidate_type: c for c in summary.score_components}

        topic_stats = by_type["topic"]
        assert topic_stats.avg_semantic_share == 0.2 / 0.5
        assert topic_stats.avg_tag_overlap_share == 0.15 / 0.5
        assert topic_stats.avg_person_share == 0.05 / 0.5
        assert topic_stats.avg_temporal_share == 0.1 / 0.5

        item_stats = by_type["memory_item"]
        assert item_stats.avg_semantic_share == 0.2 / 0.4
        assert item_stats.avg_tag_overlap_share is None
        assert item_stats.avg_person_share is None


class TestNearMisses:
    def test_includes_candidates_within_margin_excludes_others(self):
        storage = MockStorage()
        topic_close = storage.create_topic("Close call", workspace_id="u1")
        topic_far = storage.create_topic("Not close", workspace_id="u1")
        events = [
            _scoring_event([
                _candidate("topic", 99, 0.50, included=True),
                _candidate("topic", topic_close, 0.46, included=False),  # within 0.05 margin
                _candidate("topic", topic_far, 0.40, included=False),  # outside margin
            ]),
        ]
        reader = RetrievalQualityReader(FakeTraceReader({"retrieval_scoring_summary": events}), storage)

        summary = reader.summary(_NOW)

        near_miss_ids = {nm.candidate_id for nm in summary.near_misses}
        assert topic_close in near_miss_ids
        assert topic_far not in near_miss_ids
        near_miss = next(nm for nm in summary.near_misses if nm.candidate_id == topic_close)
        assert near_miss.near_miss_count == 1
        assert near_miss.final_score == 0.46

    def test_no_near_misses_when_no_included_candidates(self):
        storage = MockStorage()
        events = [_scoring_event([_candidate("topic", 1, 0.3, included=False)])]
        reader = RetrievalQualityReader(FakeTraceReader({"retrieval_scoring_summary": events}), storage)

        assert reader.summary(_NOW).near_misses == []

    def test_near_miss_kind_filter_scopes_to_one_kind(self):
        storage = MockStorage()
        topic_id = storage.create_topic("Close topic", workspace_id="u1")
        item_id = storage.store_memory_item(MemoryItem(
            content="close item", item_type="fact", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=topic_id,
        ))
        events = [
            _scoring_event([
                _candidate("topic", 99, 0.50, included=True),
                _candidate("topic", topic_id, 0.46, included=False),  # within margin
                _candidate("memory_item", item_id, 0.46, included=False),  # within margin
            ]),
        ]
        reader = RetrievalQualityReader(FakeTraceReader({"retrieval_scoring_summary": events}), storage)

        summary = reader.summary(_NOW, near_miss_kind="topic")

        assert {nm.kind for nm in summary.near_misses} == {"topic"}
        assert {nm.candidate_id for nm in summary.near_misses} == {topic_id}

    def test_near_miss_kind_filter_empty_for_absent_kind(self):
        storage = MockStorage()
        topic_id = storage.create_topic("Close topic", workspace_id="u1")
        events = [
            _scoring_event([
                _candidate("topic", 99, 0.50, included=True),
                _candidate("topic", topic_id, 0.46, included=False),
            ]),
        ]
        reader = RetrievalQualityReader(FakeTraceReader({"retrieval_scoring_summary": events}), storage)

        summary = reader.summary(_NOW, near_miss_kind="cluster")

        assert summary.near_misses == []


class TestTopInjectedFacts:
    def test_latest_value_wins_and_counts_injections(self):
        storage = MockStorage()
        events = [
            _context_event([{"category": "work", "key": "role", "value": "engineer"}]),
            _context_event([{"category": "work", "key": "role", "value": "senior engineer"}]),
        ]
        reader = RetrievalQualityReader(FakeTraceReader({"context_retrieval": events}), storage)

        summary = reader.summary(_NOW)

        assert len(summary.top_injected_facts) == 1
        fact = summary.top_injected_facts[0]
        assert fact.category == "work"
        assert fact.key == "role"
        assert fact.value == "senior engineer"
        assert fact.injection_count == 2


class TestEmptyWindow:
    def test_no_events_returns_empty_summary(self):
        storage = MockStorage()
        reader = RetrievalQualityReader(FakeTraceReader(), storage)

        summary = reader.summary(_NOW)

        assert summary.total_events == 0
        assert summary.top_retrieved == []
        assert summary.never_included == []
        assert summary.score_components == []
        assert summary.near_misses == []
        assert summary.top_injected_facts == []
