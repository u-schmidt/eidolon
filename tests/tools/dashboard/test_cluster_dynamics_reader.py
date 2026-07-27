"""Unit tests for ClusterDynamicsReader — the aggregation layer behind
the dashboard's cluster-dynamics page. Uses MockStorage (get_cluster
lookups) plus a simple in-memory fake for TraceReader (time-window
filtering itself is already covered by
tests/core/trace/test_*_trace_reader.py); these tests only exercise the
decision-classification/most-active aggregation math."""

from __future__ import annotations

from datetime import UTC, datetime

from tests.mocks import MockStorage
from tools.dashboard.cluster_dynamics_reader import ClusterDynamicsReader

_NOW = datetime(2026, 7, 29, 12, 0, 0, tzinfo=UTC)


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


def _event(match_index: int | None, candidates: list[dict], resulting_cluster_id: int | None) -> dict:
    return {
        "payload": {
            "match_index": match_index,
            "candidates": candidates,
            "resulting_cluster_id": resulting_cluster_id,
        },
    }


class TestDecisionStats:
    def test_matched_existing_when_candidate_kind_is_cluster(self):
        events = [_event(1, [{"kind": "cluster", "id": 5, "label": "Thread"}], resulting_cluster_id=5)]
        reader = ClusterDynamicsReader(
            FakeTraceReader({"cluster_consolidation": events}), MockStorage(),
        )

        stats = reader.summary(_NOW).decisions

        assert stats.total_calls == 1
        assert stats.matched_existing == 1
        assert stats.new_cluster_created == 0
        assert stats.no_match == 0

    def test_new_cluster_created_when_candidate_kind_is_topic(self):
        events = [_event(1, [{"kind": "topic", "id": 9, "label": "Old chat"}], resulting_cluster_id=42)]
        reader = ClusterDynamicsReader(
            FakeTraceReader({"cluster_consolidation": events}), MockStorage(),
        )

        stats = reader.summary(_NOW).decisions

        assert stats.new_cluster_created == 1
        assert stats.matched_existing == 0

    def test_no_match_when_match_index_is_none(self):
        events = [_event(None, [{"kind": "cluster", "id": 5, "label": "Thread"}], resulting_cluster_id=None)]
        reader = ClusterDynamicsReader(
            FakeTraceReader({"cluster_consolidation": events}), MockStorage(),
        )

        stats = reader.summary(_NOW).decisions

        assert stats.no_match == 1

    def test_no_match_when_match_index_out_of_range(self):
        events = [_event(5, [{"kind": "cluster", "id": 5, "label": "Thread"}], resulting_cluster_id=None)]
        reader = ClusterDynamicsReader(
            FakeTraceReader({"cluster_consolidation": events}), MockStorage(),
        )

        assert reader.summary(_NOW).decisions.no_match == 1


class TestMostActiveClusters:
    def test_ranks_by_activity_count_and_resolves_label(self):
        storage = MockStorage()
        cid = storage.create_cluster("u1", "Recurring topic", "summary")
        storage.clusters[cid]["topic_count"] = 7
        events = [
            _event(1, [{"kind": "cluster", "id": cid, "label": "Recurring topic"}], resulting_cluster_id=cid),
            _event(1, [{"kind": "cluster", "id": cid, "label": "Recurring topic"}], resulting_cluster_id=cid),
        ]
        reader = ClusterDynamicsReader(FakeTraceReader({"cluster_consolidation": events}), storage)

        active = reader.summary(_NOW).most_active_clusters

        assert len(active) == 1
        assert active[0].cluster_id == cid
        assert active[0].label == "Recurring topic"
        assert active[0].activity_count == 2
        assert active[0].total_topic_count == 7

    def test_deleted_cluster_resolves_to_none_label(self):
        events = [_event(1, [{"kind": "cluster", "id": 999, "label": "Gone"}], resulting_cluster_id=999)]
        reader = ClusterDynamicsReader(FakeTraceReader({"cluster_consolidation": events}), MockStorage())

        active = reader.summary(_NOW).most_active_clusters

        assert active[0].cluster_id == 999
        assert active[0].label is None
        assert active[0].total_topic_count == 0

    def test_events_with_no_resulting_cluster_excluded(self):
        events = [_event(None, [], resulting_cluster_id=None)]
        reader = ClusterDynamicsReader(FakeTraceReader({"cluster_consolidation": events}), MockStorage())

        assert reader.summary(_NOW).most_active_clusters == []


class TestEmptyWindow:
    def test_no_events_returns_empty_summary(self):
        reader = ClusterDynamicsReader(FakeTraceReader(), MockStorage())

        summary = reader.summary(_NOW)

        assert summary.decisions.total_calls == 0
        assert summary.most_active_clusters == []
