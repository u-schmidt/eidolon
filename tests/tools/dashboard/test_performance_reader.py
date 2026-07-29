"""Unit tests for PerformanceReader — the aggregation layer behind the
dashboard's performance/reliability page. Uses a simple in-memory fake
for MetricsReader (time-window filtering itself is already covered by
tests/core/test_metrics_reader_contract.py); these tests only exercise
the percentile/error-rate/bucketing/ranking math."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tools.dashboard.performance_reader import PerformanceReader

_NOW = datetime(2026, 7, 26, 18, 0, 0, tzinfo=UTC)


class FakeMetricsReader:
    def __init__(
        self,
        llm_calls: list[dict] | None = None,
        embedding_calls: list[dict] | None = None,
        events: list[dict] | None = None,
    ) -> None:
        self._llm_calls = llm_calls or []
        self._embedding_calls = embedding_calls or []
        self._events = events or []

    def llm_calls_since(self, since):
        return self._llm_calls

    def embedding_calls_since(self, since):
        return self._embedding_calls

    def events_since(self, since):
        return self._events


def _llm_call(call_type: str, duration_ms: float, succeeded: bool = True, created_at=None) -> dict:
    return {
        "call_type": call_type, "duration_ms": duration_ms, "succeeded": succeeded,
        "workspace_id": "u1", "session_id": None, "created_at": (created_at or _NOW).isoformat(),
    }


def _embedding_call(purpose: str, duration_ms: float, succeeded: bool = True) -> dict:
    return {
        "purpose": purpose, "duration_ms": duration_ms, "dimension": 768, "succeeded": succeeded,
        "entity_type": None, "entity_id": None, "workspace_id": "u1", "session_id": None,
        "created_at": _NOW.isoformat(),
    }


def _error_event(workspace_id: str | None = "u1", session_id: str | None = "s1") -> dict:
    return {
        "severity": "error", "step": "extraction", "message": "boom",
        "workspace_id": workspace_id, "session_id": session_id, "created_at": _NOW.isoformat(),
    }


class TestCallTypeStats:
    def test_groups_by_call_type(self):
        reader = PerformanceReader(FakeMetricsReader(llm_calls=[
            _llm_call("topic_detection", 10.0),
            _llm_call("extraction", 20.0),
        ]))

        stats = reader.summary(_NOW - timedelta(hours=1)).call_type_stats
        assert {s.call_type for s in stats} == {"topic_detection", "extraction"}
        assert all(s.count == 1 for s in stats)

    def test_percentiles_nearest_rank(self):
        calls = [_llm_call("topic_detection", d) for d in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
        reader = PerformanceReader(FakeMetricsReader(llm_calls=calls))

        stats = reader.summary(_NOW - timedelta(hours=1)).call_type_stats
        assert len(stats) == 1
        assert stats[0].p50_ms == 50
        assert stats[0].p95_ms == 100
        assert stats[0].p99_ms == 100

    def test_error_count_reflects_failed_calls(self):
        reader = PerformanceReader(FakeMetricsReader(llm_calls=[
            _llm_call("topic_detection", 10.0, succeeded=True),
            _llm_call("topic_detection", 20.0, succeeded=False),
        ]))

        stats = reader.summary(_NOW - timedelta(hours=1)).call_type_stats
        assert stats[0].count == 2
        assert stats[0].error_count == 1

    def test_empty_when_no_calls(self):
        reader = PerformanceReader(FakeMetricsReader())
        assert reader.summary(_NOW - timedelta(hours=1)).call_type_stats == []


class TestEmbeddingStats:
    def test_groups_by_purpose(self):
        reader = PerformanceReader(FakeMetricsReader(embedding_calls=[
            _embedding_call("retrieval_query", 5.0),
            _embedding_call("memory_item", 8.0),
        ]))

        stats = reader.summary(_NOW - timedelta(hours=1)).embedding_stats
        assert {s.purpose for s in stats} == {"retrieval_query", "memory_item"}


class TestHourlyThroughput:
    def test_buckets_by_hour(self):
        reader = PerformanceReader(FakeMetricsReader(llm_calls=[
            _llm_call("topic_detection", 10.0, created_at=_NOW),
            _llm_call("topic_detection", 10.0, created_at=_NOW + timedelta(minutes=10)),
            _llm_call("topic_detection", 10.0, created_at=_NOW + timedelta(hours=1)),
        ]))

        buckets = reader.summary(_NOW - timedelta(hours=1)).hourly_throughput
        assert len(buckets) == 2
        assert buckets[0].llm_call_count == 2
        assert buckets[1].llm_call_count == 1

    def test_empty_when_no_calls(self):
        reader = PerformanceReader(FakeMetricsReader())
        assert reader.summary(_NOW - timedelta(hours=1)).hourly_throughput == []


class TestTopErrorSources:
    def test_ranks_workspaces_by_error_count(self):
        reader = PerformanceReader(FakeMetricsReader(events=[
            _error_event(workspace_id="u1"),
            _error_event(workspace_id="u1"),
            _error_event(workspace_id="u2"),
        ]))

        top = reader.summary(_NOW - timedelta(hours=1)).top_error_workspaces
        assert top[0].key == "u1"
        assert top[0].error_count == 2
        assert top[1].key == "u2"
        assert top[1].error_count == 1

    def test_ranks_sessions_separately(self):
        reader = PerformanceReader(FakeMetricsReader(events=[
            _error_event(workspace_id="u1", session_id="s1"),
            _error_event(workspace_id="u1", session_id="s2"),
        ]))

        top = reader.summary(_NOW - timedelta(hours=1)).top_error_sessions
        assert {s.key for s in top} == {"s1", "s2"}

    def test_ignores_missing_workspace_or_session(self):
        reader = PerformanceReader(FakeMetricsReader(events=[
            _error_event(workspace_id=None, session_id=None),
        ]))

        summary = reader.summary(_NOW - timedelta(hours=1))
        assert summary.top_error_workspaces == []
        assert summary.top_error_sessions == []

    def test_non_error_severity_excluded(self):
        reader = PerformanceReader(FakeMetricsReader(events=[
            {"severity": "warning", "step": "extraction", "message": "hm",
             "workspace_id": "u1", "session_id": "s1", "created_at": _NOW.isoformat()},
        ]))

        summary = reader.summary(_NOW - timedelta(hours=1))
        assert summary.top_error_workspaces == []
