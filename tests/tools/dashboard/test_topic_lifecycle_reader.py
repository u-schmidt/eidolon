"""Unit tests for TopicLifecycleReader — the aggregation layer behind the
dashboard's topic-lifecycle page. Uses a bespoke FakeStorage (not
MockStorage — real created_at/closed_at/message_count timestamp
semantics aren't modeled there, see tests/mocks.py's
get_closed_topics_since docstring; real filtering correctness is covered
by tests/core/test_storage_provider_contract.py) plus a simple in-memory
fake for TraceReader; these tests only exercise the lifespan/stale-sweep
aggregation math."""

from __future__ import annotations

from datetime import UTC, datetime

from tools.dashboard.topic_lifecycle_reader import TopicLifecycleReader

_NOW = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)


class FakeStorage:
    def __init__(self, closed_topics: list[dict] | None = None) -> None:
        self._closed_topics = closed_topics or []

    def get_closed_topics_since(self, since: datetime) -> list[dict]:
        return self._closed_topics


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


def _topic(topic_id: int, label: str, created_at: str, closed_at: str, message_count: int) -> dict:
    return {
        "id": topic_id, "peer_id": "u1", "label": label,
        "created_at": created_at, "closed_at": closed_at, "message_count": message_count,
    }


class TestLifespanStats:
    def test_computes_avg_and_percentiles(self):
        closed_topics = [
            _topic(1, "A", "2026-07-28T10:00:00", "2026-07-28T11:00:00", 3),  # 1h
            _topic(2, "B", "2026-07-28T10:00:00", "2026-07-28T13:00:00", 5),  # 3h
            _topic(3, "C", "2026-07-28T10:00:00", "2026-07-28T20:00:00", 10),  # 10h
        ]
        reader = TopicLifecycleReader(FakeTraceReader(), FakeStorage(closed_topics))

        stats = reader.summary(_NOW).lifespan

        assert stats.total_closed == 3
        assert stats.avg_lifespan_hours == (1 + 3 + 10) / 3
        assert stats.avg_message_count == (3 + 5 + 10) / 3

    def test_empty_closed_topics(self):
        reader = TopicLifecycleReader(FakeTraceReader(), FakeStorage([]))

        stats = reader.summary(_NOW).lifespan

        assert stats.total_closed == 0
        assert stats.avg_lifespan_hours == 0.0

    def test_topic_with_missing_timestamp_excluded_from_lifespan_but_not_total(self):
        closed_topics = [
            _topic(1, "Good", "2026-07-28T10:00:00", "2026-07-28T11:00:00", 2),
            {"id": 2, "peer_id": "u1", "label": "Bad", "created_at": None, "closed_at": "2026-07-28T11:00:00", "message_count": 1},
        ]
        reader = TopicLifecycleReader(FakeTraceReader(), FakeStorage(closed_topics))

        stats = reader.summary(_NOW).lifespan

        assert stats.total_closed == 2
        assert stats.avg_lifespan_hours == 1.0  # only the "Good" topic contributes


class TestShortestLived:
    def test_orders_ascending_by_lifespan(self):
        closed_topics = [
            _topic(1, "Long", "2026-07-28T00:00:00", "2026-07-28T10:00:00", 5),  # 10h
            _topic(2, "Short", "2026-07-28T00:00:00", "2026-07-28T00:30:00", 1),  # 0.5h
        ]
        reader = TopicLifecycleReader(FakeTraceReader(), FakeStorage(closed_topics))

        shortest = reader.summary(_NOW).shortest_lived

        assert [s.topic_id for s in shortest] == [2, 1]
        assert shortest[0].lifespan_hours == 0.5


class TestStaleSweep:
    def test_classifies_swept_vs_natural(self):
        closed_topics = [
            _topic(1, "Swept", "2026-07-28T10:00:00", "2026-07-28T11:00:00", 2),
            _topic(2, "Natural", "2026-07-28T10:00:00", "2026-07-28T11:00:00", 2),
        ]
        sweep_events = [{"payload": {"closed_topic_ids": [1]}}]
        reader = TopicLifecycleReader(
            FakeTraceReader({"stale_topic_cleanup": sweep_events}), FakeStorage(closed_topics),
        )

        stats = reader.summary(_NOW).stale_sweep

        assert stats.total_closed == 2
        assert stats.stale_swept == 1
        assert stats.natural == 1

    def test_all_natural_when_no_sweep_events(self):
        closed_topics = [_topic(1, "Natural", "2026-07-28T10:00:00", "2026-07-28T11:00:00", 2)]
        reader = TopicLifecycleReader(FakeTraceReader(), FakeStorage(closed_topics))

        stats = reader.summary(_NOW).stale_sweep

        assert stats.stale_swept == 0
        assert stats.natural == 1


class TestEmptyWindow:
    def test_no_closed_topics_returns_empty_summary(self):
        reader = TopicLifecycleReader(FakeTraceReader(), FakeStorage([]))

        summary = reader.summary(_NOW)

        assert summary.lifespan.total_closed == 0
        assert summary.shortest_lived == []
        assert summary.stale_sweep.total_closed == 0
