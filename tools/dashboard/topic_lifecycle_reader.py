"""Composes the storage + trace reader into an aggregate topic-lifecycle
view — how long topics live before closing (too short = over-eager
NEW_TOPIC detection?) and how many closes come from the stale-topic sweep
versus a natural close.

Unlike ``tools/dashboard/prompt_quality_reader.py``/
``retrieval_quality_reader.py``, the lifespan half of this page needs no
debug tracing at all — ``topics.created_at``/``closed_at``/
``message_count`` are always-on. Only the stale-sweep split needs the
debug-gated ``stale_topic_cleanup`` trace event. Lives in ``tools/``, not
``core/``, for the same reason the other dashboard readers do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from core.interfaces import StorageProvider, TraceReader


def _percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile — no interpolation. Same approach
    ``tools/dashboard/performance_reader.py`` uses, duplicated here per
    this codebase's established preference for small independent copies
    over cross-module import of a `_`-prefixed helper."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = max(0, min(n - 1, math.ceil(p * n) - 1))
    return sorted_values[idx]


def _parse_timestamp(value: object) -> datetime | None:
    """``created_at``/``closed_at`` are raw strings for SqliteStore rows
    and real ``datetime`` objects for PostgresStore rows — same dual-shape
    handling as ``performance_reader.py``'s ``_parse_created_at``."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


@dataclass
class TopicLifespanStats:
    total_closed: int
    avg_lifespan_hours: float
    p50_lifespan_hours: float
    p95_lifespan_hours: float
    avg_message_count: float


@dataclass
class ShortLivedTopic:
    topic_id: int
    label: str
    lifespan_hours: float
    message_count: int


@dataclass
class StaleSweepStats:
    total_closed: int
    stale_swept: int
    natural: int


@dataclass
class TopicLifecycleSummary:
    since: datetime
    lifespan: TopicLifespanStats
    stale_sweep: StaleSweepStats
    shortest_lived: list[ShortLivedTopic] = field(default_factory=list)


class TopicLifecycleReader:
    """Read-only aggregation over ``StorageProvider``/``TraceReader`` for
    the dashboard's topic-lifecycle page (see ``tools/dashboard/app.py``).
    """

    def __init__(self, trace_reader: TraceReader, storage: StorageProvider) -> None:
        self._trace_reader = trace_reader
        self._storage = storage

    def summary(self, since: datetime, top_n: int = 15) -> TopicLifecycleSummary:
        closed_topics = self._storage.get_closed_topics_since(since)
        sweep_events = self._trace_reader.events_of_type_since("stale_topic_cleanup", since)
        swept_ids = {
            tid for e in sweep_events for tid in e["payload"].get("closed_topic_ids", [])
        }

        with_lifespan = [
            (t, lifespan)
            for t in closed_topics
            if (lifespan := self._lifespan_hours(t)) is not None
        ]

        return TopicLifecycleSummary(
            since=since,
            lifespan=self._lifespan_stats(closed_topics, with_lifespan),
            stale_sweep=self._stale_sweep_stats(closed_topics, swept_ids),
            shortest_lived=self._shortest_lived(with_lifespan, top_n),
        )

    @staticmethod
    def _lifespan_hours(topic: dict) -> float | None:
        created = _parse_timestamp(topic["created_at"])
        closed = _parse_timestamp(topic["closed_at"])
        if created is None or closed is None:
            return None
        return max(0.0, (closed - created).total_seconds() / 3600.0)

    @staticmethod
    def _lifespan_stats(
        closed_topics: list[dict], with_lifespan: list[tuple[dict, float]]
    ) -> TopicLifespanStats:
        lifespans = sorted(lifespan for _, lifespan in with_lifespan)
        message_counts = [t["message_count"] for t in closed_topics]
        return TopicLifespanStats(
            total_closed=len(closed_topics),
            avg_lifespan_hours=sum(lifespans) / len(lifespans) if lifespans else 0.0,
            p50_lifespan_hours=_percentile(lifespans, 0.50),
            p95_lifespan_hours=_percentile(lifespans, 0.95),
            avg_message_count=sum(message_counts) / len(message_counts) if message_counts else 0.0,
        )

    @staticmethod
    def _stale_sweep_stats(closed_topics: list[dict], swept_ids: set[int]) -> StaleSweepStats:
        total = len(closed_topics)
        swept = sum(1 for t in closed_topics if t["id"] in swept_ids)
        return StaleSweepStats(total_closed=total, stale_swept=swept, natural=total - swept)

    @staticmethod
    def _shortest_lived(with_lifespan: list[tuple[dict, float]], top_n: int) -> list[ShortLivedTopic]:
        ordered = sorted(with_lifespan, key=lambda pair: pair[1])
        return [
            ShortLivedTopic(
                topic_id=t["id"], label=t["label"], lifespan_hours=lifespan, message_count=t["message_count"],
            )
            for t, lifespan in ordered[:top_n]
        ]
