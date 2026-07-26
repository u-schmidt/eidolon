"""Composes the operational-metrics reader into an aggregate
performance/reliability view — latency percentiles, error rates,
throughput, and top error sources over a time window.

Needs no debug tracing — sourced entirely from the always-on
``events``/``llm_calls``/``embedding_calls`` tables (``MetricsReader``'s
new bulk ``*_since`` methods), unlike the turn-detail dashboard's rich
mode. Lives in ``tools/``, not ``core/``, for the same reason
``tools/dashboard/turn_reader.py`` does — read + aggregation glue for a
dashboard view, not core orchestration logic.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from core.interfaces import MetricsReader

_TOP_ERROR_SOURCES_LIMIT = 10


def _percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile — no interpolation. Keeps both backends
    on the same code path since SQLite has no native percentile
    aggregate (see ``core/stores/retrieval_scorer.py`` for the same
    "centralize the math in Python once" rationale)."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = max(0, min(n - 1, math.ceil(p * n) - 1))
    return sorted_values[idx]


def _parse_created_at(value: object) -> datetime | None:
    """``created_at`` is a raw string for SqliteMetricsReader rows and a
    real ``datetime`` for PostgresMetricsReader rows (see
    ``core/stores/retrieval_scorer.py``'s ``_recency_score`` for the
    same dual-shape handling)."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


@dataclass
class CallTypeStats:
    call_type: str
    count: int
    error_count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float


@dataclass
class EmbeddingPurposeStats:
    purpose: str
    count: int
    error_count: int
    p50_ms: float
    p95_ms: float


@dataclass
class HourlyBucket:
    hour: str  # "2026-07-26 14:00"
    llm_call_count: int


@dataclass
class TopErrorSource:
    key: str
    error_count: int


@dataclass
class PerformanceSummary:
    since: datetime
    call_type_stats: list[CallTypeStats] = field(default_factory=list)
    embedding_stats: list[EmbeddingPurposeStats] = field(default_factory=list)
    hourly_throughput: list[HourlyBucket] = field(default_factory=list)
    top_error_peers: list[TopErrorSource] = field(default_factory=list)
    top_error_sessions: list[TopErrorSource] = field(default_factory=list)


class PerformanceReader:
    """Read-only aggregation over ``MetricsReader`` for the dashboard's
    performance/reliability page (see ``tools/dashboard/app.py``)."""

    def __init__(self, metrics_reader: MetricsReader) -> None:
        self._metrics_reader = metrics_reader

    def summary(self, since: datetime) -> PerformanceSummary:
        llm_calls = self._metrics_reader.llm_calls_since(since)
        embedding_calls = self._metrics_reader.embedding_calls_since(since)
        events = self._metrics_reader.events_since(since)

        return PerformanceSummary(
            since=since,
            call_type_stats=self._call_type_stats(llm_calls),
            embedding_stats=self._embedding_stats(embedding_calls),
            hourly_throughput=self._hourly_throughput(llm_calls),
            top_error_peers=self._top_error_sources(events, "peer_id"),
            top_error_sessions=self._top_error_sources(events, "session_id"),
        )

    @staticmethod
    def _call_type_stats(llm_calls: list[dict]) -> list[CallTypeStats]:
        by_type: dict[str, list[dict]] = {}
        for call in llm_calls:
            by_type.setdefault(call["call_type"], []).append(call)

        stats = []
        for call_type, calls in by_type.items():
            durations = sorted(c["duration_ms"] for c in calls)
            stats.append(CallTypeStats(
                call_type=call_type,
                count=len(calls),
                error_count=sum(1 for c in calls if not c["succeeded"]),
                p50_ms=_percentile(durations, 0.50),
                p95_ms=_percentile(durations, 0.95),
                p99_ms=_percentile(durations, 0.99),
            ))
        stats.sort(key=lambda s: s.count, reverse=True)
        return stats

    @staticmethod
    def _embedding_stats(embedding_calls: list[dict]) -> list[EmbeddingPurposeStats]:
        by_purpose: dict[str, list[dict]] = {}
        for call in embedding_calls:
            by_purpose.setdefault(call["purpose"], []).append(call)

        stats = []
        for purpose, calls in by_purpose.items():
            durations = sorted(c["duration_ms"] for c in calls)
            stats.append(EmbeddingPurposeStats(
                purpose=purpose,
                count=len(calls),
                error_count=sum(1 for c in calls if not c["succeeded"]),
                p50_ms=_percentile(durations, 0.50),
                p95_ms=_percentile(durations, 0.95),
            ))
        stats.sort(key=lambda s: s.count, reverse=True)
        return stats

    @staticmethod
    def _hourly_throughput(llm_calls: list[dict]) -> list[HourlyBucket]:
        counts: Counter[str] = Counter()
        for call in llm_calls:
            ts = _parse_created_at(call["created_at"])
            if ts is None:
                continue
            hour = ts.replace(minute=0, second=0, microsecond=0)
            counts[hour.strftime("%Y-%m-%d %H:00")] += 1
        return [HourlyBucket(hour=hour, llm_call_count=count) for hour, count in sorted(counts.items())]

    @staticmethod
    def _top_error_sources(events: list[dict], key: str) -> list[TopErrorSource]:
        counts: Counter[str] = Counter(
            e[key] for e in events if e["severity"] == "error" and e[key]
        )
        return [
            TopErrorSource(key=k, error_count=count)
            for k, count in counts.most_common(_TOP_ERROR_SOURCES_LIMIT)
        ]
