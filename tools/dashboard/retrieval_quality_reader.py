"""Composes the trace reader + storage into an aggregate retrieval-quality
view — which topics/clusters/memory items get retrieved most/least often,
whether the configured ``retrieval.*`` weights actually influence scoring,
near-miss candidates just below the cutoff, and which facts get injected
most often.

Needs ``debug.enabled`` trace data (``retrieval_scoring_summary``/
``context_retrieval`` events) — unlike
``tools/dashboard/performance_reader.py``, this is *not* an always-on
page. Lives in ``tools/``, not ``core/``, for the same reason
``tools/dashboard/turn_reader.py``/``performance_reader.py`` do — read +
aggregation glue for a dashboard view, not core orchestration logic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from core.interfaces import StorageProvider, TraceReader

_PREVIEW_MAX_CHARS = 80

# First-pass heuristic: an excluded candidate scoring within this many
# points of the lowest included candidate in the same event counts as a
# "near miss" — a plausible signal that a limit/threshold is too strict.
# Tunable later; not derived from any real data yet.
_NEAR_MISS_MARGIN = 0.05


def _preview(text: str) -> str:
    text = text or ""
    return text if len(text) <= _PREVIEW_MAX_CHARS else text[:_PREVIEW_MAX_CHARS] + "…"


def _candidate_kind(breakdown: dict) -> str:
    """The candidate's real entity kind. ``candidate_kind`` disambiguates
    ``cluster_candidate`` rows (which score both topics and clusters in
    one call — see ``core/stores/retrieval_scorer.py``'s ``finalize()``);
    every other row's own ``candidate_type`` is the kind directly. Same
    rule ``TurnReader._annotate_candidate_labels`` already uses."""
    return breakdown.get("candidate_kind") or breakdown["candidate_type"]


@dataclass
class CandidateFrequency:
    kind: str
    candidate_id: int
    label: str | None
    included_count: int
    scored_count: int


@dataclass
class ScoreComponentAverage:
    candidate_type: str
    count: int
    avg_semantic_share: float
    avg_tag_overlap_share: float | None  # None when this type never carries it
    avg_person_share: float | None
    avg_temporal_share: float


@dataclass
class NearMiss:
    kind: str
    candidate_id: int
    label: str | None
    final_score: float
    near_miss_count: int


@dataclass
class InjectedFact:
    category: str
    key: str
    value: str
    injection_count: int


@dataclass
class RetrievalQualitySummary:
    since: datetime
    total_events: int
    top_retrieved: list[CandidateFrequency] = field(default_factory=list)
    never_included: list[CandidateFrequency] = field(default_factory=list)
    score_components: list[ScoreComponentAverage] = field(default_factory=list)
    near_misses: list[NearMiss] = field(default_factory=list)
    top_injected_facts: list[InjectedFact] = field(default_factory=list)


class RetrievalQualityReader:
    """Read-only aggregation over ``TraceReader``/``StorageProvider`` for
    the dashboard's retrieval-quality page (see ``tools/dashboard/app.py``).
    """

    def __init__(self, trace_reader: TraceReader, storage: StorageProvider) -> None:
        self._trace_reader = trace_reader
        self._storage = storage

    def summary(self, since: datetime, top_n: int = 15) -> RetrievalQualitySummary:
        scoring_events = self._trace_reader.events_of_type_since("retrieval_scoring_summary", since)
        context_events = self._trace_reader.events_of_type_since("context_retrieval", since)
        counts = self._tally(scoring_events)

        return RetrievalQualitySummary(
            since=since,
            total_events=len(scoring_events),
            top_retrieved=self._top_retrieved(counts, top_n),
            never_included=self._never_included(counts, top_n),
            score_components=self._score_components(scoring_events),
            near_misses=self._near_misses(scoring_events, top_n),
            top_injected_facts=self._top_injected_facts(context_events, top_n),
        )

    @staticmethod
    def _tally(scoring_events: list[dict]) -> dict[tuple[str, int], list[int]]:
        """``(kind, candidate_id) -> [included_count, scored_count]``
        across every ``top_candidates`` row in every event."""
        counts: dict[tuple[str, int], list[int]] = {}
        for event in scoring_events:
            for c in event["payload"].get("top_candidates", []):
                key = (_candidate_kind(c), c["candidate_id"])
                entry = counts.setdefault(key, [0, 0])
                entry[1] += 1
                if c.get("included"):
                    entry[0] += 1
        return counts

    def _top_retrieved(
        self, counts: dict[tuple[str, int], list[int]], top_n: int
    ) -> list[CandidateFrequency]:
        ranked = sorted(
            ((k, v) for k, v in counts.items() if v[0] > 0),
            key=lambda kv: kv[1][0], reverse=True,
        )[:top_n]
        return [self._to_frequency(kind, cid, inc, scr) for (kind, cid), (inc, scr) in ranked]

    def _never_included(
        self, counts: dict[tuple[str, int], list[int]], top_n: int
    ) -> list[CandidateFrequency]:
        """Scored at least once but never made the cut — the decay-analog
        for topics/clusters that the original question catalogue asked
        for (not just facts)."""
        ranked = sorted(
            ((k, v) for k, v in counts.items() if v[0] == 0),
            key=lambda kv: kv[1][1], reverse=True,
        )[:top_n]
        return [self._to_frequency(kind, cid, inc, scr) for (kind, cid), (inc, scr) in ranked]

    def _to_frequency(
        self, kind: str, candidate_id: int, included_count: int, scored_count: int
    ) -> CandidateFrequency:
        return CandidateFrequency(
            kind=kind,
            candidate_id=candidate_id,
            label=self._resolve_label(kind, candidate_id),
            included_count=included_count,
            scored_count=scored_count,
        )

    @staticmethod
    def _score_components(scoring_events: list[dict]) -> list[ScoreComponentAverage]:
        """Does ``semantic`` actually dominate, or do ``tag_overlap``/
        ``person``/``temporal`` meaningfully contribute — validates
        whether the configured ``retrieval.*`` weights do anything.
        Grouped by raw ``candidate_type`` (not ``kind``): which fields a
        row carries depends on which search call produced it, not what
        it scored as, so ``cluster_candidate`` stays its own bucket.
        """
        by_type: dict[str, list[dict]] = {}
        for event in scoring_events:
            for c in event["payload"].get("top_candidates", []):
                by_type.setdefault(c["candidate_type"], []).append(c)

        result = [
            ScoreComponentAverage(
                candidate_type=candidate_type,
                count=len(rows),
                avg_semantic_share=RetrievalQualityReader._share_average(rows, "semantic_weighted"),
                avg_tag_overlap_share=RetrievalQualityReader._optional_share_average(rows, "tag_overlap_weighted"),
                avg_person_share=RetrievalQualityReader._optional_share_average(rows, "person_weighted"),
                avg_temporal_share=RetrievalQualityReader._share_average(rows, "temporal_weighted"),
            )
            for candidate_type, rows in by_type.items()
        ]
        result.sort(key=lambda s: s.count, reverse=True)
        return result

    @staticmethod
    def _share_average(rows: list[dict], weighted_field: str) -> float:
        """Average ``weighted_field / final_score`` across rows that
        carry the field. Every ``top_candidates`` row always carries
        ``semantic_weighted``/``temporal_weighted``
        (``RetrievalScorer.score()`` sets both unconditionally) and only
        ever has ``final_score > 0`` (``finalize()``/``finalize_tiered()``
        already filter to positively-scored rows before persisting) — but
        this reads back historical trace data that could predate either
        invariant, so both are guarded rather than assumed."""
        shares = [
            row[weighted_field] / row["final_score"]
            for row in rows
            if weighted_field in row and row.get("final_score", 0) > 0
        ]
        return sum(shares) / len(shares) if shares else 0.0

    @staticmethod
    def _optional_share_average(rows: list[dict], weighted_field: str) -> float | None:
        """Like ``_share_average``, but ``None`` (not ``0.0``) when
        ``weighted_field`` never appears in any row — e.g.
        ``tag_overlap_weighted`` for ``memory_item`` rows,
        ``person_weighted`` for anything but topics (see
        ``core/stores/retrieval_scorer.py``'s module docstring) — so the
        page can distinguish "doesn't apply to this type" from "applies
        but contributed nothing"."""
        if not any(weighted_field in row for row in rows):
            return None
        return RetrievalQualityReader._share_average(rows, weighted_field)

    def _near_misses(self, scoring_events: list[dict], top_n: int) -> list[NearMiss]:
        counts: Counter[tuple[str, int]] = Counter()
        best_score: dict[tuple[str, int], float] = {}
        for event in scoring_events:
            candidates = event["payload"].get("top_candidates", [])
            included_scores = [c["final_score"] for c in candidates if c.get("included")]
            if not included_scores:
                continue
            cutoff = min(included_scores)
            for c in candidates:
                if c.get("included"):
                    continue
                if c["final_score"] >= cutoff - _NEAR_MISS_MARGIN:
                    key = (_candidate_kind(c), c["candidate_id"])
                    counts[key] += 1
                    best_score[key] = max(best_score.get(key, 0.0), c["final_score"])

        ranked = counts.most_common(top_n)
        return [
            NearMiss(
                kind=kind,
                candidate_id=cid,
                label=self._resolve_label(kind, cid),
                final_score=best_score[(kind, cid)],
                near_miss_count=count,
            )
            for (kind, cid), count in ranked
        ]

    @staticmethod
    def _top_injected_facts(context_events: list[dict], top_n: int) -> list[InjectedFact]:
        counts: Counter[tuple[str, str]] = Counter()
        latest_value: dict[tuple[str, str], str] = {}
        for event in context_events:  # oldest first — later events overwrite the value
            for f in event["payload"].get("facts", []):
                key = (f["category"], f["key"])
                counts[key] += 1
                latest_value[key] = f["value"]

        ranked = counts.most_common(top_n)
        return [
            InjectedFact(category=cat, key=k, value=latest_value[(cat, k)], injection_count=count)
            for (cat, k), count in ranked
        ]

    def _resolve_label(self, kind: str, candidate_id: int) -> str | None:
        if kind == "topic":
            topic = self._storage.get_topic(candidate_id)
            return topic["label"] if topic else None
        if kind == "cluster":
            cluster = self._storage.get_cluster(candidate_id)
            return cluster["label"] if cluster else None
        if kind == "memory_item":
            item = self._storage.get_memory_item(candidate_id)
            return _preview(item["content"]) if item else None
        return None
