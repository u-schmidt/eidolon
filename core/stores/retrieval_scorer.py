"""Candidate scoring and ranking shared by all of SqliteStore's search_* methods.

Topic, cluster, memory-item, and cluster-candidate search all end in the
same tail: a weighted semantic + tag_overlap + person + temporal blend per
candidate, a sort/filter/slice-to-limit, and one batched
``retrieval_scoring_summary`` trace event. That tail lived inline in each
of the four methods; ``RetrievalScorer`` owns it once. The store still owns
the entity-specific *fetch* (SQL, vec-prefilter) and resolving each row's
raw semantic similarity — the scorer only turns those into ranked results.

Which scoring terms apply is entity-specific and passed in explicitly
(``include_tag_overlap``/``include_person``), so each entity's breakdown
dict keeps exactly the keys it always had: memory items carry no
``tag_overlap_*``; only topics carry ``person_weighted``; cluster
candidates add ``candidate_kind``. That shape is observable output (the
trace JSON is persisted and diffed across runs), not an implementation
detail.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypeVar

from core.interfaces import Tracer

# Fallback scoring weights when the caller doesn't supply an explicit
# `weights` override (e.g. from EidolonConfig.retrieval). Matches
# AssociativeRetriever's own defaults so a configured value means the same
# thing everywhere.
_DEFAULT_WEIGHTS = {"semantic": 0.4, "tag_overlap": 0.3, "person": 0.2, "temporal": 0.1}

# Recency half-life: a candidate's temporal score halves for every this-many
# days of age (see `_recency_score`). The *shape* of the decay is fixed here;
# how much it matters is the tunable "temporal" weight above.
_TEMPORAL_HALF_LIFE_DAYS = 30.0

# How many extra scored candidates beyond `limit` to persist in a
# retrieval_scoring_summary trace event, so near-misses just below the
# cutoff are visible when tuning weights — without persisting every
# positively-scored candidate (which scales with DB size, not turn count).
_TRACE_TOP_K_PADDING = 10

_T = TypeVar("_T")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    # strict=False keeps the historical behavior of truncating to the
    # shorter vector rather than raising on a length mismatch.
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _recency_score(timestamp: str | datetime | None, now: datetime, half_life_days: float) -> float:
    """Exponential recency decay in (0, 1]: ~1.0 for "just now", halving
    every ``half_life_days`` of age.

    Accepts either a raw SQLite ``CURRENT_TIMESTAMP`` string (e.g.
    "2026-07-20 16:57:00", stored in UTC — SqliteStore's shape) or a real
    ``datetime`` (Postgres/psycopg returns actual ``datetime`` objects for
    ``TIMESTAMPTZ`` columns, not strings — PostgresStore's shape). An
    unknown/unparseable timestamp scores 0.0 (no recency signal), and a
    future timestamp (host clock skew) is clamped to 1.0.
    """
    if not timestamp:
        return 0.0
    if isinstance(timestamp, datetime):
        ts = timestamp
    else:
        try:
            ts = datetime.fromisoformat(timestamp)
        except ValueError:
            return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    age_days = (now - ts).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


class RetrievalScorer:
    """Scores and ranks retrieval candidates for one search call.

    Constructed per search call, bound to that call's weights and reference
    time. The store calls ``score`` once per fetched candidate, collects the
    ``(final_score, result_object, breakdown)`` tuples, and hands them to
    ``finalize`` for sorting, slicing, and the trace event.
    """

    def __init__(self, weights: dict[str, float] | None = None, now: datetime | None = None) -> None:
        self._w = weights or _DEFAULT_WEIGHTS
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        self._now = now

    def score(
        self,
        *,
        candidate_type: str,
        candidate_id: int,
        semantic_raw: float,
        recency_timestamp: str | datetime | None,
        include_tag_overlap: bool = False,
        query_tags: list[str] | None = None,
        candidate_tags: list[str] | None = None,
        include_person: bool = False,
        person_match: bool = False,
        extra_breakdown: dict | None = None,
    ) -> tuple[float, dict]:
        """Score one candidate. ``semantic_raw`` is already resolved by the
        store (a KNN-precomputed similarity, a freshly computed cosine, or
        0.0 when there's no embedding). Returns ``(final_score, breakdown)``.

        The ``final = score + temporal if score > 0 else 0.0`` gate makes the
        temporal term a recency *re-ranking* signal, never a relevance source:
        a merely-recent candidate with no semantic/tag/person relevance stays
        at 0.0 and is dropped.
        """
        semantic_weighted = semantic_raw * self._w.get("semantic", _DEFAULT_WEIGHTS["semantic"])
        score = semantic_weighted

        tag_overlap_raw = tag_overlap_weighted = 0.0
        if include_tag_overlap and query_tags:
            overlap = len(set(query_tags) & set(candidate_tags or []))
            if overlap > 0:
                tag_overlap_raw = overlap / max(len(query_tags), 1)
                tag_overlap_weighted = tag_overlap_raw * self._w.get(
                    "tag_overlap", _DEFAULT_WEIGHTS["tag_overlap"]
                )
                score += tag_overlap_weighted

        person_weighted = 0.0
        if include_person and person_match:
            person_weighted = self._w.get("person", _DEFAULT_WEIGHTS["person"])
            score += person_weighted

        temporal_raw = _recency_score(recency_timestamp, self._now, _TEMPORAL_HALF_LIFE_DAYS)
        temporal_weighted = temporal_raw * self._w.get("temporal", _DEFAULT_WEIGHTS["temporal"])
        final_score = score + temporal_weighted if score > 0 else 0.0

        breakdown: dict = {"candidate_type": candidate_type}
        if extra_breakdown:
            breakdown.update(extra_breakdown)
        breakdown["candidate_id"] = candidate_id
        breakdown["semantic_raw"] = semantic_raw
        breakdown["semantic_weighted"] = semantic_weighted
        if include_tag_overlap:
            breakdown["tag_overlap_raw"] = tag_overlap_raw
            breakdown["tag_overlap_weighted"] = tag_overlap_weighted
        if include_person:
            breakdown["person_weighted"] = person_weighted
        breakdown["temporal_raw"] = temporal_raw
        breakdown["temporal_weighted"] = temporal_weighted
        breakdown["final_score"] = final_score
        return final_score, breakdown

    def finalize(
        self,
        *,
        scored: list[tuple[float, _T, dict]],
        candidate_type: str,
        limit: int,
        tracer: Tracer | None,
        request_id: str | None,
        extra_payload: dict | None = None,
    ) -> list[_T]:
        """Sort by score, drop non-positive, slice to ``limit``, and emit one
        batched ``retrieval_scoring_summary`` event (the top ``limit +
        _TRACE_TOP_K_PADDING`` candidates, each flagged whether it made the
        cut). Returns the ranked result objects.
        """
        scored.sort(key=lambda x: x[0], reverse=True)
        top_results = [(s, obj, b) for s, obj, b in scored if s > 0][:limit]
        # One keying rule for both the plain-id entities and cluster
        # candidates' (kind, id): non-candidate breakdowns have no
        # candidate_kind, so .get(...) yields (None, id) consistently.
        included_ids = {(b.get("candidate_kind"), b["candidate_id"]) for _, _, b in top_results}

        if tracer is not None:
            positively_scored = [(s, b) for s, _, b in scored if s > 0]
            top = positively_scored[: limit + _TRACE_TOP_K_PADDING]
            for _, breakdown in top:
                breakdown["included"] = (
                    breakdown.get("candidate_kind"),
                    breakdown["candidate_id"],
                ) in included_ids
            tracer.record(
                "retrieval_scoring_summary",
                {
                    "candidate_type": candidate_type,
                    **(extra_payload or {}),
                    "total_scored": len(positively_scored),
                    "included_count": len(included_ids),
                    "top_candidates": [b for _, b in top],
                },
                request_id=request_id,
            )

        return [obj for _, obj, _ in top_results]
