"""Tests for RetrievalScorer — the scoring/ranking/trace tail shared by
SqliteStore's search_* methods, exercised directly without any SQL."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.models import MemoryItem, MemoryItemBudget
from core.stores.retrieval_scorer import _TRACE_TOP_K_PADDING, RetrievalScorer, _recency_score
from tests.mocks import SpyTracer

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_WEIGHTS = {"semantic": 0.5, "tag_overlap": 0.3, "person": 0.2, "temporal": 0.1}
# A timestamp at exactly _NOW → recency 1.0, so the temporal term is a
# predictable 1.0 * temporal_weight.
_AT_NOW = "2026-01-01 00:00:00"


def _scorer(weights: dict | None = None) -> RetrievalScorer:
    return RetrievalScorer(weights or _WEIGHTS, _NOW)


class TestScore:
    def test_semantic_only_omits_tag_and_person_keys(self):
        score, breakdown = _scorer().score(
            candidate_type="memory_item", candidate_id=7, semantic_raw=1.0, recency_timestamp=None,
        )
        assert breakdown == {
            "candidate_type": "memory_item",
            "candidate_id": 7,
            "semantic_raw": 1.0,
            "semantic_weighted": pytest.approx(0.5),
            "temporal_raw": 0.0,
            "temporal_weighted": 0.0,
            "final_score": pytest.approx(0.5),
        }

    def test_tag_overlap_contributes_and_adds_its_keys(self):
        score, breakdown = _scorer().score(
            candidate_type="topic", candidate_id=1, semantic_raw=0.0, recency_timestamp=None,
            include_tag_overlap=True, query_tags=["a", "b"], candidate_tags=["a", "z"],
        )
        # overlap 1 of 2 query tags → 0.5 raw * 0.3 weight
        assert breakdown["tag_overlap_raw"] == pytest.approx(0.5)
        assert breakdown["tag_overlap_weighted"] == pytest.approx(0.15)
        assert score == pytest.approx(0.15)

    def test_tag_keys_present_but_zero_when_no_query_tags(self):
        _, breakdown = _scorer().score(
            candidate_type="topic", candidate_id=1, semantic_raw=1.0, recency_timestamp=None,
            include_tag_overlap=True, query_tags=None, candidate_tags=["a"],
        )
        assert breakdown["tag_overlap_raw"] == 0.0
        assert breakdown["tag_overlap_weighted"] == 0.0

    def test_person_bonus_added_only_on_match_but_key_always_present(self):
        _, matched = _scorer().score(
            candidate_type="topic", candidate_id=1, semantic_raw=1.0, recency_timestamp=None,
            include_person=True, person_match=True,
        )
        _, unmatched = _scorer().score(
            candidate_type="topic", candidate_id=2, semantic_raw=1.0, recency_timestamp=None,
            include_person=True, person_match=False,
        )
        assert matched["person_weighted"] == pytest.approx(0.2)
        assert unmatched["person_weighted"] == 0.0

    def test_temporal_is_a_reranking_signal_not_a_relevance_source(self):
        # No semantic/tag/person relevance → final stays 0.0 even though the
        # row is maximally recent (the `score > 0` gate).
        score, breakdown = _scorer().score(
            candidate_type="topic", candidate_id=1, semantic_raw=0.0, recency_timestamp=_AT_NOW,
        )
        assert breakdown["temporal_raw"] == pytest.approx(1.0)
        assert breakdown["temporal_weighted"] == pytest.approx(0.1)
        assert score == 0.0

    def test_temporal_added_when_there_is_relevance(self):
        score, _ = _scorer().score(
            candidate_type="topic", candidate_id=1, semantic_raw=1.0, recency_timestamp=_AT_NOW,
        )
        assert score == pytest.approx(0.6)  # 0.5 semantic + 0.1 temporal

    def test_extra_breakdown_inserts_candidate_kind_after_type(self):
        _, breakdown = _scorer().score(
            candidate_type="cluster_candidate", candidate_id=1, semantic_raw=1.0, recency_timestamp=None,
            include_tag_overlap=True, extra_breakdown={"candidate_kind": "topic"},
        )
        keys = list(breakdown.keys())
        assert keys[0] == "candidate_type"
        assert keys[1] == "candidate_kind"
        assert breakdown["candidate_kind"] == "topic"

    def test_weights_override_is_respected(self):
        _, breakdown = _scorer({"semantic": 1.0}).score(
            candidate_type="topic", candidate_id=1, semantic_raw=0.8, recency_timestamp=None,
        )
        assert breakdown["semantic_weighted"] == pytest.approx(0.8)


class TestFinalize:
    def _scored(self, *pairs: tuple[float, int]) -> list[tuple[float, dict, dict]]:
        """Build (score, result_obj, breakdown) triples from (score, id) pairs."""
        return [
            (score, {"id": cid}, {"candidate_type": "topic", "candidate_id": cid, "final_score": score})
            for score, cid in pairs
        ]

    def test_sorts_descending_drops_nonpositive_and_slices_to_limit(self):
        results = _scorer().finalize(
            scored=self._scored((0.1, 1), (0.9, 2), (0.0, 3), (0.5, 4)),
            candidate_type="topic", limit=2, tracer=None, request_id=None,
        )
        assert [r["id"] for r in results] == [2, 4]  # 0.0 dropped, sorted desc, top 2

    def test_no_tracer_still_returns_ranked_results(self):
        results = _scorer().finalize(
            scored=self._scored((0.3, 1)), candidate_type="topic", limit=5, tracer=None, request_id=None,
        )
        assert [r["id"] for r in results] == [1]

    def test_emits_one_event_with_included_flags(self):
        tracer = SpyTracer()
        _scorer().finalize(
            scored=self._scored((0.9, 1), (0.1, 2)), candidate_type="topic", limit=1,
            tracer=tracer, request_id="req1",
        )
        events = tracer.events_of_type("retrieval_scoring_summary")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert events[0]["request_id"] == "req1"
        assert payload["candidate_type"] == "topic"
        assert payload["total_scored"] == 2
        assert payload["included_count"] == 1
        flags = {c["candidate_id"]: c["included"] for c in payload["top_candidates"]}
        assert flags == {1: True, 2: False}

    def test_top_candidates_capped_at_limit_plus_padding(self):
        tracer = SpyTracer()
        many = self._scored(*[(1.0 - i * 0.001, i) for i in range(_TRACE_TOP_K_PADDING + 20)])
        _scorer().finalize(
            scored=many, candidate_type="topic", limit=1, tracer=tracer, request_id=None,
        )
        payload = tracer.events_of_type("retrieval_scoring_summary")[0]["payload"]
        assert len(payload["top_candidates"]) == 1 + _TRACE_TOP_K_PADDING

    def test_extra_payload_merged_into_event(self):
        tracer = SpyTracer()
        _scorer().finalize(
            scored=self._scored((0.5, 1)), candidate_type="cluster_candidate", limit=5,
            tracer=tracer, request_id=None, extra_payload={"topic_id": 42},
        )
        payload = tracer.events_of_type("retrieval_scoring_summary")[0]["payload"]
        assert payload["topic_id"] == 42

    def test_included_keying_uses_candidate_kind_when_present(self):
        # Two candidates share id 5 but differ by kind — both can't be
        # conflated; only the cluster one makes the limit-1 cut here.
        scored = [
            (0.9, {"kind": "cluster", "id": 5},
             {"candidate_type": "cluster_candidate", "candidate_kind": "cluster", "candidate_id": 5,
              "final_score": 0.9}),
            (0.1, {"kind": "topic", "id": 5},
             {"candidate_type": "cluster_candidate", "candidate_kind": "topic", "candidate_id": 5,
              "final_score": 0.1}),
        ]
        tracer = SpyTracer()
        _scorer().finalize(
            scored=scored, candidate_type="cluster_candidate", limit=1, tracer=tracer, request_id=None,
        )
        payload = tracer.events_of_type("retrieval_scoring_summary")[0]["payload"]
        flags = {c["candidate_kind"]: c["included"] for c in payload["top_candidates"]}
        assert flags == {"cluster": True, "topic": False}


class TestFinalizeTiered:
    def _item(self, item_id: int, item_type: str) -> MemoryItem:
        return MemoryItem(
            id=item_id, content=f"item {item_id}", item_type=item_type,
            source_peer="user", subject_peer="user", attribution_type="self_statement",
        )

    def _scored(self, *entries: tuple[float, int, str]) -> list[tuple[float, MemoryItem, dict]]:
        """Build (score, MemoryItem, breakdown) triples from (score, id, item_type) entries."""
        return [
            (score, self._item(item_id, item_type),
             {"candidate_type": "memory_item", "candidate_id": item_id, "final_score": score})
            for score, item_id, item_type in entries
        ]

    def test_reserved_tier_wins_its_slot_even_when_a_free_candidate_scores_higher(self):
        # A fact-typed item scores lower than an essence-typed item, but the
        # fact tier still fills from its own bucket, not a global top-K —
        # tiers aren't merged into one blind ranking.
        scored = self._scored((0.1, 1, "fact"), (0.9, 2, "essence"))
        budget = MemoryItemBudget(fact=1, surprise=0, feeling=0, free=1)

        results = RetrievalScorer().finalize_tiered(
            scored=scored, budget=budget, tracer=None, request_id=None,
        )

        assert {r.id for r in results} == {1, 2}

    def test_unfilled_reserved_tier_rolls_into_free_pool(self):
        # No surprise-type candidates exist — its reserved slot rolls over,
        # so the free pool effectively gets 1 extra slot (2 total instead
        # of the base 1), letting both essence candidates through.
        scored = self._scored((0.9, 1, "essence"), (0.8, 2, "essence"))
        budget = MemoryItemBudget(fact=0, surprise=1, feeling=0, free=1)

        results = RetrievalScorer().finalize_tiered(
            scored=scored, budget=budget, tracer=None, request_id=None,
        )

        assert {r.id for r in results} == {1, 2}

    def test_final_order_is_tier_order_not_score_order(self):
        scored = self._scored((0.1, 1, "fact"), (0.9, 2, "feeling"), (0.5, 3, "surprise"))
        budget = MemoryItemBudget(fact=1, surprise=1, feeling=1, free=0)

        results = RetrievalScorer().finalize_tiered(
            scored=scored, budget=budget, tracer=None, request_id=None,
        )

        assert [r.id for r in results] == [1, 3, 2]  # fact, surprise, feeling

    def test_excess_within_a_tier_falls_through_to_free_pool(self):
        # Two fact candidates but only 1 reserved fact slot — the loser
        # isn't dropped outright, it competes for the free pool like any
        # other leftover candidate.
        scored = self._scored((0.9, 1, "fact"), (0.8, 2, "fact"))
        budget = MemoryItemBudget(fact=1, surprise=0, feeling=0, free=1)

        results = RetrievalScorer().finalize_tiered(
            scored=scored, budget=budget, tracer=None, request_id=None,
        )

        assert [r.id for r in results] == [1, 2]

    def test_emits_one_event_with_tier_included_counts(self):
        tracer = SpyTracer()
        scored = self._scored((0.9, 1, "fact"), (0.8, 2, "essence"), (0.1, 3, "surprise"))
        budget = MemoryItemBudget(fact=1, surprise=1, feeling=0, free=1)

        RetrievalScorer().finalize_tiered(
            scored=scored, budget=budget, tracer=tracer, request_id="req1",
        )

        events = tracer.events_of_type("retrieval_scoring_summary")
        assert len(events) == 1
        payload = events[0]["payload"]
        assert events[0]["request_id"] == "req1"
        assert payload["candidate_type"] == "memory_item"
        assert payload["total_scored"] == 3
        assert payload["included_count"] == 3
        assert payload["tier_included_counts"] == {"fact": 1, "surprise": 1, "feeling": 0, "free": 1}

    def test_included_flags_reflect_final_selection(self):
        tracer = SpyTracer()
        scored = self._scored((0.9, 1, "fact"), (0.8, 2, "fact"))
        budget = MemoryItemBudget(fact=1, surprise=0, feeling=0, free=0)

        RetrievalScorer().finalize_tiered(
            scored=scored, budget=budget, tracer=tracer, request_id=None,
        )

        payload = tracer.events_of_type("retrieval_scoring_summary")[0]["payload"]
        flags = {c["candidate_id"]: c["included"] for c in payload["top_candidates"]}
        assert flags == {1: True, 2: False}


class TestRecencyScore:
    """Moved from tests/core/test_sqlite_store.py — this tests
    _recency_score directly, which lives in this module, not SqliteStore.
    """

    _NOW = datetime(2026, 1, 1, tzinfo=UTC)

    def test_just_now_scores_one(self):
        assert _recency_score("2026-01-01 00:00:00", self._NOW, 30.0) == pytest.approx(1.0)

    def test_one_half_life_scores_half(self):
        # 30 days before _NOW, half_life 30 -> 0.5
        assert _recency_score("2025-12-02 00:00:00", self._NOW, 30.0) == pytest.approx(0.5)

    def test_two_half_lives_scores_quarter(self):
        # 60 days before _NOW, half_life 30 -> 0.25
        assert _recency_score("2025-11-02 00:00:00", self._NOW, 30.0) == pytest.approx(0.25)

    def test_none_scores_zero(self):
        assert _recency_score(None, self._NOW, 30.0) == 0.0

    def test_unparseable_scores_zero(self):
        assert _recency_score("not-a-date", self._NOW, 30.0) == 0.0

    def test_future_timestamp_clamps_to_one(self):
        assert _recency_score("2027-01-01 00:00:00", self._NOW, 30.0) == 1.0

    def test_naive_now_is_treated_as_utc(self):
        # A naive `now` must not raise on the aware/naive subtraction.
        assert _recency_score("2020-01-01 00:00:00", datetime(2026, 1, 1), 30.0) < 0.01

    def test_accepts_a_real_datetime_not_just_a_sqlite_string(self):
        """PostgresStore passes real tz-aware datetime objects (psycopg's
        TIMESTAMPTZ representation) instead of SQLite's timestamp
        strings — both must produce the same score for the same instant."""
        as_datetime = _recency_score(datetime(2026, 1, 1, tzinfo=UTC), self._NOW, 30.0)
        as_string = _recency_score("2026-01-01 00:00:00", self._NOW, 30.0)
        assert as_datetime == pytest.approx(as_string)

    def test_naive_datetime_is_treated_as_utc(self):
        # A tz-naive datetime (shouldn't normally happen, but must not raise).
        assert _recency_score(datetime(2020, 1, 1), self._NOW, 30.0) < 0.01
