"""Scores actual pipeline output against an eval script's expectations."""

from __future__ import annotations

from core.interfaces import EmbedderProvider
from core.models import Fact, MemoryItem, TopicChange
from tools.eval.models import EndExpectation, FactExpectation, MemoryItemHint, TurnExpectation
from tools.eval.results import EndResult, FactMatch, MemoryItemMatch, TurnResult
from tools.eval.similarity import cosine_similarity


def _fact_text(fact: Fact | FactExpectation) -> str:
    """Render a Fact or FactExpectation as one string for embedding —
    both share the same category/key/value shape."""
    return f"{fact.category}: {fact.key} = {fact.value}"


class EvalScorer:
    """Turns raw pipeline output into tier-1/tier-2 scored results.

    Tier 1 (topic_change only) is a hard assert. Tier 2 (topic labels,
    memory items, end-of-script facts) is embedding-similarity scored or
    reported, never asserted — see plan-phase2.md for why. Facts were
    tier 1 (exact category/key/value match) originally, but Fact is never
    consumed programmatically anywhere in core — only injected into
    prompts — so a model choosing a different-but-equally-good key
    ("finance/budget_item" vs. an expected "preference/bike_budget") was
    scored as a failure despite being a fine result. Scored like
    memory_items instead.
    """

    def __init__(self, embedder: EmbedderProvider) -> None:
        self._embedder = embedder

    def score_turn(
        self,
        turn_number: int,
        user_message: str,
        expectation: TurnExpectation,
        actual_topic_change: TopicChange,
        actual_items: list[MemoryItem],
    ) -> TurnResult:
        topic_change_pass = (
            actual_topic_change.type == expectation.topic_change
            if expectation.topic_change is not None
            else None
        )
        matches, unmatched = self._match_memory_items(expectation.memory_items, actual_items)
        return TurnResult(
            turn_number=turn_number,
            user_message=user_message,
            topic_change_actual=actual_topic_change.type,
            topic_change_expected=expectation.topic_change,
            topic_change_pass=topic_change_pass,
            topic_label_hint=expectation.topic_label_hint,
            topic_label_actual=actual_topic_change.label,
            memory_item_matches=matches,
            unmatched_actual_items=unmatched,
        )

    def score_end(
        self,
        expectation: EndExpectation,
        actual_facts: list[Fact],
        actual_topics: list[dict],
    ) -> EndResult:
        return EndResult(
            fact_matches=self._match_facts(expectation.facts, actual_facts),
            actual_facts=actual_facts,
            expected_topics=expectation.topics,
            actual_topics=actual_topics,
        )

    def _match_memory_items(
        self, hints: list[MemoryItemHint], actual_items: list[MemoryItem]
    ) -> tuple[list[MemoryItemMatch], list[MemoryItem]]:
        matches: list[MemoryItemMatch] = []
        claimed_ids: set[int | None] = set()

        for hint in hints:
            best_score = 0.0
            best_item: MemoryItem | None = None
            if actual_items:
                hint_embedding = self._embedder.embed(hint.content_hint)
                for item in actual_items:
                    if item.embedding is None:
                        continue
                    score = cosine_similarity(hint_embedding, item.embedding)
                    if score > best_score:
                        best_score = score
                        best_item = item
            if best_item is not None:
                claimed_ids.add(best_item.id)
            matches.append(
                MemoryItemMatch(
                    hint=hint,
                    score=best_score,
                    matched_item=best_item,
                    type_match=best_item is not None and best_item.item_type == hint.item_type,
                )
            )

        unmatched = [item for item in actual_items if item.id not in claimed_ids]
        return matches, unmatched

    def _match_facts(self, expected: list[FactExpectation], actual: list[Fact]) -> list[FactMatch]:
        # Facts carry no stored embedding (unlike memory items) — both
        # sides are embedded fresh here.
        actual_embedded = [(fact, self._embedder.embed(_fact_text(fact))) for fact in actual]

        matches: list[FactMatch] = []
        for fe in expected:
            best_score = 0.0
            best_fact: Fact | None = None
            if actual_embedded:
                expected_embedding = self._embedder.embed(_fact_text(fe))
                for fact, fact_embedding in actual_embedded:
                    score = cosine_similarity(expected_embedding, fact_embedding)
                    if score > best_score:
                        best_score = score
                        best_fact = fact
            matches.append(FactMatch(expected=fe, score=best_score, matched_fact=best_fact))
        return matches
