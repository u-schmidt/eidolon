"""Scores actual pipeline output against an eval script's expectations."""

from __future__ import annotations

from core.interfaces import EmbedderProvider
from core.models import Fact, MemoryItem, TopicChange
from tools.eval.models import EndExpectation, FactExpectation, MemoryItemHint, TurnExpectation
from tools.eval.results import EndResult, FactCheckResult, MemoryItemMatch, TurnResult
from tools.eval.similarity import cosine_similarity


class EvalScorer:
    """Turns raw pipeline output into tier-1/tier-2 scored results.

    Tier 1 (topic_change, end-of-script facts) is a hard assert. Tier 2
    (topic labels, memory items) is embedding-similarity scored or
    reported, never asserted — see plan-phase2.md for why.
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
        fact_results = [
            FactCheckResult(expected=fe, found=self._fact_present(fe, actual_facts))
            for fe in expectation.facts
        ]
        return EndResult(
            fact_results=fact_results,
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

    @staticmethod
    def _fact_present(expected: FactExpectation, actual: list[Fact]) -> bool:
        return any(
            f.category == expected.category and f.key == expected.key and f.value == expected.value
            for f in actual
        )
