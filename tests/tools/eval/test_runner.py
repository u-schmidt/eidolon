"""Tests for EvalRunner."""

from __future__ import annotations

from core import Eidolon
from core.models import Fact
from tests.core.test_eidolon import MockEmbedder, MockStorage
from tools.eval.models import EndExpectation, EvalScript, EvalTurn, FactExpectation, MemoryItemHint, TurnExpectation
from tools.eval.runner import EvalRunner
from tools.eval.scorer import EvalScorer


class QueuedLLM:
    """Returns detection responses from a queue, one per call — used to
    drive a specific CONTINUE/NEW_TOPIC sequence deterministically,
    unlike MockLLM's single constant response."""

    def __init__(self, detection_responses: list[str], extraction_response: str = "[]") -> None:
        self._detection_responses = list(detection_responses)
        self._extraction_response = extraction_response

    def complete(self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None) -> str:
        sp = (system_prompt or "").lower()
        if "topic detection" in sp:
            return self._detection_responses.pop(0)
        if "memory extraction" in sp:
            return self._extraction_response
        return ""


def _script(turns: list[EvalTurn], expect_end: EndExpectation | None = None) -> EvalScript:
    return EvalScript(name="test script", peer_id="eval-peer", turns=turns, expect_end=expect_end or EndExpectation())


class TestEvalRunner:
    def test_tracks_active_topic_across_turns_like_an_adapter_would(self):
        storage = MockStorage()
        llm = QueuedLLM(["NEW_TOPIC:Bike", "CONTINUE"])
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)
        runner = EvalRunner(eidolon, storage, EvalScorer(embedder))

        script = _script([
            EvalTurn(user="I want a bike", ai="Nice", expect=TurnExpectation(topic_change="NEW_TOPIC")),
            EvalTurn(user="around 2000 euros", ai="Got it", expect=TurnExpectation(topic_change="CONTINUE")),
        ])

        report = runner.run(script, model_identity="sim/sim", embedder_identity="sim/sim", db_path=":memory:")

        assert [tr.topic_change_actual for tr in report.turn_results] == ["NEW_TOPIC", "CONTINUE"]
        assert [tr.topic_change_pass for tr in report.turn_results] == [True, True]
        # Both turns landed on the same (single) topic — the CONTINUE turn
        # didn't spawn a second one.
        assert len(storage.topics) == 1
        (topic_id,) = storage.topics.keys()
        assert len(storage.get_messages_by_topic(topic_id)) == 4  # 2 messages per turn

    def test_captures_extracted_memory_items_for_scoring(self):
        storage = MockStorage()
        extraction_response = (
            '[{"content": "user wants a bike", "item_type": "essence", '
            '"source_peer": "user", "subject_peer": "user", "attribution_type": "self_statement"}]'
        )
        llm = QueuedLLM(["NEW_TOPIC:Bike"], extraction_response=extraction_response)
        embedder = MockEmbedder()  # returns the same fixed vector for every text
        eidolon = Eidolon(storage, llm, embedder)
        runner = EvalRunner(eidolon, storage, EvalScorer(embedder))

        script = _script([
            EvalTurn(
                user="I want a bike", ai="Nice",
                expect=TurnExpectation(
                    topic_change="NEW_TOPIC",
                    memory_items=[MemoryItemHint(item_type="essence", content_hint="wants a bike")],
                ),
            ),
        ])

        report = runner.run(script, model_identity="sim/sim", embedder_identity="sim/sim", db_path=":memory:")

        match = report.turn_results[0].memory_item_matches[0]
        assert match.matched_item is not None
        assert match.matched_item.content == "user wants a bike"
        assert match.score == 1.0  # identical embedding for hint and item
        assert match.type_match is True

    def test_end_result_reads_final_facts_from_storage(self):
        storage = MockStorage()
        storage.facts = [
            Fact(peer_id="eval-peer", category="preference", key="bike_budget", value="2000 EUR", source="user")
        ]
        llm = QueuedLLM(["CONTINUE"])
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)
        runner = EvalRunner(eidolon, storage, EvalScorer(embedder))

        script = _script(
            [EvalTurn(user="hi", ai="hello", expect=TurnExpectation())],
            expect_end=EndExpectation(
                facts=[FactExpectation(category="preference", key="bike_budget", value="2000 EUR")]
            ),
        )

        report = runner.run(script, model_identity="sim/sim", embedder_identity="sim/sim", db_path=":memory:")

        match = report.end_result.fact_matches[0]
        assert match.matched_fact is not None
        assert match.score == 1.0
