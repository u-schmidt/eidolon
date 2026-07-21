"""Tests for the fragmentary extractor."""

from __future__ import annotations

import pytest

from core.extraction.fragmentary_extractor import (
    FragmentaryExtractor,
    _build_extraction_prompt,
    _normalize_subject_peer,
    _parse_extraction_response,
)
from core.models import EnvironmentSnapshot, MemoryItem
from tests.mocks import SpyTracer


class MockLLM:
    """A mock LLM that returns a predefined JSON response, or raises."""

    def __init__(self, response: str = "[]", should_raise: bool = False) -> None:
        self.response = response
        self.should_raise = should_raise
        self.last_prompt: str | None = None
        self.last_system: str | None = None
        self.last_max_tokens: int | None = None

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.last_prompt = prompt
        self.last_system = system_prompt
        self.last_max_tokens = max_tokens
        if self.should_raise:
            raise RuntimeError("LLM backend unavailable")
        return self.response


class TestParseExtractionResponse:
    def test_empty_array(self):
        result = _parse_extraction_response("[]")
        assert result == []

    def test_single_item(self):
        result = _parse_extraction_response(
            '[{"content": "User talked about Bello", "item_type": "essence", '
            '"source_peer": "user", "subject_peer": "user", "attribution_type": "self_statement"}]'
        )
        assert len(result) == 1
        assert result[0]["content"] == "User talked about Bello"
        assert result[0]["item_type"] == "essence"

    def test_multiple_items(self):
        result = _parse_extraction_response(
            '[{"content": "First", "item_type": "essence", '
            '"source_peer": "user", "subject_peer": "user", "attribution_type": "self_statement"},'
            '{"content": "Second", "item_type": "fact", '
            '"source_peer": "ai", "subject_peer": "user", "attribution_type": "observation"}]'
        )
        assert len(result) == 2

    def test_with_markdown_code_fence(self):
        result = _parse_extraction_response(
            '```json\n[{"content": "Inside fence", "item_type": "essence", '
            '"source_peer": "user", "subject_peer": "user", "attribution_type": "self_statement"}]\n```'
        )
        assert len(result) == 1
        assert result[0]["content"] == "Inside fence"

    def test_with_markdown_fence_no_language(self):
        result = _parse_extraction_response(
            '```\n[{"content": "No lang", "item_type": "essence", '
            '"source_peer": "user", "subject_peer": "user", "attribution_type": "self_statement"}]\n```'
        )
        assert len(result) == 1
        assert result[0]["content"] == "No lang"

    def test_malformed_json_raises(self):
        with pytest.raises(ValueError, match="malformed extraction response"):
            _parse_extraction_response("not json at all")

    def test_not_a_list_raises(self):
        with pytest.raises(ValueError, match="not a JSON array"):
            _parse_extraction_response('{"not": "a list"}')


class TestBuildExtractionPrompt:
    def test_with_topic(self):
        prompt = _build_extraction_prompt(
            user_message="Hello",
            ai_response="Hi there!",
            topic_label="Greetings",
        )
        assert "Current topic: Greetings" in prompt
        assert "User: Hello" in prompt
        assert "AI: Hi there!" in prompt

    def test_without_topic(self):
        prompt = _build_extraction_prompt(
            user_message="Hello",
            ai_response="Hi there!",
        )
        assert "Current topic:" not in prompt
        assert "User: Hello" in prompt
        assert "AI: Hi there!" in prompt


class TestNormalizeSubjectPeer:
    def test_valid_values_pass_through_unchanged(self):
        assert _normalize_subject_peer("user", source_peer="ai") == "user"
        assert _normalize_subject_peer("ai", source_peer="user") == "ai"
        assert _normalize_subject_peer("other", source_peer="user") == "other"

    def test_missing_value_falls_back_to_source_peer(self):
        assert _normalize_subject_peer(None, source_peer="user") == "user"

    def test_out_of_vocabulary_value_falls_back_to_source_peer(self):
        # Seen in practice: a model substituting "self" for "user" despite
        # the extraction prompt explicitly restricting this field to
        # "user"/"ai"/"other" — this is what silently broke fact
        # consolidation before this normalization existed.
        assert _normalize_subject_peer("self", source_peer="user") == "user"

    def test_out_of_vocabulary_value_falls_back_to_the_items_own_source_peer(self):
        assert _normalize_subject_peer("self", source_peer="ai") == "ai"


class TestFragmentaryExtractor:
    def test_extract_returns_memory_items(self):
        llm = MockLLM(
            response='[{"content": "User is frustrated with FitZone", '
            '"item_type": "essence", "source_peer": "user", '
            '"subject_peer": "user", "attribution_type": "self_statement"}]'
        )
        extractor = FragmentaryExtractor(llm)
        items = extractor.extract(
            user_message="FitZone ist immer voll",
            ai_response="Das klingt frustrierend!",
        )
        assert len(items) == 1
        assert isinstance(items[0], MemoryItem)
        assert items[0].content == "User is frustrated with FitZone"
        assert items[0].item_type == "essence"
        assert items[0].source_peer == "user"
        assert items[0].subject_peer == "user"

    def test_extract_empty_response(self):
        llm = MockLLM(response="[]")
        extractor = FragmentaryExtractor(llm)
        items = extractor.extract(
            user_message="Hello",
            ai_response="Hi!",
        )
        assert items == []

    def test_extract_with_topic_label(self):
        llm = MockLLM(response="[]")
        extractor = FragmentaryExtractor(llm)
        extractor.extract(
            user_message="Hello",
            ai_response="Hi!",
            topic_label="Bello Features",
        )
        assert "Current topic: Bello Features" in llm.last_prompt

    def test_extract_with_environment(self):
        llm = MockLLM(
            response='[{"content": "Test", "item_type": "essence", '
            '"source_peer": "user", "subject_peer": "user", '
            '"attribution_type": "self_statement"}]'
        )
        extractor = FragmentaryExtractor(llm)
        env = EnvironmentSnapshot(location="home", situation="morning")
        items = extractor.extract(
            user_message="Test",
            ai_response="OK",
            environment=env,
        )
        assert items[0].context is not None
        assert items[0].context.location == "home"

    def test_system_prompt_is_set(self):
        llm = MockLLM(response="[]")
        extractor = FragmentaryExtractor(llm)
        extractor.extract(user_message="Hello", ai_response="Hi!")
        assert llm.last_system is not None
        assert "memory extraction" in llm.last_system.lower()

    def test_max_tokens_is_32000(self):
        llm = MockLLM()
        extractor = FragmentaryExtractor(llm)
        extractor.extract(user_message="Hi", ai_response="Hello")
        assert llm.last_max_tokens == 32000

    def test_language_parameter_accepted(self):
        llm = MockLLM(response="[]")
        extractor = FragmentaryExtractor(llm, language="de")
        assert extractor._language == "de"
        # Prompts are still in English
        extractor.extract(user_message="Hallo", ai_response="Hallo!")
        assert "memory extraction" in llm.last_system.lower()

    def test_extract_filters_empty_content(self):
        llm = MockLLM(
            response='[{"content": "", "item_type": "essence", '
            '"source_peer": "user", "subject_peer": "user", '
            '"attribution_type": "self_statement"}]'
        )
        extractor = FragmentaryExtractor(llm)
        items = extractor.extract(user_message="Test", ai_response="OK")
        assert items == []

    def test_extract_normalizes_out_of_vocabulary_subject_peer(self):
        # Reproduces a real response from gemma-4-e4b-uncensored-hauhaucs-
        # aggressive: subject_peer "self" instead of the prompt's required
        # "user"/"ai"/"other". Before normalization this item's
        # subject_peer never matched FactConsolidator's _PROFILE_SUBJECT
        # ("user"), so a real, correctly extracted fact silently never
        # reached the durable facts profile.
        llm = MockLLM(
            response='[{"content": "The budget is approximately 2000 euros.", '
            '"item_type": "fact", "source_peer": "user", '
            '"subject_peer": "self", "attribution_type": "self_statement"}]'
        )
        extractor = FragmentaryExtractor(llm)

        items = extractor.extract(user_message="around 2000 euros", ai_response="Got it")

        assert len(items) == 1
        assert items[0].source_peer == "user"
        assert items[0].subject_peer == "user"

    def test_extract_multiple_items(self):
        llm = MockLLM(
            response='['
            '{"content": "First item", "item_type": "essence", '
            '"source_peer": "user", "subject_peer": "user", "attribution_type": "self_statement"},'
            '{"content": "Second item", "item_type": "fact", '
            '"source_peer": "ai", "subject_peer": "user", "attribution_type": "observation"}'
            ']'
        )
        extractor = FragmentaryExtractor(llm)
        items = extractor.extract(user_message="Test", ai_response="OK")
        assert len(items) == 2
        assert items[0].content == "First item"
        assert items[1].content == "Second item"

    def test_records_extraction_event(self):
        llm = MockLLM(
            response='[{"content": "User is frustrated with FitZone", '
            '"item_type": "essence", "source_peer": "user", '
            '"subject_peer": "user", "attribution_type": "self_statement"}]'
        )
        tracer = SpyTracer()
        extractor = FragmentaryExtractor(llm, tracer=tracer)
        extractor.extract(
            user_message="FitZone ist immer voll",
            ai_response="Das klingt frustrierend!",
            request_id="req1",
            session_id="sess1",
        )

        events = tracer.events_of_type("extraction")
        assert len(events) == 1
        event = events[0]
        assert event["request_id"] == "req1"
        assert event["session_id"] == "sess1"
        assert event["duration_ms"] is not None
        assert event["payload"]["user_message"] == "FitZone ist immer voll"
        assert len(event["payload"]["items"]) == 1
        assert event["payload"]["items"][0]["content"] == "User is frustrated with FitZone"

    def test_default_tracer_is_noop(self):
        llm = MockLLM(response="[]")
        extractor = FragmentaryExtractor(llm)
        extractor.extract(user_message="Hello", ai_response="Hi!")

    def test_extract_llm_call_failure_returns_empty_and_records_error(self):
        llm = MockLLM(should_raise=True)
        tracer = SpyTracer()
        extractor = FragmentaryExtractor(llm, tracer=tracer)

        items = extractor.extract(
            user_message="FitZone ist immer voll", ai_response="Das klingt frustrierend!",
            request_id="req1", session_id="sess1",
        )

        assert items == []
        error_events = tracer.events_of_type("error")
        assert len(error_events) == 1
        assert error_events[0]["payload"]["step"] == "extraction"
        assert "LLM backend unavailable" in error_events[0]["payload"]["exception"]
        assert error_events[0]["request_id"] == "req1"
        assert error_events[0]["session_id"] == "sess1"
        # The normal extraction event still fires — items just reads 0.
        extraction_events = tracer.events_of_type("extraction")
        assert len(extraction_events) == 1
        assert extraction_events[0]["payload"]["items"] == []

    def test_extract_malformed_response_returns_empty_and_records_error(self):
        llm = MockLLM(response="not json at all")
        tracer = SpyTracer()
        extractor = FragmentaryExtractor(llm, tracer=tracer)

        items = extractor.extract(user_message="Hello", ai_response="Hi!")

        assert items == []
        error_events = tracer.events_of_type("error")
        assert len(error_events) == 1
        assert error_events[0]["payload"]["step"] == "extraction"
        assert "malformed extraction response" in error_events[0]["payload"]["exception"]
        assert error_events[0]["payload"]["response"] == "not json at all"
        extraction_events = tracer.events_of_type("extraction")
        assert len(extraction_events) == 1
        assert extraction_events[0]["payload"]["items"] == []
