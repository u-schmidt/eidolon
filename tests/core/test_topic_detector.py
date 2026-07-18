"""Tests for the topic detector."""

from __future__ import annotations

import pytest

from core.detection.topic_detector import TopicDetector, _parse_detection_response
from core.models import TopicChange
from tests.mocks import SpyTracer


class MockLLM:
    """A mock LLM that returns a predefined response."""

    def __init__(self, response: str = "CONTINUE") -> None:
        self.response = response
        self.last_prompt: str | None = None
        self.last_system: str | None = None

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 100,
    ) -> str:
        self.last_prompt = prompt
        self.last_system = system_prompt
        return self.response


class TestParseDetectionResponse:
    def test_continue(self):
        result = _parse_detection_response("CONTINUE")
        assert result.type == "CONTINUE"
        assert result.label is None

    def test_new_topic(self):
        result = _parse_detection_response("NEW_TOPIC:Bello Features")
        assert result.type == "NEW_TOPIC"
        assert result.label == "Bello Features"

    def test_new_topic_with_colon(self):
        result = _parse_detection_response("NEW_TOPIC: Bello Features")
        assert result.type == "NEW_TOPIC"
        assert result.label == "Bello Features"

    def test_new_topic_lowercase(self):
        result = _parse_detection_response("new_topic:bello features")
        assert result.type == "NEW_TOPIC"
        assert result.label == "bello features"

    def test_new_topic_with_period(self):
        result = _parse_detection_response("NEW_TOPIC:Bello.")
        assert result.type == "NEW_TOPIC"
        assert result.label == "Bello"

    def test_new_topic_fallback(self):
        result = _parse_detection_response("I think this is NEW_TOPIC about Bello")
        assert result.type == "NEW_TOPIC"
        assert result.label == "about Bello"

    def test_unrecognized_defaults_to_continue(self):
        result = _parse_detection_response("SOMETHING_ELSE")
        assert result.type == "CONTINUE"
        assert result.label is None

    def test_empty_response_defaults_to_continue(self):
        result = _parse_detection_response("")
        assert result.type == "CONTINUE"


class TestTopicDetector:
    def test_detect_continue(self):
        llm = MockLLM(response="CONTINUE")
        detector = TopicDetector(llm)
        result = detector.detect("jep, mach weiter", current_topic_summary="Bello features")
        assert result.type == "CONTINUE"

    def test_detect_new_topic(self):
        llm = MockLLM(response="NEW_TOPIC:Shopping List")
        detector = TopicDetector(llm)
        result = detector.detect("Lass uns über Mampfi sprechen")
        assert result.type == "NEW_TOPIC"
        assert result.label == "Shopping List"

    def test_detect_first_message(self):
        llm = MockLLM(response="NEW_TOPIC:Topic Memory System")
        detector = TopicDetector(llm)
        result = detector.detect("Lass uns über das Topic Memory System sprechen")
        assert result.type == "NEW_TOPIC"
        assert result.label == "Topic Memory System"

    def test_prompt_includes_current_topic(self):
        llm = MockLLM(response="CONTINUE")
        detector = TopicDetector(llm)
        detector.detect("ok", current_topic_summary="Bello features")
        assert "Current topic: Bello features" in llm.last_prompt

    def test_prompt_without_current_topic(self):
        llm = MockLLM(response="NEW_TOPIC:Test")
        detector = TopicDetector(llm)
        detector.detect("Hello")
        assert "Current topic:" not in llm.last_prompt
        assert "User: Hello" in llm.last_prompt

    def test_system_prompt_is_set(self):
        llm = MockLLM(response="CONTINUE")
        detector = TopicDetector(llm)
        detector.detect("test")
        assert llm.last_system is not None
        assert "topic detection" in llm.last_system.lower()

    def test_records_topic_detection_event(self):
        llm = MockLLM(response="NEW_TOPIC:Bello Features")
        tracer = SpyTracer()
        detector = TopicDetector(llm, tracer=tracer)
        detector.detect(
            "Lass uns über Bello sprechen",
            current_topic_summary="prior topic",
            request_id="req1",
            session_id="sess1",
        )

        events = tracer.events_of_type("topic_detection")
        assert len(events) == 1
        event = events[0]
        assert event["request_id"] == "req1"
        assert event["session_id"] == "sess1"
        assert event["duration_ms"] is not None
        assert event["payload"]["message"] == "Lass uns über Bello sprechen"
        assert event["payload"]["current_topic_summary"] == "prior topic"
        assert event["payload"]["response"] == "NEW_TOPIC:Bello Features"
        assert event["payload"]["result_type"] == "NEW_TOPIC"
        assert event["payload"]["result_label"] == "Bello Features"

    def test_default_tracer_is_noop(self):
        # No tracer passed — must not raise.
        llm = MockLLM(response="CONTINUE")
        detector = TopicDetector(llm)
        detector.detect("hello")
