"""Tests for the topic detector."""

from __future__ import annotations

from core.detection.topic_detector import TopicDetector, _build_detection_prompt, _parse_detection_response
from core.models import TopicChange
from tests.mocks import SpyMetricsRecorder, SpyTracer


class MockLLM:
    """A mock LLM that returns a predefined response, or raises."""

    def __init__(self, response: str = "CONTINUE", should_raise: bool = False) -> None:
        self.response = response
        self.should_raise = should_raise
        self.last_prompt: str | None = None
        self.last_system: str | None = None

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.last_prompt = prompt
        self.last_system = system_prompt
        if self.should_raise:
            raise RuntimeError("LLM backend unavailable")
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


class TestBuildDetectionPrompt:
    def test_includes_recent_conversation_block(self):
        recent = [
            {"role": "user", "content": "das war nur ein test"},
            {"role": "ai", "content": "alles klar"},
        ]
        prompt = _build_detection_prompt("und jetzt?", "General conversation", recent)
        assert prompt == (
            "Current topic: General conversation\n"
            "Recent conversation:\n"
            "user: das war nur ein test\n"
            "ai: alles klar\n"
            "User: und jetzt?\n"
            "→"
        )

    def test_omits_recent_conversation_block_when_none(self):
        prompt = _build_detection_prompt("ok", "Bello features", None)
        assert "Recent conversation:" not in prompt
        assert prompt == "Current topic: Bello features\nUser: ok\n→"

    def test_omits_recent_conversation_block_when_empty(self):
        prompt = _build_detection_prompt("ok", "Bello features", [])
        assert "Recent conversation:" not in prompt

    def test_no_current_topic_and_no_recent_messages(self):
        prompt = _build_detection_prompt("Hello", None, None)
        assert prompt == "User: Hello\n→"


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

    def test_records_llm_call_metric(self):
        llm = MockLLM(response="NEW_TOPIC:Bello Features")
        metrics = SpyMetricsRecorder()
        detector = TopicDetector(llm, metrics=metrics)
        detector.detect("Lass uns über Bello sprechen", request_id="req1", session_id="sess1")

        assert len(metrics.llm_calls) == 1
        call = metrics.llm_calls[0]
        assert call["call_type"] == "topic_detection"
        assert call["succeeded"] is True
        assert call["duration_ms"] is not None
        assert call["request_id"] == "req1"
        assert call["session_id"] == "sess1"

    def test_default_metrics_is_noop(self):
        # No metrics passed — must not raise.
        llm = MockLLM(response="CONTINUE")
        detector = TopicDetector(llm)
        detector.detect("hello")

    def test_default_tracer_is_noop(self):
        # No tracer passed — must not raise.
        llm = MockLLM(response="CONTINUE")
        detector = TopicDetector(llm)
        detector.detect("hello")

    def test_detect_llm_call_failure_defaults_to_continue_and_records_error(self, caplog):
        llm = MockLLM(should_raise=True)
        tracer = SpyTracer()
        metrics = SpyMetricsRecorder()
        detector = TopicDetector(llm, tracer=tracer, metrics=metrics)

        with caplog.at_level("WARNING"):
            result = detector.detect("hello", request_id="req1", session_id="sess1")

        assert result == TopicChange(type="CONTINUE")
        error_events = tracer.events_of_type("error")
        assert len(error_events) == 1
        assert error_events[0]["payload"]["step"] == "topic_detection"
        assert "LLM backend unavailable" in error_events[0]["payload"]["exception"]
        assert error_events[0]["request_id"] == "req1"
        assert error_events[0]["session_id"] == "sess1"
        assert "topic_detection failed" in caplog.text
        # The normal topic_detection event still fires, with an empty response.
        detection_events = tracer.events_of_type("topic_detection")
        assert len(detection_events) == 1
        assert detection_events[0]["payload"]["response"] == ""
        assert detection_events[0]["payload"]["result_type"] == "CONTINUE"
        # Metrics: one error event, and the llm_call still fires but
        # marked failed (mirrors the tracer's own always-fires behavior).
        assert len(metrics.events) == 1
        assert metrics.events[0]["severity"] == "error"
        assert metrics.events[0]["step"] == "topic_detection"
        assert metrics.events[0]["request_id"] == "req1"
        assert len(metrics.llm_calls) == 1
        assert metrics.llm_calls[0]["succeeded"] is False

    def test_recent_messages_passed_into_prompt(self):
        llm = MockLLM(response="CONTINUE")
        detector = TopicDetector(llm)
        recent = [{"role": "user", "content": "vorher gesagt"}, {"role": "ai", "content": "verstanden"}]
        detector.detect("und weiter?", current_topic_summary="General conversation", recent_messages=recent)
        assert "Recent conversation:" in llm.last_prompt
        assert "user: vorher gesagt" in llm.last_prompt

    def test_recent_message_count_recorded_in_trace(self):
        llm = MockLLM(response="CONTINUE")
        tracer = SpyTracer()
        detector = TopicDetector(llm, tracer=tracer)
        recent = [{"role": "user", "content": "a"}, {"role": "ai", "content": "b"}]
        detector.detect("c", current_topic_summary="topic", recent_messages=recent)

        event = tracer.events_of_type("topic_detection")[0]
        assert event["payload"]["recent_message_count"] == 2

    def test_recent_message_count_zero_when_none(self):
        llm = MockLLM(response="CONTINUE")
        tracer = SpyTracer()
        detector = TopicDetector(llm, tracer=tracer)
        detector.detect("c")

        event = tracer.events_of_type("topic_detection")[0]
        assert event["payload"]["recent_message_count"] == 0
