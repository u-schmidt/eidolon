"""Tests for TopicRelabeler.

Reuses the facade-level mocks from tests.core.test_eidolon (MockStorage,
MockLLM, MockEmbedder) so the relabeler is exercised against the same
storage/LLM behavior the Eidolon facade sees.
"""

from __future__ import annotations

from core.summarization.topic_relabeler import (
    TopicRelabeler,
    _parse_relabel_response,
)
from tests.core.test_eidolon import MockLLM, MockStorage
from tests.mocks import SpyMetricsRecorder, SpyTracer


def _relabeler(storage, llm, tracer=None, metrics=None) -> TopicRelabeler:
    return TopicRelabeler(storage, llm, tracer=tracer, metrics=metrics)


class TestParseRelabelResponse:
    def test_strips_whitespace(self):
        assert _parse_relabel_response("  Buying a bike  ") == "Buying a bike"

    def test_strips_trailing_period(self):
        assert _parse_relabel_response("Buying a bike.") == "Buying a bike"

    def test_strips_surrounding_quotes(self):
        assert _parse_relabel_response('"Buying a bike"') == "Buying a bike"
        assert _parse_relabel_response("'Buying a bike'") == "Buying a bike"

    def test_blank_response_yields_empty(self):
        assert _parse_relabel_response("   ") == ""


class TestRelabel:
    def test_persists_label_on_success(self):
        storage = MockStorage()
        tid = storage.create_topic("", workspace_id="user")
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="user", topic_id=tid)
        storage.store_message(turn_id=1, role="ai", content="hello", workspace_id="ai", topic_id=tid)
        llm = MockLLM(relabel_response="Buying a bike")

        result = _relabeler(storage, llm).relabel(tid)

        assert result is True
        assert storage.topics[tid]["label"] == "Buying a bike"

    def test_does_not_close_topic(self):
        storage = MockStorage()
        tid = storage.create_topic("", workspace_id="user")
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="user", topic_id=tid)
        llm = MockLLM(relabel_response="Buying a bike")

        _relabeler(storage, llm).relabel(tid)

        assert storage.topics[tid]["status"] == "active"

    def test_does_not_touch_summary_or_tags(self):
        storage = MockStorage()
        tid = storage.create_topic("", workspace_id="user")
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="user", topic_id=tid)
        llm = MockLLM(relabel_response="Buying a bike")

        _relabeler(storage, llm).relabel(tid)

        assert storage.topics[tid]["summary"] == ""
        assert tid not in storage.topic_tags

    def test_returns_false_when_no_messages(self):
        storage = MockStorage()
        tid = storage.create_topic("", workspace_id="user")
        llm = MockLLM(relabel_response="Buying a bike")

        assert _relabeler(storage, llm).relabel(tid) is False
        assert storage.topics[tid]["label"] == ""

    def test_returns_false_on_blank_response(self):
        storage = MockStorage()
        tid = storage.create_topic("", workspace_id="user")
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="user", topic_id=tid)
        llm = MockLLM(relabel_response="   ")

        assert _relabeler(storage, llm).relabel(tid) is False
        assert storage.topics[tid]["label"] == ""

    def test_returns_false_on_exception(self):
        class FailingStorage(MockStorage):
            def get_messages_by_topic(self, topic_id):
                raise RuntimeError("db unavailable")

        storage = FailingStorage()
        tid = storage.create_topic("", workspace_id="user")
        llm = MockLLM(relabel_response="Buying a bike")

        assert _relabeler(storage, llm).relabel(tid) is False

    def test_records_topic_relabel_event(self):
        storage = MockStorage()
        tid = storage.create_topic("", workspace_id="user")
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="user", topic_id=tid)
        llm = MockLLM(relabel_response="Buying a bike")
        tracer = SpyTracer()

        _relabeler(storage, llm, tracer=tracer).relabel(tid, request_id="req1", session_id="sess1")

        events = tracer.events_of_type("topic_relabel")
        assert len(events) == 1
        event = events[0]
        assert event["request_id"] == "req1"
        assert event["session_id"] == "sess1"
        assert event["payload"]["topic_id"] == tid
        assert event["payload"]["message_count"] == 1
        assert event["payload"]["label"] == "Buying a bike"

    def test_records_error_on_failure(self, caplog):
        class FailingStorage(MockStorage):
            def get_messages_by_topic(self, topic_id):
                raise RuntimeError("db unavailable")

        storage = FailingStorage()
        tid = storage.create_topic("", workspace_id="user")
        llm = MockLLM(relabel_response="Buying a bike")
        tracer = SpyTracer()
        metrics = SpyMetricsRecorder()

        with caplog.at_level("WARNING"):
            _relabeler(storage, llm, tracer=tracer, metrics=metrics).relabel(tid)  # must not raise

        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "topic_relabel" for e in error_events)
        assert "topic_relabel failed" in caplog.text
        assert any(e["step"] == "topic_relabel" for e in metrics.events)

    def test_records_llm_call_metric(self):
        storage = MockStorage()
        tid = storage.create_topic("", workspace_id="user")
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="user", topic_id=tid)
        llm = MockLLM(relabel_response="Buying a bike")
        metrics = SpyMetricsRecorder()

        _relabeler(storage, llm, metrics=metrics).relabel(tid, request_id="req1")

        assert len(metrics.llm_calls) == 1
        assert metrics.llm_calls[0]["call_type"] == "topic_relabel"
        assert metrics.llm_calls[0]["succeeded"] is True
        assert metrics.llm_calls[0]["request_id"] == "req1"
