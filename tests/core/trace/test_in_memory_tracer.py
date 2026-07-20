"""Tests for the in-memory tracer."""

from __future__ import annotations

from core.trace.in_memory_tracer import InMemoryTracer


class TestInMemoryTracer:
    def test_connect_close_are_noop(self):
        tracer = InMemoryTracer()
        tracer.connect()
        tracer.close()

    def test_record_collects_event(self):
        tracer = InMemoryTracer()
        tracer.record(
            "topic_detection",
            {"message": "hello"},
            request_id="req1",
            session_id="sess1",
            duration_ms=12.3,
        )

        assert len(tracer.events) == 1
        event = tracer.events[0]
        assert event["event_type"] == "topic_detection"
        assert event["payload"] == {"message": "hello"}
        assert event["request_id"] == "req1"
        assert event["session_id"] == "sess1"
        assert event["duration_ms"] == 12.3
        assert event["created_at"] is not None

    def test_record_without_optional_args(self):
        tracer = InMemoryTracer()
        tracer.record("error", {"exception": "boom"})

        event = tracer.events[0]
        assert event["request_id"] is None
        assert event["session_id"] is None
        assert event["duration_ms"] is None

    def test_events_of_type_filters(self):
        tracer = InMemoryTracer()
        tracer.record("topic_detection", {})
        tracer.record("error", {})
        tracer.record("topic_detection", {})

        assert len(tracer.events_of_type("topic_detection")) == 2
        assert len(tracer.events_of_type("error")) == 1
        assert tracer.events_of_type("nonexistent") == []
