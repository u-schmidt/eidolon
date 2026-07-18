"""Tests for the no-op tracer."""

from __future__ import annotations

from core.trace.null_tracer import NullTracer


class TestNullTracer:
    def test_connect_close_are_noop(self):
        tracer = NullTracer()
        tracer.connect()
        tracer.close()

    def test_record_does_not_raise(self):
        tracer = NullTracer()
        tracer.record(
            "topic_detection",
            {"message": "hello"},
            request_id="req1",
            session_id="sess1",
            duration_ms=12.3,
        )

    def test_record_without_optional_args(self):
        tracer = NullTracer()
        tracer.record("error", {"exception": "boom"})
