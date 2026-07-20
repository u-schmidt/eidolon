"""Tests for the fail-safe tracer decorator."""

from __future__ import annotations

from core.trace.safe_tracer import SafeTracer


class RaisingTracer:
    """A Tracer whose every method raises, to verify SafeTracer swallows it."""

    def connect(self) -> None:
        raise RuntimeError("connect failed")

    def close(self) -> None:
        raise RuntimeError("close failed")

    def record(self, event_type, payload, *, request_id=None, session_id=None, duration_ms=None) -> None:
        raise RuntimeError("record failed")


class SpyTracer:
    """A Tracer that records calls made to it, for pass-through assertions."""

    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.records: list[dict] = []

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.closed = True

    def record(self, event_type, payload, *, request_id=None, session_id=None, duration_ms=None) -> None:
        self.records.append({
            "event_type": event_type, "payload": payload,
            "request_id": request_id, "session_id": session_id, "duration_ms": duration_ms,
        })


class TestSafeTracerSwallowsFailures:
    def test_connect_does_not_raise_when_wrapped_raises(self):
        tracer = SafeTracer(RaisingTracer())
        tracer.connect()  # must not raise

    def test_close_does_not_raise_when_wrapped_raises(self):
        tracer = SafeTracer(RaisingTracer())
        tracer.close()  # must not raise

    def test_record_does_not_raise_when_wrapped_raises(self):
        tracer = SafeTracer(RaisingTracer())
        tracer.record("topic_detection", {"message": "hi"})  # must not raise


class TestSafeTracerPassesThrough:
    def test_connect_calls_wrapped(self):
        wrapped = SpyTracer()
        SafeTracer(wrapped).connect()
        assert wrapped.connected is True

    def test_close_calls_wrapped(self):
        wrapped = SpyTracer()
        SafeTracer(wrapped).close()
        assert wrapped.closed is True

    def test_record_calls_wrapped_with_same_arguments(self):
        wrapped = SpyTracer()
        SafeTracer(wrapped).record(
            "topic_detection", {"message": "hi"},
            request_id="req1", session_id="sess1", duration_ms=12.3,
        )
        assert len(wrapped.records) == 1
        recorded = wrapped.records[0]
        assert recorded["event_type"] == "topic_detection"
        assert recorded["payload"] == {"message": "hi"}
        assert recorded["request_id"] == "req1"
        assert recorded["session_id"] == "sess1"
        assert recorded["duration_ms"] == 12.3
