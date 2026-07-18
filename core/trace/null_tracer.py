"""No-op tracer for Eidolon.

Default when debug tracing is not configured — every call is a no-op, so
production behavior and performance are unaffected.
"""

from __future__ import annotations


class NullTracer:
    """Tracer implementation that discards everything.

    Implements the ``Tracer`` protocol structurally.
    """

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def record(
        self,
        event_type: str,
        payload: dict,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        pass
