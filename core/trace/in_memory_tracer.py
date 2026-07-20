"""In-memory tracer for Eidolon.

Collects trace events in a Python list instead of writing them anywhere
persistent — for one-off inspection of a single call (e.g. a live
"show me everything" recompute) without needing debug tracing enabled
or a database file at all.
"""

from __future__ import annotations

from datetime import UTC, datetime


class InMemoryTracer:
    """Tracer implementation that collects events for inspection.

    Implements the ``Tracer`` protocol structurally.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

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
        self.events.append({
            "event_type": event_type,
            "payload": payload,
            "request_id": request_id,
            "session_id": session_id,
            "duration_ms": duration_ms,
            "created_at": datetime.now(UTC).isoformat(),
        })

    def events_of_type(self, event_type: str) -> list[dict]:
        return [e for e in self.events if e["event_type"] == event_type]
