"""No-op metrics recorder for Eidolon.

Constructor-level default for callers that wire nothing at all (tests,
ad hoc scripts) — every instrumented class falls back to this when no
``metrics`` is passed in directly. Unlike ``NullTracer``, the real
production wiring path (``ProviderFactory.create_metrics_recorder()``)
never chooses this — operational metrics are always on there.
"""

from __future__ import annotations


class NullMetricsRecorder:
    """MetricsRecorder implementation that discards everything.

    Implements the ``MetricsRecorder`` protocol structurally.
    """

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def record_event(
        self,
        severity: str,
        step: str,
        message: str,
        *,
        peer_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        pass

    def record_llm_call(
        self,
        call_type: str,
        duration_ms: float,
        succeeded: bool,
        *,
        peer_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        pass

    def record_embedding_call(
        self,
        purpose: str,
        duration_ms: float,
        dimension: int,
        succeeded: bool,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        peer_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        pass
