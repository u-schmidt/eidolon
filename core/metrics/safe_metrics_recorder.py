"""Fail-safe metrics-recorder decorator for Eidolon.

Wraps another MetricsRecorder so a broken operational-metrics backend can
never take down real extraction, storage, or detection. More important
here than for ``SafeTracer``: unlike debug tracing, this path always
runs in production, never behind an opt-in flag.
"""

from __future__ import annotations

import logging

from core.interfaces import MetricsRecorder

logger = logging.getLogger(__name__)


class SafeMetricsRecorder:
    """MetricsRecorder decorator that swallows failures from a wrapped
    MetricsRecorder.

    Implements the ``MetricsRecorder`` protocol structurally. Every call
    is forwarded to ``wrapped``; if that raises, the exception is logged
    via the standard ``logging`` module instead of propagating.
    """

    def __init__(self, wrapped: MetricsRecorder) -> None:
        self._wrapped = wrapped

    def connect(self) -> None:
        try:
            self._wrapped.connect()
        except Exception:
            logger.exception("MetricsRecorder.connect() failed — metrics disabled for this session")

    def close(self) -> None:
        try:
            self._wrapped.close()
        except Exception:
            logger.exception("MetricsRecorder.close() failed")

    def record_event(
        self,
        severity: str,
        step: str,
        message: str,
        *,
        workspace_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        try:
            self._wrapped.record_event(
                severity, step, message,
                workspace_id=workspace_id, request_id=request_id, session_id=session_id,
            )
        except Exception:
            logger.exception("MetricsRecorder.record_event(%s) failed — event lost", step)

    def record_llm_call(
        self,
        call_type: str,
        duration_ms: float,
        succeeded: bool,
        *,
        workspace_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        try:
            self._wrapped.record_llm_call(
                call_type, duration_ms, succeeded,
                workspace_id=workspace_id, request_id=request_id, session_id=session_id,
            )
        except Exception:
            logger.exception("MetricsRecorder.record_llm_call(%s) failed — event lost", call_type)

    def record_embedding_call(
        self,
        purpose: str,
        duration_ms: float,
        dimension: int,
        succeeded: bool,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        workspace_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        try:
            self._wrapped.record_embedding_call(
                purpose, duration_ms, dimension, succeeded,
                entity_type=entity_type, entity_id=entity_id,
                workspace_id=workspace_id, request_id=request_id, session_id=session_id,
            )
        except Exception:
            logger.exception("MetricsRecorder.record_embedding_call(%s) failed — event lost", purpose)
