"""Fail-safe tracer decorator for Eidolon.

Wraps another Tracer so a broken debug/observability backend can never
take down real extraction, storage, or detection.
"""

from __future__ import annotations

import logging

from core.interfaces import Tracer

logger = logging.getLogger(__name__)


class SafeTracer:
    """Tracer decorator that swallows failures from a wrapped Tracer.

    Implements the ``Tracer`` protocol structurally. Every call is
    forwarded to ``wrapped``; if that raises, the exception is logged via
    the standard ``logging`` module instead of propagating — tracing is a
    side channel and must never be able to abort the turn it's observing.
    """

    def __init__(self, wrapped: Tracer) -> None:
        self._wrapped = wrapped

    def connect(self) -> None:
        try:
            self._wrapped.connect()
        except Exception:
            logger.exception("Tracer.connect() failed — tracing disabled for this session")

    def close(self) -> None:
        try:
            self._wrapped.close()
        except Exception:
            logger.exception("Tracer.close() failed")

    def record(
        self,
        event_type: str,
        payload: dict,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        try:
            self._wrapped.record(
                event_type, payload,
                request_id=request_id, session_id=session_id, duration_ms=duration_ms,
            )
        except Exception:
            logger.exception("Tracer.record(%s) failed — event lost", event_type)
