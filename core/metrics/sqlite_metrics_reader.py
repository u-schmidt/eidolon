"""SQLite-backed read access to Eidolon's operational-metrics tables.

Read-only counterpart to ``SqliteMetricsRecorder`` — see
``core/interfaces.py``'s ``MetricsReader`` for why this is a separate
class rather than reusing the writer.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def _parse_sqlite_timestamp(raw: str | None) -> datetime | None:
    """Parse a raw SQLite ``CURRENT_TIMESTAMP`` string (e.g.
    "2026-07-20 16:57:00", stored in UTC) into a tz-aware datetime.
    None for a missing/unparseable value.

    Duplicated from ``core/stores/sqlite_store.py``'s private helper of
    the same name rather than imported — a cross-module import of a
    `_`-prefixed helper would leak an implementation detail across the
    core/stores vs core/metrics boundary for one 8-line function.
    """
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


class SqliteMetricsReader:
    """SQLite implementation of ``MetricsReader``.

    Parameters
    ----------
    path : str
        Filesystem path to the *main* Eidolon database file (the same
        one ``SqliteStore``/``SqliteMetricsRecorder`` use).
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path).expanduser().resolve()
        self._db: sqlite3.Connection | None = None

    def connect(self) -> None:
        self._db = sqlite3.connect(str(self._path), check_same_thread=False)

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("SqliteMetricsReader not connected — call connect() first")
        return self._db

    def events_for_request(self, request_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT severity, step, message, peer_id, session_id, created_at "
            "FROM events WHERE request_id = ? ORDER BY id",
            (request_id,),
        ).fetchall()
        return [
            {
                "severity": r[0], "step": r[1], "message": r[2],
                "peer_id": r[3], "session_id": r[4], "created_at": r[5],
            }
            for r in rows
        ]

    def llm_calls_for_request(self, request_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT call_type, duration_ms, succeeded, peer_id, session_id, created_at "
            "FROM llm_calls WHERE request_id = ? ORDER BY id",
            (request_id,),
        ).fetchall()
        return [
            {
                "call_type": r[0], "duration_ms": r[1], "succeeded": bool(r[2]),
                "peer_id": r[3], "session_id": r[4], "created_at": r[5],
            }
            for r in rows
        ]

    def embedding_calls_for_request(self, request_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT purpose, duration_ms, dimension, succeeded, entity_type, entity_id, "
            "peer_id, session_id, created_at FROM embedding_calls WHERE request_id = ? ORDER BY id",
            (request_id,),
        ).fetchall()
        return [
            {
                "purpose": r[0], "duration_ms": r[1], "dimension": r[2], "succeeded": bool(r[3]),
                "entity_type": r[4], "entity_id": r[5], "peer_id": r[6],
                "session_id": r[7], "created_at": r[8],
            }
            for r in rows
        ]

    def events_since(self, since: datetime) -> list[dict]:
        rows = self._conn.execute(
            "SELECT severity, step, message, peer_id, session_id, created_at FROM events ORDER BY id"
        ).fetchall()
        return [
            {
                "severity": r[0], "step": r[1], "message": r[2],
                "peer_id": r[3], "session_id": r[4], "created_at": r[5],
            }
            for r in rows
            if (ts := _parse_sqlite_timestamp(r[5])) is not None and ts >= since
        ]

    def llm_calls_since(self, since: datetime) -> list[dict]:
        rows = self._conn.execute(
            "SELECT call_type, duration_ms, succeeded, peer_id, session_id, created_at "
            "FROM llm_calls ORDER BY id"
        ).fetchall()
        return [
            {
                "call_type": r[0], "duration_ms": r[1], "succeeded": bool(r[2]),
                "peer_id": r[3], "session_id": r[4], "created_at": r[5],
            }
            for r in rows
            if (ts := _parse_sqlite_timestamp(r[5])) is not None and ts >= since
        ]

    def embedding_calls_since(self, since: datetime) -> list[dict]:
        rows = self._conn.execute(
            "SELECT purpose, duration_ms, dimension, succeeded, entity_type, entity_id, "
            "peer_id, session_id, created_at FROM embedding_calls ORDER BY id"
        ).fetchall()
        return [
            {
                "purpose": r[0], "duration_ms": r[1], "dimension": r[2], "succeeded": bool(r[3]),
                "entity_type": r[4], "entity_id": r[5], "peer_id": r[6],
                "session_id": r[7], "created_at": r[8],
            }
            for r in rows
            if (ts := _parse_sqlite_timestamp(r[8])) is not None and ts >= since
        ]
