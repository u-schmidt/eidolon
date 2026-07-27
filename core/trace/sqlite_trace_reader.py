"""SQLite-backed read access to Eidolon's debug trace database.

Read-only counterpart to ``SqliteTracer``. Unlike ``tools/trace.py``'s
own ``TraceReader`` (a CLI tool that intentionally raises
``FileNotFoundError`` when the trace database is missing — the user
asked for it explicitly), this degrades gracefully: a dashboard page
showing a turn that happened to run with ``debug.enabled: false`` (or
was never turned on at all) should just show no rich trace data, not
crash the page.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def _parse_sqlite_timestamp(raw: str | None) -> datetime | None:
    """Parse a raw SQLite ``CURRENT_TIMESTAMP`` string (e.g.
    "2026-07-20 16:57:00", stored in UTC) into a tz-aware datetime. None
    for a missing/unparseable value.

    Duplicated from ``core/metrics/sqlite_metrics_reader.py``'s private
    helper of the same name rather than imported — that module's own
    docstring already documents why: a cross-module import of a
    `_`-prefixed helper would leak an implementation detail across
    module boundaries for one 8-line function.
    """
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


class SqliteTraceReader:
    """SQLite implementation of ``TraceReader``.

    Parameters
    ----------
    path : str
        Filesystem path to the trace database file (``debug.trace_db_path``).
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path).expanduser().resolve()
        self._db: sqlite3.Connection | None = None

    def connect(self) -> None:
        if self._path.exists():
            self._db = sqlite3.connect(str(self._path), check_same_thread=False)

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def events_for_request(self, request_id: str) -> list[dict]:
        if self._db is None:
            return []
        rows = self._db.execute(
            "SELECT event_type, payload, duration_ms, session_id, created_at "
            "FROM events WHERE request_id = ? ORDER BY id",
            (request_id,),
        ).fetchall()
        return [
            {
                "event_type": r[0],
                "payload": json.loads(r[1]),
                "duration_ms": r[2],
                "session_id": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    def events_of_type_since(self, event_type: str, since: datetime) -> list[dict]:
        if self._db is None:
            return []
        rows = self._db.execute(
            "SELECT payload, duration_ms, session_id, request_id, created_at "
            "FROM events WHERE event_type = ? ORDER BY id",
            (event_type,),
        ).fetchall()
        return [
            {
                "payload": json.loads(r[0]),
                "duration_ms": r[1],
                "session_id": r[2],
                "request_id": r[3],
                "created_at": r[4],
            }
            for r in rows
            if (ts := _parse_sqlite_timestamp(r[4])) is not None and ts >= since
        ]
