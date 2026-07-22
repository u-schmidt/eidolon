"""SQLite-backed tracer for Eidolon.

Writes trace events to a separate SQLite file from the main memory
database — different lifecycle (disposable, rotatable), different write
volume, no coupling to the memory schema.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from core.trace.migrations import migrate


class SqliteTracer:
    """SQLite implementation of ``Tracer``.

    Parameters
    ----------
    path : str
        Filesystem path to the trace database file.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path).expanduser().resolve()
        self._db: sqlite3.Connection | None = None

    def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._path), check_same_thread=False)
        # See the matching comment in core/stores/sqlite_store.py — WAL's
        # shared-memory requirement is unsafe over network/bridged
        # filesystems (e.g. Docker Desktop's virtiofs bind mounts).
        self._db.execute("PRAGMA journal_mode=DELETE")
        migrate(self._db)

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("SqliteTracer not connected — call connect() first")
        return self._db

    def record(
        self,
        event_type: str,
        payload: dict,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO events (request_id, session_id, event_type, payload, duration_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (request_id, session_id, event_type, json.dumps(payload, default=str), duration_ms),
        )
        self._conn.commit()
