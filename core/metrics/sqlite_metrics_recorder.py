"""SQLite-backed operational-metrics recorder for Eidolon.

Writes into the *same* database file ``SqliteStore`` uses — a second,
independent connection, not a shared connection object (keeps
``MetricsRecorder`` and ``StorageProvider`` decoupled, same as
``Tracer``/``StorageProvider`` already are). Runs no migration of its
own: ``core/stores/migrations.py`` already guarantees the
``events``/``llm_calls``/``embedding_calls`` tables exist, since storage
always connects (and migrates) before metrics does — see
``ProviderFactory``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SqliteMetricsRecorder:
    """SQLite implementation of ``MetricsRecorder``.

    Parameters
    ----------
    path : str
        Filesystem path to the *main* Eidolon database file (the same
        one ``SqliteStore`` was given).
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path).expanduser().resolve()
        self._db: sqlite3.Connection | None = None

    def connect(self) -> None:
        self._db = sqlite3.connect(str(self._path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=DELETE")

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("SqliteMetricsRecorder not connected — call connect() first")
        return self._db

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
        self._conn.execute(
            "INSERT INTO events (severity, step, message, peer_id, request_id, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (severity, step, message, peer_id, request_id, session_id),
        )
        self._conn.commit()

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
        self._conn.execute(
            "INSERT INTO llm_calls (call_type, duration_ms, succeeded, peer_id, request_id, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (call_type, duration_ms, int(succeeded), peer_id, request_id, session_id),
        )
        self._conn.commit()

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
        self._conn.execute(
            "INSERT INTO embedding_calls "
            "(purpose, duration_ms, dimension, succeeded, entity_type, entity_id, peer_id, request_id, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (purpose, duration_ms, dimension, int(succeeded), entity_type, entity_id,
             peer_id, request_id, session_id),
        )
        self._conn.commit()
