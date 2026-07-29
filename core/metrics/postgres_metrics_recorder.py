"""Postgres-backed operational-metrics recorder for Eidolon.

Writes into the *same* Postgres instance ``PostgresStore`` uses — a
second, independent connection (same reasoning as
``SqliteMetricsRecorder``). Runs no migration of its own:
``core/stores/postgres_migrations.py`` already guarantees the
``events``/``llm_calls``/``embedding_calls`` tables exist.
"""

from __future__ import annotations

import psycopg


class PostgresMetricsRecorder:
    """Postgres implementation of ``MetricsRecorder``.

    Parameters
    ----------
    host, port, dbname, user, password : str | int
        Connection details — the same fields already on ``StorageConfig``.
    """

    def __init__(self, host: str, port: int, dbname: str, user: str, password: str) -> None:
        self._host = host
        self._port = port
        self._dbname = dbname
        self._user = user
        self._password = password
        self._db: psycopg.Connection | None = None

    def connect(self) -> None:
        """Connects with ``autocommit=True`` — each ``record_*`` method
        is a single INSERT with no need for multi-statement atomicity,
        so there's nothing an explicit transaction buys here (see
        ``core/stores/postgres_store.py``'s ``connect()`` docstring for
        why a long-lived connection without autocommit is a real
        footgun for methods that forget their own trailing commit)."""
        self._db = psycopg.connect(
            host=self._host,
            port=self._port,
            dbname=self._dbname,
            user=self._user,
            password=self._password,
            autocommit=True,
        )

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def _conn(self) -> psycopg.Connection:
        if self._db is None:
            raise RuntimeError("PostgresMetricsRecorder not connected — call connect() first")
        return self._db

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
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO events (severity, step, message, workspace_id, request_id, session_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (severity, step, message, workspace_id, request_id, session_id),
            )

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
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO llm_calls (call_type, duration_ms, succeeded, workspace_id, request_id, session_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (call_type, duration_ms, succeeded, workspace_id, request_id, session_id),
            )

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
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO embedding_calls "
                "(purpose, duration_ms, dimension, succeeded, entity_type, entity_id, workspace_id, request_id, session_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (purpose, duration_ms, dimension, succeeded, entity_type, entity_id,
                 workspace_id, request_id, session_id),
            )
