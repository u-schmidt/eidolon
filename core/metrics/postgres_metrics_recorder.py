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
        self._db = psycopg.connect(
            host=self._host,
            port=self._port,
            dbname=self._dbname,
            user=self._user,
            password=self._password,
            autocommit=False,
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
        peer_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO events (severity, step, message, peer_id, request_id, session_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
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
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO llm_calls (call_type, duration_ms, succeeded, peer_id, request_id, session_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (call_type, duration_ms, succeeded, peer_id, request_id, session_id),
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
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO embedding_calls "
                "(purpose, duration_ms, dimension, succeeded, entity_type, entity_id, peer_id, request_id, session_id) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (purpose, duration_ms, dimension, succeeded, entity_type, entity_id,
                 peer_id, request_id, session_id),
            )
        self._conn.commit()
