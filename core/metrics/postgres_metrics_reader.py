"""Postgres-backed read access to Eidolon's operational-metrics tables.

Read-only counterpart to ``PostgresMetricsRecorder`` — see
``core/interfaces.py``'s ``MetricsReader`` for why this is a separate
class rather than reusing the writer.
"""

from __future__ import annotations

import psycopg


class PostgresMetricsReader:
    """Postgres implementation of ``MetricsReader``.

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
            raise RuntimeError("PostgresMetricsReader not connected — call connect() first")
        return self._db

    def events_for_request(self, request_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT severity, step, message, peer_id, session_id, created_at "
            "FROM events WHERE request_id = %s ORDER BY id",
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
            "FROM llm_calls WHERE request_id = %s ORDER BY id",
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
            "peer_id, session_id, created_at FROM embedding_calls WHERE request_id = %s ORDER BY id",
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
