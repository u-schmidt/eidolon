"""Postgres-backed read access to Eidolon's debug trace table.

Read-only counterpart to ``PostgresTracer`` — reads ``trace_events``
from the same Postgres instance production storage uses.
"""

from __future__ import annotations

import psycopg

from core.trace.postgres_migrations import migrate


class PostgresTraceReader:
    """Postgres implementation of ``TraceReader``.

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
        """Connects with ``autocommit=False`` only for ``migrate()``,
        which commits its own work internally per migration step; every
        real read after that runs with autocommit on instead — a
        long-lived connection without it leaves each ``SELECT`` sitting
        "idle in transaction" indefinitely (no trailing commit), the
        same live issue ``core/stores/postgres_store.py``'s own
        ``connect()`` docstring documents and fixes for the main
        storage connection."""
        self._db = psycopg.connect(
            host=self._host,
            port=self._port,
            dbname=self._dbname,
            user=self._user,
            password=self._password,
            autocommit=False,
        )
        # Ensures trace_events exists even if debug.enabled was never
        # turned on for this deployment (a real PostgresTracer.connect()
        # never ran) — idempotent, guarded by its own version-tracking
        # table, same call PostgresTracer.connect() makes.
        migrate(self._db)
        self._db.autocommit = True

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def _conn(self) -> psycopg.Connection:
        if self._db is None:
            raise RuntimeError("PostgresTraceReader not connected — call connect() first")
        return self._db

    def events_for_request(self, request_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT event_type, payload, duration_ms, session_id, created_at "
            "FROM trace_events WHERE request_id = %s ORDER BY id",
            (request_id,),
        ).fetchall()
        return [
            {
                "event_type": r[0],
                "payload": r[1],
                "duration_ms": r[2],
                "session_id": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]
