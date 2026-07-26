"""Postgres-backed tracer for Eidolon.

Writes trace events into the same Postgres instance as production
storage instead of a separate SQLite file on a (possibly bind-mounted,
possibly unreliable — see core/stores/postgres_store.py's docstring)
disk location. Selected over ``SqliteTracer`` when ``storage.provider:
postgres`` — see ``ProviderFactory.create_tracer()``.
"""

from __future__ import annotations

import json

import psycopg
from psycopg.types.json import Jsonb

from core.trace.postgres_migrations import migrate


def _dumps(obj: object) -> str:
    """Matches SqliteTracer's json.dumps(payload, default=str) — payloads
    can carry non-JSON-native values (e.g. datetimes from Fact/Topic
    fields)."""
    return json.dumps(obj, default=str)


class PostgresTracer:
    """Postgres implementation of ``Tracer``.

    Parameters
    ----------
    host, port, dbname, user, password : str | int
        Connection details — the same fields already on ``StorageConfig``,
        reused as-is since this always targets the same Postgres instance
        as storage.
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
        migrate(self._db)

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def _conn(self) -> psycopg.Connection:
        if self._db is None:
            raise RuntimeError("PostgresTracer not connected — call connect() first")
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
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO trace_events (request_id, session_id, event_type, payload, duration_ms) "
                "VALUES (%s, %s, %s, %s, %s)",
                (request_id, session_id, event_type, Jsonb(payload, dumps=_dumps), duration_ms),
            )
        self._conn.commit()
