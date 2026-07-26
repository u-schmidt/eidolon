"""Auto-migration for the Eidolon Postgres-backed trace database.

Same ``trace_schema_migrations``-table-tracked approach as
``core/stores/postgres_migrations.py`` (Postgres has no ``PRAGMA
user_version``), kept as an independent version sequence rather than
folded into that file — this schema is optional and debug-gated (only
ever created in environments that turn on ``debug.enabled``), unlike the
main storage schema or the always-on operational-metrics tables in
``core/stores/postgres_migrations.py``, which must always exist.
"""

from __future__ import annotations

import psycopg

TRACE_MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS trace_events (
        id SERIAL PRIMARY KEY,
        request_id TEXT,
        session_id TEXT,
        event_type TEXT NOT NULL,
        payload JSONB NOT NULL,
        duration_ms REAL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_trace_events_request ON trace_events(request_id);
    CREATE INDEX IF NOT EXISTS idx_trace_events_session ON trace_events(session_id);
    """,
}


def migrate(conn: psycopg.Connection) -> None:
    """Run all pending migrations on the given trace connection.

    Safe to call on every connect() — only unapplied versions execute.
    """
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS trace_schema_migrations "
            "(version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        cur.execute("SELECT COALESCE(MAX(version), 0) FROM trace_schema_migrations")
        row = cur.fetchone()
        version = row[0] if row else 0
    conn.commit()

    for v in range(version + 1, max(TRACE_MIGRATIONS) + 1):
        if v not in TRACE_MIGRATIONS:
            continue
        with conn.cursor() as cur:
            cur.execute(TRACE_MIGRATIONS[v])
            cur.execute("INSERT INTO trace_schema_migrations (version) VALUES (%s)", (v,))
        conn.commit()
