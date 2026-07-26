"""Auto-migration for Postgres storage.

Postgres has no ``PRAGMA user_version`` equivalent, so applied versions
are tracked in a small ``schema_migrations`` table instead. Each
migration runs inside its own transaction — a failed migration never
leaves a half-applied version marked complete, unlike SQLite's
autocommit-per-statement ``executescript`` (see core/stores/migrations.py).
"""

from __future__ import annotations

from pathlib import Path

import psycopg

_SCHEMA_SQL_PATH = Path(__file__).parent / "postgres_schema.sql"

# Version 1: the consolidated initial schema (postgres_schema.sql) — not a
# replay of SQLite's 7 migrations, since no legacy Postgres data exists to
# carry forward. Future Postgres schema changes append new versions here,
# the same way core/stores/migrations.py's MIGRATIONS dict grows.
POSTGRES_MIGRATIONS: dict[int, str] = {
    1: _SCHEMA_SQL_PATH.read_text(),
    # Real, persisted per-topic message count ("mini-topic" soft-hide,
    # see ROADMAP.md) — see core/stores/migrations.py's version 8 for
    # the full rationale (mirrors it exactly).
    2: """
    ALTER TABLE topics ADD COLUMN message_count INTEGER NOT NULL DEFAULT 0;

    UPDATE topics SET message_count = (
        SELECT COUNT(*) FROM messages WHERE messages.topic_id = topics.id
    );
    """,
    # Always-on operational metrics — see core/stores/migrations.py's
    # version 9 for the full rationale (mirrors it exactly).
    3: """
    CREATE TABLE IF NOT EXISTS events (
        id SERIAL PRIMARY KEY,
        severity TEXT NOT NULL,
        step TEXT NOT NULL,
        message TEXT NOT NULL,
        peer_id TEXT,
        request_id TEXT,
        session_id TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_events_step ON events(step);
    CREATE INDEX IF NOT EXISTS idx_events_request ON events(request_id);

    CREATE TABLE IF NOT EXISTS llm_calls (
        id SERIAL PRIMARY KEY,
        call_type TEXT NOT NULL,
        duration_ms REAL NOT NULL,
        succeeded BOOLEAN NOT NULL,
        peer_id TEXT,
        request_id TEXT,
        session_id TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_llm_calls_type ON llm_calls(call_type);

    CREATE TABLE IF NOT EXISTS embedding_calls (
        id SERIAL PRIMARY KEY,
        purpose TEXT NOT NULL,
        duration_ms REAL NOT NULL,
        dimension INTEGER,
        succeeded BOOLEAN NOT NULL,
        entity_type TEXT,
        entity_id TEXT,
        peer_id TEXT,
        request_id TEXT,
        session_id TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_embedding_calls_purpose ON embedding_calls(purpose);
    """,
    # Lets a turn's own messages be looked up by request_id — see
    # core/stores/migrations.py's version 10 for the full rationale
    # (mirrors it exactly).
    4: """
    ALTER TABLE messages ADD COLUMN request_id TEXT;
    CREATE INDEX IF NOT EXISTS idx_messages_request ON messages(request_id);
    """,
}


def migrate(conn: psycopg.Connection, embedding_dim: int) -> None:
    """Run all pending migrations on the given connection.

    Safe to call on every connect() — only unapplied versions execute.
    ``embedding_dim`` parameterizes migration 1's ``VECTOR(n)`` columns
    (pgvector's vector width is fixed at column-creation time).
    """
    with conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        cur.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
        row = cur.fetchone()
        version = row[0] if row else 0
    conn.commit()

    for v in range(version + 1, max(POSTGRES_MIGRATIONS) + 1):
        if v not in POSTGRES_MIGRATIONS:
            continue
        sql = POSTGRES_MIGRATIONS[v].replace("{embedding_dim}", str(embedding_dim))
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (v,))
        conn.commit()
