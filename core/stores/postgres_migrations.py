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
