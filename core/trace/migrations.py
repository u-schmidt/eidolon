"""Auto-migration for the Eidolon trace database.

Same ``PRAGMA user_version`` approach as ``core/stores/migrations.py``, kept
as a separate small copy rather than a shared/generalized function — the
trace schema is deliberately independent of (and disposable relative to)
the main memory schema.
"""

from __future__ import annotations

import sqlite3

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT,
        session_id TEXT,
        event_type TEXT NOT NULL,
        payload TEXT NOT NULL,
        duration_ms REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_events_request ON events(request_id);
    CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
    """,
}


def migrate(db: sqlite3.Connection) -> None:
    """Run all pending migrations on the given trace database connection.

    Safe to call on every startup — only unapplied versions are executed.
    """
    version = db.execute("PRAGMA user_version").fetchone()[0]

    for v in range(version + 1, max(MIGRATIONS) + 1):
        if v in MIGRATIONS:
            db.executescript(MIGRATIONS[v])
            db.execute(f"PRAGMA user_version = {v}")
