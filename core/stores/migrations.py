"""Auto-migration for SQLite storage.

Uses ``PRAGMA user_version`` to track schema versions.
Each migration is a simple SQL script executed sequentially on startup.
"""

from __future__ import annotations

import sqlite3

# Schema migrations keyed by version number.
# Version 1: initial schema for V1.
# Future versions append new migrations here.
MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        peer_id TEXT NOT NULL,
        category TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        source TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(peer_id, category, key)
    );

    CREATE TABLE IF NOT EXISTS topics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT NOT NULL,
        summary TEXT DEFAULT '',
        tags TEXT DEFAULT '[]',
        embedding BLOB,
        status TEXT DEFAULT 'active',
        peer_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        closed_at TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS memory_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL,
        item_type TEXT NOT NULL,
        source_peer TEXT NOT NULL,
        subject_peer TEXT NOT NULL,
        attribution_type TEXT NOT NULL,
        topic_id INTEGER REFERENCES topics(id),
        embedding BLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """,
    2: """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        peer_id TEXT NOT NULL,
        topic_id INTEGER REFERENCES topics(id),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    ALTER TABLE memory_items ADD COLUMN message_id INTEGER REFERENCES messages(id);
    """,
}


def migrate(db: sqlite3.Connection) -> None:
    """Run all pending migrations on the given database connection.

    Safe to call on every startup — only unapplied versions are executed.
    """
    version = db.execute("PRAGMA user_version").fetchone()[0]

    for v in range(version + 1, max(MIGRATIONS) + 1):
        if v in MIGRATIONS:
            db.executescript(MIGRATIONS[v])
            db.execute(f"PRAGMA user_version = {v}")
