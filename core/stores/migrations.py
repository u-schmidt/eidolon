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
    3: """
    CREATE TABLE IF NOT EXISTS fact_consolidation_state (
        peer_id TEXT PRIMARY KEY,
        last_item_id INTEGER NOT NULL DEFAULT 0,
        last_run_at TIMESTAMP
    );
    """,
    4: """
    ALTER TABLE topics ADD COLUMN session_id TEXT;
    """,
    5: """
    -- Indexes for the exact-match / range / order-by query paths hit on
    -- every turn (session resume, message loads, turn counting) and during
    -- consolidation. These do NOT speed up the associative search
    -- (search_topics / search_memory_items), which score every row in
    -- Python — that needs a vector index, tracked separately.
    CREATE INDEX IF NOT EXISTS idx_messages_topic ON messages(topic_id);
    CREATE INDEX IF NOT EXISTS idx_messages_peer_role ON messages(peer_id, role);
    CREATE INDEX IF NOT EXISTS idx_topics_peer_status ON topics(peer_id, status);
    CREATE INDEX IF NOT EXISTS idx_memory_items_subject_type ON memory_items(subject_peer, item_type);
    """,
    6: """
    -- Recurring-topic clustering (Phase 2 workstream #5). A topic joins
    -- at most one cluster; once it does, it's excluded from individual
    -- topic retrieval (see search_topics/get_recent_topics) in favor of
    -- its cluster's consolidated summary.
    CREATE TABLE IF NOT EXISTS topic_clusters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        peer_id TEXT NOT NULL,
        label TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        tags TEXT DEFAULT '[]',
        embedding BLOB,
        topic_count INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    ALTER TABLE topics ADD COLUMN cluster_id INTEGER REFERENCES topic_clusters(id);

    CREATE INDEX IF NOT EXISTS idx_topics_cluster ON topics(cluster_id);
    CREATE INDEX IF NOT EXISTS idx_topic_clusters_peer ON topic_clusters(peer_id);
    """,
    7: """
    -- Lets the consolidator (and injection ordering) rank facts instead
    -- of treating the whole profile as equally important — see
    -- core/consolidation/fact_consolidator.py.
    ALTER TABLE facts ADD COLUMN importance TEXT NOT NULL DEFAULT 'medium';
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
