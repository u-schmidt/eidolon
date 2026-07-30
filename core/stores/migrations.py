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
    8: """
    -- Real, persisted per-topic message count ("mini-topic" soft-hide,
    -- see ROADMAP.md). Kept in sync incrementally by
    -- StorageProvider.store_message() (an explicit UPDATE alongside the
    -- INSERT INTO messages — no DB trigger), not recomputed at query
    -- time. Filtered in search_topics/get_recent_topics via a
    -- `message_count > :threshold` WHERE clause (config:
    -- retrieval.min_topic_messages) — a raw stored count rather than a
    -- boolean flag baked in at topic-close time, so changing the
    -- threshold later reclassifies every topic, old and new, for free,
    -- with no backfill ever needed again after this one.
    ALTER TABLE topics ADD COLUMN message_count INTEGER NOT NULL DEFAULT 0;

    UPDATE topics SET message_count = (
        SELECT COUNT(*) FROM messages WHERE messages.topic_id = topics.id
    );
    """,
    9: """
    -- Always-on operational metrics (see core/interfaces.py's
    -- MetricsRecorder) — deliberately separate from the debug-gated,
    -- JSON-payload trace mechanism in core/trace/: three typed tables,
    -- one per concern, always created and always written regardless of
    -- debug.enabled, so error/duration visibility never depends on a
    -- debug flag. Kept in the main schema (not a trace-style independent
    -- migration path) because they must always exist.
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        severity TEXT NOT NULL,
        step TEXT NOT NULL,
        message TEXT NOT NULL,
        peer_id TEXT,
        request_id TEXT,
        session_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_events_step ON events(step);
    CREATE INDEX IF NOT EXISTS idx_events_request ON events(request_id);

    CREATE TABLE IF NOT EXISTS llm_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        call_type TEXT NOT NULL,
        duration_ms REAL NOT NULL,
        succeeded INTEGER NOT NULL,
        peer_id TEXT,
        request_id TEXT,
        session_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_llm_calls_type ON llm_calls(call_type);

    CREATE TABLE IF NOT EXISTS embedding_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        purpose TEXT NOT NULL,
        duration_ms REAL NOT NULL,
        dimension INTEGER,
        succeeded INTEGER NOT NULL,
        entity_type TEXT,
        entity_id TEXT,
        peer_id TEXT,
        request_id TEXT,
        session_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_embedding_calls_purpose ON embedding_calls(purpose);
    """,
    10: """
    -- Lets a turn's own messages be looked up by request_id — the
    -- correlation key already threaded through every step of a turn
    -- (see core/__init__.py's store_turn). Without this, the turn-detail
    -- dashboard (tools/dashboard/) could only show real message text for
    -- turns captured by the debug-gated trace; with it, message content
    -- is always available regardless of debug.enabled, same as the
    -- events/llm_calls/embedding_calls tables above.
    ALTER TABLE messages ADD COLUMN request_id TEXT;
    CREATE INDEX IF NOT EXISTS idx_messages_request ON messages(request_id);
    """,
    11: """
    -- peer_id -> workspace_id: the concept this column always held was
    -- "which tenant/Hermes-profile this belongs to" (a hard partition
    -- key) — never "the other party in a conversation," which is what
    -- "peer" conventionally means (peer-to-peer, peer review). That
    -- second, genuinely different concept is memory_items.source_peer/
    -- subject_peer, deliberately left untouched here. "workspace" is
    -- the term this project's own Hermes-integration siblings already
    -- use for the identical boundary (Honcho's honcho.json "workspace"
    -- field) — not copied from Honcho, arrived at independently as the
    -- standard term (Slack/Notion/Linear use it the same way).
    ALTER TABLE topics RENAME COLUMN peer_id TO workspace_id;
    ALTER TABLE messages RENAME COLUMN peer_id TO workspace_id;
    ALTER TABLE facts RENAME COLUMN peer_id TO workspace_id;
    ALTER TABLE fact_consolidation_state RENAME COLUMN peer_id TO workspace_id;
    ALTER TABLE topic_clusters RENAME COLUMN peer_id TO workspace_id;
    ALTER TABLE events RENAME COLUMN peer_id TO workspace_id;
    ALTER TABLE llm_calls RENAME COLUMN peer_id TO workspace_id;
    ALTER TABLE embedding_calls RENAME COLUMN peer_id TO workspace_id;

    -- SQLite has no ALTER INDEX RENAME — drop and recreate under matching names.
    DROP INDEX IF EXISTS idx_messages_peer_role;
    CREATE INDEX IF NOT EXISTS idx_messages_workspace_role ON messages(workspace_id, role);
    DROP INDEX IF EXISTS idx_topics_peer_status;
    CREATE INDEX IF NOT EXISTS idx_topics_workspace_status ON topics(workspace_id, status);
    DROP INDEX IF EXISTS idx_topic_clusters_peer;
    CREATE INDEX IF NOT EXISTS idx_topic_clusters_workspace ON topic_clusters(workspace_id);
    """,
    12: """
    -- Named people/places/organizations mentioned in a topic, extracted
    -- alongside its tags at topic-close (see TopicSummarizer) but kept
    -- separate from the `tags` column deliberately: `tags` is capped at
    -- "3-5 key tags" for retrieval relevance, so a topic mentioning
    -- several names could crowd out real topic tags or have names
    -- themselves truncated. named_entities is never capped — every name
    -- survives here even though only a subset (if any) also gets merged
    -- into `tags`. This also gives a future per-entity profile feature a
    -- cheap, LLM-free way to count how often a name has recurred across
    -- topics (`SELECT ... WHERE named_entities LIKE ...`), which a
    -- capped, generic-word-mixed `tags` column can't answer reliably.
    ALTER TABLE topics ADD COLUMN named_entities TEXT DEFAULT '[]';
    """,
    13: """
    -- Per-entity profile cards ("fact-extraction hardening" Track 2) —
    -- mirrors the facts/fact_consolidation_state pair (see Migration 1/3)
    -- for a per-named-entity fact profile instead of a per-workspace one.
    --
    -- entity_scan_state: workspace-wide cursor over closed topics already
    -- scanned for named_entities mentions. Independent of any one
    -- entity's own consolidation progress below, since topics are
    -- scanned in id order regardless of which entities they mention.
    CREATE TABLE IF NOT EXISTS entity_scan_state (
        workspace_id TEXT PRIMARY KEY,
        last_topic_id INTEGER NOT NULL DEFAULT 0,
        last_run_at TIMESTAMP
    );

    -- entity_profiles: one row per (workspace_id, entity_name), tracking
    -- how many distinct topics have mentioned it and whether it has
    -- crossed the materialization threshold into a real card yet.
    -- last_consolidated_topic_id starts at 0 (not yet consolidated) so
    -- an entity's first consolidation pass after materializing pulls its
    -- entire mention history, not just the batch that crossed the
    -- threshold — entities materialize at different times, so this
    -- can't share entity_scan_state's cursor above.
    CREATE TABLE IF NOT EXISTS entity_profiles (
        workspace_id TEXT NOT NULL,
        entity_name TEXT NOT NULL,
        mention_count INTEGER NOT NULL DEFAULT 0,
        materialized_at TIMESTAMP,
        last_consolidated_topic_id INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (workspace_id, entity_name)
    );

    -- entity_facts: mirrors facts exactly, plus entity_name.
    CREATE TABLE IF NOT EXISTS entity_facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workspace_id TEXT NOT NULL,
        entity_name TEXT NOT NULL,
        category TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        source TEXT NOT NULL,
        importance TEXT NOT NULL DEFAULT 'medium',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(workspace_id, entity_name, category, key)
    );
    """,
}


def migrate(db: sqlite3.Connection) -> None:
    """Run all pending migrations on the given database connection.

    Safe to call on every startup — only unapplied versions are executed.

    Each version's script (plus its own ``PRAGMA user_version`` bump) is
    wrapped in one explicit ``BEGIN``/``COMMIT`` and, on failure, rolled
    back — ``executescript`` on its own performs no implicit transaction
    control, and DDL statements passed to it run under SQLite's own
    per-statement autocommit, so a mid-script failure would otherwise
    leave earlier statements in that version already durably committed
    while ``user_version`` stays at the prior number (verified directly:
    without the explicit wrapping, a failing 3rd statement in a 4-statement
    script left the first two committed on disk).
    """
    version = db.execute("PRAGMA user_version").fetchone()[0]

    for v in range(version + 1, max(MIGRATIONS) + 1):
        if v not in MIGRATIONS:
            continue
        try:
            db.executescript(f"BEGIN;\n{MIGRATIONS[v]}\nPRAGMA user_version = {v};\nCOMMIT;")
        except sqlite3.Error:
            db.rollback()
            raise
