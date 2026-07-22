-- Eidolon Postgres schema (pgvector-backed).
--
-- A single consolidated schema, not a replay of the 7 SQLite migrations
-- in core/stores/migrations.py — there's no legacy Postgres data to carry
-- forward, so this expresses their current end-state directly with
-- native Postgres/pgvector types (VECTOR instead of BLOB, JSONB instead
-- of JSON-in-TEXT, TIMESTAMPTZ instead of naive TIMESTAMP, IDENTITY
-- instead of AUTOINCREMENT). See core/stores/postgres_migrations.py for
-- how this gets applied and versioned; the `schema_migrations` tracking
-- table itself is created there, not here (it must exist before this
-- script's own version can be recorded).
--
-- `{embedding_dim}` is substituted by PostgresStore.connect() from the
-- configured embedder's dimension before this script runs — pgvector's
-- VECTOR(n) width is fixed at column-creation time, so changing embedder
-- dimension later needs a real ALTER COLUMN TYPE migration, not a config
-- change.

CREATE EXTENSION IF NOT EXISTS vector;

-- topic_clusters before topics: topics.cluster_id references it.
CREATE TABLE IF NOT EXISTS topic_clusters (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    peer_id TEXT NOT NULL,
    label TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    embedding VECTOR({embedding_dim}),
    topic_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS topics (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    label TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    embedding VECTOR({embedding_dim}),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'closed')),
    peer_id TEXT,
    session_id TEXT,
    cluster_id INTEGER REFERENCES topic_clusters(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    turn_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    peer_id TEXT NOT NULL,
    topic_id INTEGER REFERENCES topics(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory_items (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content TEXT NOT NULL,
    item_type TEXT NOT NULL,
    source_peer TEXT NOT NULL,
    subject_peer TEXT NOT NULL,
    attribution_type TEXT NOT NULL,
    topic_id INTEGER REFERENCES topics(id),
    message_id INTEGER REFERENCES messages(id),
    embedding VECTOR({embedding_dim}),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    peer_id TEXT NOT NULL,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    source TEXT NOT NULL,
    importance TEXT NOT NULL DEFAULT 'medium' CHECK (importance IN ('low', 'medium', 'high')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (peer_id, category, key)
);

CREATE TABLE IF NOT EXISTS fact_consolidation_state (
    peer_id TEXT PRIMARY KEY,
    last_item_id INTEGER NOT NULL DEFAULT 0,
    last_run_at TIMESTAMPTZ
);

-- Exact-match / range / order-by indexes, mirroring SQLite migrations 5 + 6.
CREATE INDEX IF NOT EXISTS idx_messages_topic ON messages(topic_id);
CREATE INDEX IF NOT EXISTS idx_messages_peer_role ON messages(peer_id, role);
CREATE INDEX IF NOT EXISTS idx_topics_peer_status ON topics(peer_id, status);
CREATE INDEX IF NOT EXISTS idx_memory_items_subject_type ON memory_items(subject_peer, item_type);
CREATE INDEX IF NOT EXISTS idx_topics_cluster ON topics(cluster_id);
CREATE INDEX IF NOT EXISTS idx_topic_clusters_peer ON topic_clusters(peer_id);

-- Native ANN indexes (HNSW, cosine — matches the cosine semantics used
-- everywhere else in this project). No extension-loading fallback branch
-- exists or is needed here, unlike core/stores/vec_index.py's SQLite
-- sibling — pgvector is always available once this schema applies.
CREATE INDEX IF NOT EXISTS idx_topics_embedding_hnsw
    ON topics USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_items_embedding_hnsw
    ON memory_items USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_topic_clusters_embedding_hnsw
    ON topic_clusters USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL;
