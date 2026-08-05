"""SQLite storage backend for Eidolon.

Implements the ``StorageProvider`` protocol using SQLite with:
- Auto-migration via ``PRAGMA user_version``
- JSON tags for flexible topic tagging
- BLOB embedding storage (source of truth) + an optional sqlite-vec
  index (``VecIndex``) for fast workspace-partitioned KNN search
- Cosine-similarity brute-force fallback when the index isn't available
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from datetime import UTC, datetime
from pathlib import Path

from core.interfaces import Tracer
from core.models import Fact, MemoryItem, MemoryItemBudget
from core.stores.migrations import migrate
from core.stores.retrieval_scorer import RetrievalScorer, _cosine_similarity
from core.stores.vec_index import VecIndex

logger = logging.getLogger(__name__)

# When VecIndex is available, how many nearest-by-embedding candidates to
# pre-fetch before the existing tag_overlap/temporal re-ranking runs over
# them — generous enough that re-ranking has real headroom to reorder, not
# just the single closest semantic match. Irrelevant to the brute-force
# path, which already considers every candidate.
_KNN_PREFILTER_MIN = 50
_KNN_PREFILTER_MULTIPLIER = 10


def _knn_prefilter_k(limit: int) -> int:
    return max(limit * _KNN_PREFILTER_MULTIPLIER, _KNN_PREFILTER_MIN)


def _parse_sqlite_timestamp(raw: str | None) -> datetime | None:
    """Parse a raw SQLite ``CURRENT_TIMESTAMP`` string (e.g.
    "2026-07-20 16:57:00", stored in UTC) into a tz-aware datetime.
    None for a missing/unparseable value — the caller treats that as
    "no known activity", not "definitely stale"."""
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)


def _serialize_embedding(vec: list[float]) -> bytes:
    """Pack a float list into a binary BLOB (little-endian f64)."""
    return struct.pack(f"<{len(vec)}d", *vec)


def _deserialize_embedding(blob: bytes) -> list[float]:
    """Unpack a binary BLOB back into a float list."""
    count = len(blob) // 8
    return list(struct.unpack(f"<{count}d", blob))


class SqliteStore:
    """SQLite implementation of ``StorageProvider``.

    Parameters
    ----------
    path : str
        Filesystem path to the SQLite database file.
    use_vec_index : bool
        Whether to try loading the optional sqlite-vec extension for
        accelerated KNN search. Defaults to True; every search method
        falls back to the brute-force in-Python scan automatically if
        loading fails (not installed, or the host SQLite build lacks
        extension-loading support) — see ``VecIndex``. Set False to
        force the brute-force path regardless of what's installed
        (used by tests that need deterministic behavior independent of
        the environment).
    """

    def __init__(self, path: str, use_vec_index: bool = True) -> None:
        self._path = Path(path).expanduser().resolve()
        self._db: sqlite3.Connection | None = None
        self._use_vec_index = use_vec_index
        self._vec_index: VecIndex | None = None

    # ── Lifecycle ──────────────────────────────────────────

    def connect(self) -> None:
        """Open the database, run pending migrations, and (if enabled)
        load the optional vector index."""
        # Ensure the parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._path), check_same_thread=False)
        # Not WAL: WAL needs reliable shared-memory (mmap) coordination
        # between connections, which network/bridged filesystems (NFS,
        # SMB, and — as seen in production — Docker Desktop's virtiofs
        # bind mounts) don't guarantee. There, WAL produces exactly the
        # intermittent "database disk image is malformed" errors this
        # caused; the plain rollback journal has no such dependency.
        # This deployment is effectively single-writer, so WAL's main
        # benefit (readers never block a writer) isn't worth that risk.
        self._db.execute("PRAGMA journal_mode=DELETE")
        self._db.execute("PRAGMA foreign_keys=ON")
        migrate(self._db)

        self._vec_index = VecIndex(self._db)
        if self._use_vec_index:
            self._vec_index.try_load()
            if self._vec_index.available:
                self._backfill_vec_index()

    @property
    def vec_index(self) -> VecIndex:
        if self._vec_index is None:
            raise RuntimeError("SqliteStore not connected — call connect() first")
        return self._vec_index

    def _backfill_vec_index(self) -> None:
        """Sync every embedded row into the vector index.

        Runs on every connect() where the index is available — not
        gated on a one-time migration flag, so installing the
        sqlite-vec extra after already having data still gets
        backfilled rather than silently invisible to search forever.
        VecIndex.upsert() is idempotent, so re-running this against
        already-synced rows is a correctness no-op; the cost is a
        one-time pass at process startup, not a per-query cost.
        """
        vi = self.vec_index
        for topic_id, workspace_id, blob, cluster_id in self._conn.execute(
            "SELECT id, workspace_id, embedding, cluster_id FROM topics WHERE embedding IS NOT NULL"
        ).fetchall():
            if workspace_id is None:
                continue  # can't partition a vector search without a workspace_id
            vi.upsert("vec_topics", topic_id, workspace_id, _deserialize_embedding(blob), cluster_id=cluster_id)

        for item_id, workspace_id, blob in self._conn.execute(
            "SELECT mi.id, m.workspace_id, mi.embedding FROM memory_items mi "
            "JOIN messages m ON mi.message_id = m.id WHERE mi.embedding IS NOT NULL"
        ).fetchall():
            vi.upsert("vec_memory_items", item_id, workspace_id, _deserialize_embedding(blob))

        for cluster_id, workspace_id, blob in self._conn.execute(
            "SELECT id, workspace_id, embedding FROM topic_clusters WHERE embedding IS NOT NULL"
        ).fetchall():
            vi.upsert("vec_clusters", cluster_id, workspace_id, _deserialize_embedding(blob))

        self._conn.commit()

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def _conn(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("SqliteStore not connected — call connect() first")
        return self._db

    # ── Facts ───────────────────────────────────────────────

    def get_facts(self, workspace_id: str) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT workspace_id, category, key, value, source, importance, created_at "
            "FROM facts WHERE workspace_id = ? "
            "ORDER BY CASE importance WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, category, key",
            (workspace_id,),
        ).fetchall()
        return [
            Fact(workspace_id=r[0], category=r[1], key=r[2], value=r[3], source=r[4], importance=r[5], created_at=r[6])
            for r in rows
        ]

    def get_all_facts(self) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT workspace_id, category, key, value, source, importance, created_at "
            "FROM facts ORDER BY workspace_id, category, key",
        ).fetchall()
        return [
            Fact(workspace_id=r[0], category=r[1], key=r[2], value=r[3], source=r[4], importance=r[5], created_at=r[6])
            for r in rows
        ]

    def set_fact(
        self, workspace_id: str, category: str, key: str, value: str, source: str, importance: str = "medium"
    ) -> None:
        self._conn.execute(
            "INSERT INTO facts (workspace_id, category, key, value, source, importance) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(workspace_id, category, key) DO UPDATE SET "
            "value=excluded.value, source=excluded.source, importance=excluded.importance",
            (workspace_id, category, key, value, source, importance),
        )
        self._conn.commit()

    def delete_fact(self, workspace_id: str, category: str, key: str) -> None:
        self._conn.execute(
            "DELETE FROM facts WHERE workspace_id = ? AND category = ? AND key = ?",
            (workspace_id, category, key),
        )
        self._conn.commit()

    # ── Facts Consolidation ────────────────────────────────

    def count_user_turns(self, workspace_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE workspace_id = ? AND role = 'user'",
            (workspace_id,),
        ).fetchone()
        return row[0]

    def get_fact_items_since(self, subject_peer: str, since_item_id: int) -> list[MemoryItem]:
        rows = self._conn.execute(
            "SELECT id, content, item_type, source_peer, subject_peer, "
            "attribution_type, topic_id, message_id, created_at "
            "FROM memory_items "
            "WHERE subject_peer = ? AND id > ? "
            "ORDER BY id ASC",
            (subject_peer, since_item_id),
        ).fetchall()
        return [
            MemoryItem(
                id=r[0], content=r[1], item_type=r[2], source_peer=r[3], subject_peer=r[4],
                attribution_type=r[5], topic_id=r[6], message_id=r[7], timestamp=r[8],
            )
            for r in rows
        ]

    def get_consolidation_checkpoint(self, workspace_id: str) -> int:
        row = self._conn.execute(
            "SELECT last_item_id FROM fact_consolidation_state WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchone()
        return row[0] if row else 0

    def set_consolidation_checkpoint(self, workspace_id: str, last_item_id: int) -> None:
        self._conn.execute(
            "INSERT INTO fact_consolidation_state (workspace_id, last_item_id, last_run_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(workspace_id) DO UPDATE SET last_item_id=excluded.last_item_id, last_run_at=excluded.last_run_at",
            (workspace_id, last_item_id),
        )
        self._conn.commit()

    # ── Entity Profiles ────────────────────────────────────

    def get_topics_since(self, workspace_id: str, since_topic_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, summary, named_entities FROM topics "
            "WHERE workspace_id = ? AND status = 'closed' AND id > ? "
            "ORDER BY id ASC",
            (workspace_id, since_topic_id),
        ).fetchall()
        return [
            {"id": r[0], "label": r[1], "summary": r[2], "named_entities": json.loads(r[3])}
            for r in rows
        ]

    def increment_entity_mentions(self, workspace_id: str, entity_names: list[str]) -> None:
        if not entity_names:
            return
        self._conn.executemany(
            "INSERT INTO entity_profiles (workspace_id, entity_name, mention_count) "
            "VALUES (?, ?, 1) "
            "ON CONFLICT(workspace_id, entity_name) DO UPDATE SET mention_count = mention_count + 1",
            [(workspace_id, name) for name in entity_names],
        )
        self._conn.commit()

    def materialize_entities(self, workspace_id: str, threshold: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT entity_name FROM entity_profiles "
            "WHERE workspace_id = ? AND materialized_at IS NULL AND mention_count >= ?",
            (workspace_id, threshold),
        ).fetchall()
        names = [r[0] for r in rows]
        if names:
            self._conn.executemany(
                "UPDATE entity_profiles SET materialized_at = CURRENT_TIMESTAMP "
                "WHERE workspace_id = ? AND entity_name = ?",
                [(workspace_id, name) for name in names],
            )
            self._conn.commit()
        return names

    def get_materialized_entity_names(self, workspace_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT entity_name FROM entity_profiles WHERE workspace_id = ? AND materialized_at IS NOT NULL",
            (workspace_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_topics_mentioning_entity(
        self, workspace_id: str, entity_name: str, since_topic_id: int
    ) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, summary, named_entities FROM topics "
            "WHERE workspace_id = ? AND status = 'closed' AND id > ? "
            "ORDER BY id ASC",
            (workspace_id, since_topic_id),
        ).fetchall()
        result = []
        for r in rows:
            named_entities = json.loads(r[3])
            if entity_name in named_entities:
                result.append({"id": r[0], "label": r[1], "summary": r[2], "named_entities": named_entities})
        return result

    def get_entity_facts(self, workspace_id: str, entity_names: list[str]) -> list[Fact]:
        if not entity_names:
            return []
        placeholders = ",".join("?" * len(entity_names))
        rows = self._conn.execute(
            "SELECT workspace_id, category, key, value, source, importance, created_at, entity_name "
            f"FROM entity_facts WHERE workspace_id = ? AND entity_name IN ({placeholders}) "
            "ORDER BY entity_name, CASE importance WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, category, key",
            (workspace_id, *entity_names),
        ).fetchall()
        return [
            Fact(
                workspace_id=r[0], category=r[1], key=r[2], value=r[3], source=r[4],
                importance=r[5], created_at=r[6], entity_name=r[7],
            )
            for r in rows
        ]

    def set_entity_fact(
        self,
        workspace_id: str,
        entity_name: str,
        category: str,
        key: str,
        value: str,
        source: str,
        importance: str = "medium",
    ) -> None:
        self._conn.execute(
            "INSERT INTO entity_facts (workspace_id, entity_name, category, key, value, source, importance) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(workspace_id, entity_name, category, key) DO UPDATE SET "
            "value=excluded.value, source=excluded.source, importance=excluded.importance",
            (workspace_id, entity_name, category, key, value, source, importance),
        )
        self._conn.commit()

    def delete_entity_fact(self, workspace_id: str, entity_name: str, category: str, key: str) -> None:
        self._conn.execute(
            "DELETE FROM entity_facts WHERE workspace_id = ? AND entity_name = ? AND category = ? AND key = ?",
            (workspace_id, entity_name, category, key),
        )
        self._conn.commit()

    def list_entity_profiles(self, workspace_id: str | None = None) -> list[dict]:
        query = (
            "SELECT ep.workspace_id, ep.entity_name, ep.mention_count, ep.materialized_at, "
            "ep.last_consolidated_topic_id, COUNT(ef.id) AS fact_count "
            "FROM entity_profiles ep LEFT JOIN entity_facts ef "
            "ON ef.workspace_id = ep.workspace_id AND ef.entity_name = ep.entity_name"
        )
        params: list[str] = []
        if workspace_id is not None:
            query += " WHERE ep.workspace_id = ?"
            params.append(workspace_id)
        query += (
            " GROUP BY ep.workspace_id, ep.entity_name, ep.mention_count, ep.materialized_at, "
            "ep.last_consolidated_topic_id "
            "ORDER BY ep.mention_count DESC, ep.entity_name ASC"
        )
        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "workspace_id": r[0], "entity_name": r[1], "mention_count": r[2],
                "materialized_at": r[3], "last_consolidated_topic_id": r[4], "fact_count": r[5],
            }
            for r in rows
        ]

    def get_entity_scan_checkpoint(self, workspace_id: str) -> int:
        row = self._conn.execute(
            "SELECT last_topic_id FROM entity_scan_state WHERE workspace_id = ?",
            (workspace_id,),
        ).fetchone()
        return row[0] if row else 0

    def set_entity_scan_checkpoint(self, workspace_id: str, last_topic_id: int) -> None:
        self._conn.execute(
            "INSERT INTO entity_scan_state (workspace_id, last_topic_id, last_run_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(workspace_id) DO UPDATE SET "
            "last_topic_id=excluded.last_topic_id, last_run_at=excluded.last_run_at",
            (workspace_id, last_topic_id),
        )
        self._conn.commit()

    def get_entity_consolidation_checkpoint(self, workspace_id: str, entity_name: str) -> int:
        row = self._conn.execute(
            "SELECT last_consolidated_topic_id FROM entity_profiles WHERE workspace_id = ? AND entity_name = ?",
            (workspace_id, entity_name),
        ).fetchone()
        return row[0] if row else 0

    def set_entity_consolidation_checkpoint(
        self, workspace_id: str, entity_name: str, last_topic_id: int
    ) -> None:
        self._conn.execute(
            "UPDATE entity_profiles SET last_consolidated_topic_id = ? "
            "WHERE workspace_id = ? AND entity_name = ?",
            (last_topic_id, workspace_id, entity_name),
        )
        self._conn.commit()

    # ── Topics ─────────────────────────────────────────────

    def create_topic(
        self,
        label: str,
        summary: str = "",
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
        workspace_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        tags_json = json.dumps(tags or [])
        embed_blob = _serialize_embedding(embedding) if embedding else None
        cur = self._conn.execute(
            "INSERT INTO topics (label, summary, tags, embedding, workspace_id, session_id) VALUES (?, ?, ?, ?, ?, ?)",
            (label, summary, tags_json, embed_blob, workspace_id, session_id),
        )
        self._conn.commit()
        topic_id: int = cur.lastrowid  # type: ignore[assignment]
        if embedding and workspace_id:
            self.vec_index.upsert("vec_topics", topic_id, workspace_id, embedding, cluster_id=None)
        return topic_id

    def get_topic(self, topic_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, label, summary, tags, status, workspace_id, created_at, closed_at, message_count, "
            "named_entities FROM topics WHERE id = ?",
            (topic_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "label": row[1],
            "summary": row[2],
            "tags": json.loads(row[3]),
            "status": row[4],
            "workspace_id": row[5],
            "created_at": row[6],
            "closed_at": row[7],
            "message_count": row[8],
            "named_entities": json.loads(row[9]),
        }

    def close_topic(self, topic_id: int, summary: str) -> None:
        self._conn.execute(
            "UPDATE topics SET status='closed', summary=?, closed_at=CURRENT_TIMESTAMP WHERE id=?",
            (summary, topic_id),
        )
        self._conn.commit()

    def get_active_topic(self, workspace_id: str, session_id: str | None = None) -> int | None:
        """Return the id of the most recently opened 'active' topic for
        this workspace *and* session.

        Scoped per (workspace_id, session_id) — a workspace with several concurrent
        sessions (web, Telegram, CLI, ...) gets one independent active
        topic per session, not one shared across all of them. Passing no
        session_id matches only topics created with none (e.g. legacy
        rows from before this column existed, or single-session callers
        like the CLI) — it is not a wildcard across sessions.

        If more than one is somehow left 'active' for the same
        (workspace_id, session_id) (e.g. a crash mid topic transition), the
        most recent wins; older orphans stay open until explicitly
        closed — no automatic cleanup here.
        """
        # `id DESC` breaks ties from CURRENT_TIMESTAMP's second-level
        # resolution — two topics created within the same second would
        # otherwise sort ambiguously.
        row = self._conn.execute(
            "SELECT id FROM topics WHERE workspace_id = ? AND session_id IS ? AND status = 'active' "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (workspace_id, session_id),
        ).fetchone()
        return row[0] if row else None

    def get_stale_active_topics(self, cutoff: datetime, workspace_id: str | None = None) -> list[dict]:
        """Return 'active' topics whose last message predates `cutoff`,
        oldest first — candidates for graceful cleanup (see
        `Eidolon.close_stale_topics`) when a topic was never properly
        closed (a crashed session, an adapter that stopped calling
        `store_turn`).

        A topic's own `created_at` stands in for "last activity" when it
        has no messages yet (e.g. a topic row created but never actually
        used). `workspace_id=None` scans every workspace's active topics.
        """
        query = (
            "SELECT t.id, t.label, t.summary, t.tags, t.status, t.workspace_id, t.created_at, t.closed_at, "
            "COALESCE(MAX(m.created_at), t.created_at) AS last_activity "
            "FROM topics t LEFT JOIN messages m ON m.topic_id = t.id "
            "WHERE t.status = 'active'"
        )
        params: list[str] = []
        if workspace_id is not None:
            query += " AND t.workspace_id = ?"
            params.append(workspace_id)
        query += " GROUP BY t.id ORDER BY last_activity ASC"
        rows = self._conn.execute(query, params).fetchall()

        stale = []
        for row in rows:
            last_activity = _parse_sqlite_timestamp(row[8])
            if last_activity is not None and last_activity >= cutoff:
                continue
            stale.append({
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": json.loads(row[3]),
                "status": row[4],
                "workspace_id": row[5],
                "created_at": row[6],
                "closed_at": row[7],
                "last_activity": row[8],
            })
        return stale

    def get_recent_topics(self, workspace_id: str, limit: int, min_message_count: int = -1) -> list[dict]:
        """Return `workspace_id`'s most recently closed topics, newest first.

        Ordered purely by `closed_at` — no relevance scoring, unlike
        `search_topics`. `id DESC` as a tiebreak for the same reason as
        `get_active_topic`: CURRENT_TIMESTAMP only has second resolution.
        """
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, embedding, status, workspace_id, created_at, closed_at "
            "FROM topics WHERE status='closed' AND cluster_id IS NULL AND workspace_id = ? "
            "AND message_count > ? "
            "ORDER BY closed_at DESC, id DESC LIMIT ?",
            (workspace_id, min_message_count, limit),
        ).fetchall()
        return [
            {
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": json.loads(row[3]),
                "status": row[5],
                "workspace_id": row[6],
                "created_at": row[7],
                "closed_at": row[8],
            }
            for row in rows
        ]

    def get_closed_topics_since(self, since: datetime) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, workspace_id, label, created_at, closed_at, message_count "
            "FROM topics WHERE status = 'closed' ORDER BY closed_at ASC"
        ).fetchall()
        result = []
        for row in rows:
            closed_at = _parse_sqlite_timestamp(row[4])
            if closed_at is None or closed_at < since:
                continue
            result.append({
                "id": row[0],
                "workspace_id": row[1],
                "label": row[2],
                "created_at": row[3],
                "closed_at": row[4],
                "message_count": row[5],
            })
        return result

    def list_topics(self, status: str | None = None, workspace_id: str | None = None) -> list[dict]:
        query = (
            "SELECT id, label, status, workspace_id, cluster_id, created_at, closed_at, message_count "
            "FROM topics"
        )
        conditions = []
        params: list[str] = []
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if workspace_id is not None:
            conditions.append("workspace_id = ?")
            params.append(workspace_id)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC, id DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "id": row[0],
                "label": row[1],
                "status": row[2],
                "workspace_id": row[3],
                "cluster_id": row[4],
                "created_at": row[5],
                "closed_at": row[6],
                "message_count": row[7],
            }
            for row in rows
        ]

    def update_topic_embedding(
        self, topic_id: int, embedding: list[float]
    ) -> None:
        embed_blob = _serialize_embedding(embedding)
        self._conn.execute(
            "UPDATE topics SET embedding = ? WHERE id = ?",
            (embed_blob, topic_id),
        )
        self._conn.commit()
        # workspace_id/cluster_id aren't parameters here (this method predates
        # the vector index) — cheap lookup rather than changing the
        # public signature every other caller already depends on.
        row = self._conn.execute(
            "SELECT workspace_id, cluster_id FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
        if row and row[0]:
            self.vec_index.upsert("vec_topics", topic_id, row[0], embedding, cluster_id=row[1])

    def update_topic_tags(self, topic_id: int, tags: list[str]) -> None:
        self._conn.execute(
            "UPDATE topics SET tags = ? WHERE id = ?",
            (json.dumps(tags), topic_id),
        )
        self._conn.commit()

    def update_topic_entities(self, topic_id: int, named_entities: list[str]) -> None:
        self._conn.execute(
            "UPDATE topics SET named_entities = ? WHERE id = ?",
            (json.dumps(named_entities), topic_id),
        )
        self._conn.commit()

    def update_topic_label(self, topic_id: int, label: str) -> None:
        self._conn.execute(
            "UPDATE topics SET label = ? WHERE id = ?",
            (label, topic_id),
        )
        self._conn.commit()

    def count_messages_by_topic(self, topic_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE topic_id = ?",
            (topic_id,),
        ).fetchone()
        return row[0]

    def _semantic_similarity(
        self,
        semantic_lookup: dict[int, float] | None,
        candidate_id: int,
        query_embedding: list[float] | None,
        embedding_blob: bytes | None,
    ) -> float:
        """Resolve one candidate's raw semantic similarity: the vector
        index's pre-computed value on the accelerated path, otherwise a
        cosine over the deserialized BLOB, otherwise 0.0 (no embedding to
        compare). Shared by all four search methods so the scorer never has
        to know how a row's embedding was stored or fetched."""
        if semantic_lookup is not None:
            return semantic_lookup.get(candidate_id, 0.0)
        if query_embedding and embedding_blob:
            return _cosine_similarity(query_embedding, _deserialize_embedding(embedding_blob))
        return 0.0

    def search_topics(
        self,
        query_embedding: list[float] | None = None,
        tags: list[str] | None = None,
        workspace_id: str | None = None,
        limit: int = 5,
        weights: dict[str, float] | None = None,
        tracer: Tracer | None = None,
        request_id: str | None = None,
        now: datetime | None = None,
        min_message_count: int = -1,
    ) -> list[dict]:
        """Search topics by embedding similarity, tag overlap, workspace, and recency.

        `workspace_id`, when given, is a hard SQL filter — only that workspace's closed,
        unclustered topics are ever scanned or scored, both to bound the
        scan cost and to prevent a different workspace's topic from ever
        surfacing in this workspace's results regardless of how well it scores.
        `workspace_id=None` searches every workspace (used by callers that
        intentionally want that).

        `weights` overrides the semantic/tag_overlap/person/temporal scoring
        weights when provided (see `RetrievalScorer`). `person` still
        contributes as before — with `workspace_id` now a hard filter it's
        satisfied by every surviving candidate, so it no longer
        differentiates ranking, but its score contribution is unchanged for
        backward compatibility. The `temporal` term is a recency
        *re-ranking* signal: it's added only to candidates that already
        have some semantic/tag/person relevance, so it re-orders relevant
        topics by recency but never pulls in a merely-recent one. `now` is
        the reference time for the recency decay (defaults to the current
        UTC time); a topic's recency uses its `closed_at` (last active)
        falling back to `created_at`.

        When `tracer` is given, one `retrieval_scoring_summary` event is
        recorded for the whole call (see `RetrievalScorer.finalize`), carrying
        the top positively-scored candidates (with breakdown) plus
        total/included counts — not one event per candidate, which doesn't
        scale with DB size.

        `min_message_count` soft-hides "mini-topics" — see
        StorageProvider.search_topics's docstring. Applied in every
        branch's row-fetch SQL, before RetrievalScorer ever sees a
        candidate, not as a post-fetch discard (the accelerated path's
        vec0 KNN step itself has no message_count column to filter on,
        so this lands in the row-fetch that follows it — still well
        before scoring/ranking).

        Returns topic dicts ordered by relevance (highest first).
        """
        scorer = RetrievalScorer(weights, now)
        # Hard-filtered in SQL when given — not just a soft scoring bonus.
        # Without this, a different workspace's highly-relevant-by-content topic
        # could outscore and surface ahead of (or alongside) the querying
        # workspace's own topics: a real cross-workspace context leak, not just a
        # performance concern. `workspace_id=None` keeps the old "search
        # everyone" behavior for callers that intentionally omit it.
        #
        # When the vector index is available and there's a real query to
        # search by, it pre-filters to the nearest-by-embedding candidates
        # (see _knn_prefilter_k) so the scan below runs over that small set
        # instead of every one of the workspace's topics — the accelerated path.
        # A pre-computed similarity for each candidate comes along with it,
        # so the expensive BLOB-deserialize-and-cosine step is skipped
        # entirely for those rows.
        semantic_lookup: dict[int, float] | None = None
        if self.vec_index.available and query_embedding and workspace_id:
            knn_hits = self.vec_index.knn("vec_topics", workspace_id, query_embedding, k=_knn_prefilter_k(limit))
            semantic_lookup = dict(knn_hits)
            if not semantic_lookup:
                rows = []
            else:
                placeholders = ",".join("?" * len(semantic_lookup))
                rows = self._conn.execute(
                    f"SELECT id, label, summary, tags, embedding, status, workspace_id, created_at, closed_at "
                    f"FROM topics WHERE id IN ({placeholders}) AND status='closed' AND cluster_id IS NULL "
                    f"AND message_count > ?",
                    (*semantic_lookup.keys(), min_message_count),
                ).fetchall()
        elif workspace_id:
            rows = self._conn.execute(
                "SELECT id, label, summary, tags, embedding, status, workspace_id, created_at, closed_at "
                "FROM topics WHERE status='closed' AND cluster_id IS NULL AND workspace_id = ? "
                "AND message_count > ?",
                (workspace_id, min_message_count),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, label, summary, tags, embedding, status, workspace_id, created_at, closed_at "
                "FROM topics WHERE status='closed' AND cluster_id IS NULL AND message_count > ?",
                (min_message_count,),
            ).fetchall()

        scored: list[tuple[float, dict, dict]] = []
        for row in rows:
            topic = {
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": json.loads(row[3]),
                "status": row[5],
                "workspace_id": row[6],
                "created_at": row[7],
                "closed_at": row[8],
            }
            score, breakdown = scorer.score(
                candidate_type="topic",
                candidate_id=topic["id"],
                semantic_raw=self._semantic_similarity(semantic_lookup, topic["id"], query_embedding, row[4]),
                recency_timestamp=topic["closed_at"] or topic["created_at"],
                include_tag_overlap=True,
                query_tags=tags,
                candidate_tags=topic["tags"],
                include_person=True,
                # Not workspace_id-derived: this row's workspace_id already equals the
                # query's (the WHERE clause above guarantees it when workspace_id
                # is given, and bool(workspace_id) is False otherwise), so
                # comparing them again is a tautology, not a signal — same
                # workspace-vs-subject conflation search_memory_items' docstring
                # already documents removing once (a topic's *identity*
                # workspace_id was being used as a stand-in for whether it's
                # topically about the same *person/subject*, two different
                # things). Left as a hard False — dormant, not deleted —
                # until there's an actual subject-relevance signal to
                # compute here; person_weighted stays 0.0 either way.
                person_match=False,
            )
            scored.append((score, topic, breakdown))

        return scorer.finalize(
            scored=scored, candidate_type="topic", limit=limit, tracer=tracer, request_id=request_id,
        )

    # ── Topic Clusters ─────────────────────────────────────

    def get_clusters(self, workspace_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, workspace_id, label, summary, tags, milestones, topic_count, created_at, updated_at "
            "FROM topic_clusters WHERE workspace_id = ? ORDER BY id",
            (workspace_id,),
        ).fetchall()
        return [
            {
                "id": row[0],
                "workspace_id": row[1],
                "label": row[2],
                "summary": row[3],
                "tags": json.loads(row[4]),
                "milestones": json.loads(row[5]),
                "topic_count": row[6],
                "created_at": row[7],
                "updated_at": row[8],
            }
            for row in rows
        ]

    def list_clusters(self, workspace_id: str | None = None) -> list[dict]:
        query = (
            "SELECT id, workspace_id, label, summary, tags, milestones, topic_count, created_at, updated_at "
            "FROM topic_clusters"
        )
        params: list[str] = []
        if workspace_id is not None:
            query += " WHERE workspace_id = ?"
            params.append(workspace_id)
        query += " ORDER BY created_at DESC, id DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "id": row[0],
                "workspace_id": row[1],
                "label": row[2],
                "summary": row[3],
                "tags": json.loads(row[4]),
                "milestones": json.loads(row[5]),
                "topic_count": row[6],
                "created_at": row[7],
                "updated_at": row[8],
            }
            for row in rows
        ]

    def create_cluster(
        self,
        workspace_id: str,
        label: str,
        summary: str,
        tags: list[str] | None = None,
        milestones: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> int:
        tags_json = json.dumps(tags or [])
        milestones_json = json.dumps(milestones or [])
        embed_blob = _serialize_embedding(embedding) if embedding else None
        cur = self._conn.execute(
            "INSERT INTO topic_clusters (workspace_id, label, summary, tags, milestones, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (workspace_id, label, summary, tags_json, milestones_json, embed_blob),
        )
        self._conn.commit()
        cluster_id: int = cur.lastrowid  # type: ignore[assignment]
        if embedding:
            self.vec_index.upsert("vec_clusters", cluster_id, workspace_id, embedding)
        return cluster_id

    def get_cluster(self, cluster_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, workspace_id, label, summary, tags, milestones, topic_count, created_at, updated_at "
            "FROM topic_clusters WHERE id = ?",
            (cluster_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "workspace_id": row[1],
            "label": row[2],
            "summary": row[3],
            "tags": json.loads(row[4]),
            "milestones": json.loads(row[5]),
            "topic_count": row[6],
            "created_at": row[7],
            "updated_at": row[8],
        }

    def update_cluster(
        self,
        cluster_id: int,
        summary: str,
        label: str | None = None,
        tags: list[str] | None = None,
        milestones: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        sets = ["summary = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: list = [summary]
        if label is not None:
            sets.append("label = ?")
            params.append(label)
        if tags is not None:
            sets.append("tags = ?")
            params.append(json.dumps(tags))
        if milestones is not None:
            sets.append("milestones = ?")
            params.append(json.dumps(milestones))
        if embedding is not None:
            sets.append("embedding = ?")
            params.append(_serialize_embedding(embedding))
        params.append(cluster_id)
        self._conn.execute(f"UPDATE topic_clusters SET {', '.join(sets)} WHERE id = ?", params)
        self._conn.commit()
        if embedding is not None:
            workspace_row = self._conn.execute(
                "SELECT workspace_id FROM topic_clusters WHERE id = ?", (cluster_id,)
            ).fetchone()
            if workspace_row and workspace_row[0]:
                self.vec_index.upsert("vec_clusters", cluster_id, workspace_row[0], embedding)

    def assign_topic_to_cluster(self, topic_id: int, cluster_id: int) -> None:
        self._conn.execute("UPDATE topics SET cluster_id = ? WHERE id = ?", (cluster_id, topic_id))
        self._conn.execute(
            "UPDATE topic_clusters SET topic_count = topic_count + 1 WHERE id = ?", (cluster_id,)
        )
        self._conn.commit()
        self.vec_index.set_cluster_id("vec_topics", topic_id, cluster_id)

    def get_topics_by_cluster(self, cluster_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, status, workspace_id, created_at, closed_at "
            "FROM topics WHERE cluster_id = ? ORDER BY closed_at ASC, id ASC",
            (cluster_id,),
        ).fetchall()
        return [
            {
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": json.loads(row[3]),
                "status": row[4],
                "workspace_id": row[5],
                "created_at": row[6],
                "closed_at": row[7],
            }
            for row in rows
        ]

    def get_recent_topics_in_cluster(self, cluster_id: int, limit: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, status, workspace_id, created_at, closed_at "
            "FROM topics WHERE cluster_id = ? ORDER BY closed_at DESC, id DESC LIMIT ?",
            (cluster_id, limit),
        ).fetchall()
        return [
            {
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": json.loads(row[3]),
                "status": row[4],
                "workspace_id": row[5],
                "created_at": row[6],
                "closed_at": row[7],
            }
            for row in rows
        ]

    def search_clusters(
        self,
        query_embedding: list[float] | None = None,
        tags: list[str] | None = None,
        workspace_id: str | None = None,
        limit: int = 5,
        weights: dict[str, float] | None = None,
        tracer: Tracer | None = None,
        request_id: str | None = None,
        now: datetime | None = None,
    ) -> list[dict]:
        """See StorageProvider.search_clusters. Same scoring shape as
        search_topics, but scanning topic_clusters and hard-filtering
        workspace_id in SQL instead of scoring it as a soft "person" bonus —
        clusters must never merge across workspaces.
        """
        if not workspace_id:
            return []
        scorer = RetrievalScorer(weights, now)

        # Accelerated path (see search_topics) — pre-filter to the
        # nearest-by-embedding clusters via the vector index.
        semantic_lookup: dict[int, float] | None = None
        if self.vec_index.available and query_embedding:
            knn_hits = self.vec_index.knn("vec_clusters", workspace_id, query_embedding, k=_knn_prefilter_k(limit))
            semantic_lookup = dict(knn_hits)
            if not semantic_lookup:
                rows = []
            else:
                placeholders = ",".join("?" * len(semantic_lookup))
                rows = self._conn.execute(
                    f"SELECT id, workspace_id, label, summary, tags, milestones, embedding, "
                    f"topic_count, created_at, updated_at "
                    f"FROM topic_clusters WHERE id IN ({placeholders})",
                    tuple(semantic_lookup.keys()),
                ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, workspace_id, label, summary, tags, milestones, embedding, "
                "topic_count, created_at, updated_at "
                "FROM topic_clusters WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchall()

        scored: list[tuple[float, dict, dict]] = []
        for row in rows:
            cluster = {
                "id": row[0],
                "workspace_id": row[1],
                "label": row[2],
                "summary": row[3],
                "tags": json.loads(row[4]),
                "milestones": json.loads(row[5]),
                "topic_count": row[7],
                "created_at": row[8],
                "updated_at": row[9],
            }
            # No "person" term — workspace_id is already a hard SQL filter above.
            score, breakdown = scorer.score(
                candidate_type="cluster",
                candidate_id=cluster["id"],
                semantic_raw=self._semantic_similarity(semantic_lookup, cluster["id"], query_embedding, row[6]),
                recency_timestamp=cluster["updated_at"],
                include_tag_overlap=True,
                query_tags=tags,
                candidate_tags=cluster["tags"],
            )
            scored.append((score, cluster, breakdown))

        return scorer.finalize(
            scored=scored, candidate_type="cluster", limit=limit, tracer=tracer, request_id=request_id,
        )

    def search_cluster_candidates(
        self,
        topic_id: int,
        workspace_id: str,
        limit: int = 5,
        weights: dict[str, float] | None = None,
        tracer: Tracer | None = None,
        request_id: str | None = None,
        now: datetime | None = None,
    ) -> list[dict]:
        """See StorageProvider.search_cluster_candidates. Scores both raw
        unclustered topics and existing clusters against topic_id's own
        stored embedding/tags, so a first-ever pair can form a brand new
        cluster and not just grow an existing one.
        """
        scorer = RetrievalScorer(weights, now)

        own_row = self._conn.execute(
            "SELECT embedding, tags FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
        if own_row is None or own_row[0] is None:
            return []
        query_embedding = _deserialize_embedding(own_row[0])
        query_tags: list[str] = json.loads(own_row[1])

        scored: list[tuple[float, dict, dict]] = []

        # Topic-side candidates: still-unclustered topics for this workspace,
        # excluding topic_id itself. Accelerated path (see search_topics)
        # uses vec_topics' unclustered_only filter — vec0 has no "exclude
        # this id" option, so topic_id is dropped from the KNN hits after
        # the fact instead.
        topic_semantic_lookup: dict[int, float] | None = None
        if self.vec_index.available:
            knn_hits = self.vec_index.knn(
                "vec_topics", workspace_id, query_embedding, k=_knn_prefilter_k(limit) + 1, unclustered_only=True
            )
            topic_semantic_lookup = {tid: sim for tid, sim in knn_hits if tid != topic_id}
            if not topic_semantic_lookup:
                topic_rows = []
            else:
                placeholders = ",".join("?" * len(topic_semantic_lookup))
                topic_rows = self._conn.execute(
                    f"SELECT id, label, summary, tags, embedding, created_at, closed_at "
                    f"FROM topics WHERE id IN ({placeholders}) AND status='closed' AND cluster_id IS NULL",
                    tuple(topic_semantic_lookup.keys()),
                ).fetchall()
        else:
            topic_rows = self._conn.execute(
                "SELECT id, label, summary, tags, embedding, created_at, closed_at "
                "FROM topics WHERE status='closed' AND cluster_id IS NULL AND workspace_id = ? AND id != ?",
                (workspace_id, topic_id),
            ).fetchall()
        for row in topic_rows:
            candidate = {
                "candidate_kind": "topic",
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": json.loads(row[3]),
                "topic_count": 1,
            }
            score, breakdown = scorer.score(
                candidate_type="cluster_candidate",
                candidate_id=candidate["id"],
                semantic_raw=self._semantic_similarity(topic_semantic_lookup, row[0], query_embedding, row[4]),
                recency_timestamp=row[6],
                include_tag_overlap=True,
                query_tags=query_tags,
                candidate_tags=candidate["tags"],
                extra_breakdown={"candidate_kind": "topic"},
            )
            scored.append((score, candidate, breakdown))

        # Cluster-side candidates: existing clusters for this workspace.
        cluster_semantic_lookup: dict[int, float] | None = None
        if self.vec_index.available:
            knn_hits = self.vec_index.knn("vec_clusters", workspace_id, query_embedding, k=_knn_prefilter_k(limit))
            cluster_semantic_lookup = dict(knn_hits)
            if not cluster_semantic_lookup:
                cluster_rows = []
            else:
                placeholders = ",".join("?" * len(cluster_semantic_lookup))
                cluster_rows = self._conn.execute(
                    f"SELECT id, label, summary, tags, embedding, updated_at "
                    f"FROM topic_clusters WHERE id IN ({placeholders})",
                    tuple(cluster_semantic_lookup.keys()),
                ).fetchall()
        else:
            cluster_rows = self._conn.execute(
                "SELECT id, label, summary, tags, embedding, updated_at "
                "FROM topic_clusters WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchall()
        for row in cluster_rows:
            cluster_full = self.get_cluster(row[0])
            candidate = {
                "candidate_kind": "cluster",
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": json.loads(row[3]),
                "milestones": cluster_full["milestones"] if cluster_full else [],
                "topic_count": cluster_full["topic_count"] if cluster_full else 0,
            }
            score, breakdown = scorer.score(
                candidate_type="cluster_candidate",
                candidate_id=candidate["id"],
                semantic_raw=self._semantic_similarity(cluster_semantic_lookup, row[0], query_embedding, row[4]),
                recency_timestamp=row[5],
                include_tag_overlap=True,
                query_tags=query_tags,
                candidate_tags=candidate["tags"],
                extra_breakdown={"candidate_kind": "cluster"},
            )
            scored.append((score, candidate, breakdown))

        return scorer.finalize(
            scored=scored, candidate_type="cluster_candidate", limit=limit,
            tracer=tracer, request_id=request_id, extra_payload={"topic_id": topic_id},
        )

    # ── Memory Items ───────────────────────────────────────

    def store_memory_item(self, item: MemoryItem) -> int:
        embed_blob = _serialize_embedding(item.embedding) if item.embedding else None
        cur = self._conn.execute(
            "INSERT INTO memory_items (content, item_type, source_peer, subject_peer, "
            "attribution_type, topic_id, message_id, embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (item.content, item.item_type, item.source_peer, item.subject_peer,
             item.attribution_type, item.topic_id, item.message_id, embed_blob),
        )
        self._conn.commit()
        item_id: int = cur.lastrowid  # type: ignore[assignment]
        if item.embedding and item.message_id:
            # memory_items carries no workspace_id of its own — resolved via
            # the same messages join search_memory_items uses.
            msg_row = self._conn.execute(
                "SELECT workspace_id FROM messages WHERE id = ?", (item.message_id,)
            ).fetchone()
            if msg_row and msg_row[0]:
                self.vec_index.upsert("vec_memory_items", item_id, msg_row[0], item.embedding)
        return item_id

    def get_memory_item(self, item_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, content, item_type, source_peer, subject_peer, attribution_type, "
            "topic_id, created_at FROM memory_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "content": row[1],
            "item_type": row[2],
            "source_peer": row[3],
            "subject_peer": row[4],
            "attribution_type": row[5],
            "topic_id": row[6],
            "created_at": row[7],
        }

    def search_memory_items(
        self,
        query_embedding: list[float] | None = None,
        workspace_id: str | None = None,
        budget: MemoryItemBudget | None = None,
        weights: dict[str, float] | None = None,
        tracer: Tracer | None = None,
        request_id: str | None = None,
        now: datetime | None = None,
    ) -> list[MemoryItem]:
        """Search memory items by embedding similarity, workspace, and recency.

        `workspace_id`, when given, is a hard SQL filter, joined through
        `messages` (memory_items carry no workspace_id column of their own —
        only a `message_id`, and `messages.workspace_id` is the real identifier).
        `workspace_id=None` searches every workspace.

        This replaces a previous soft "person" score that compared
        `workspace_id` against `source_peer`/`subject_peer` — those are always
        `FragmentaryExtractor`'s fixed `"user"/"ai"/"other"` vocabulary, a
        different value space from the real workspace_id (same confusion
        `FactConsolidator` had — see `_PROFILE_SUBJECT` there). That
        comparison only ever matched when `workspace_id` happened to literally
        equal `"user"`; for any real multi-tenant workspace_id it silently never
        fired, so memory-item retrieval had no working workspace-scoping signal
        at all beyond semantic similarity. There is no `person` scoring
        dimension anymore — the join makes it a hard prerequisite, not a
        ranking signal.

        `budget` (defaults to `MemoryItemBudget()`) selects via a
        per-item-type reserved budget instead of a flat result count — see
        `RetrievalScorer.finalize_tiered`. The vector-index KNN prefilter
        below is embedding-similarity-only, with no item_type awareness;
        `surprise`/`feeling` are rare enough that it could exclude every
        candidate of one of those types before type-aware selection ever
        runs, so when the budget reserves either, a second, un-prefiltered
        fetch scoped to just those two types supplements the KNN pool.

        `weights` overrides the semantic/temporal scoring weights when
        provided (see `RetrievalScorer`). `tag_overlap` doesn't apply
        (memory items carry no tags). The `temporal` term is a recency
        *re-ranking* signal, added only to candidates that already have
        semantic relevance (see `search_topics`); an item's recency uses
        its `created_at`. `now` is the reference time for the decay
        (defaults to the current UTC time).

        When `tracer` is given, one `retrieval_scoring_summary` event is
        recorded for the whole call — see `RetrievalScorer.finalize_tiered`.
        """
        budget = budget or MemoryItemBudget()
        scorer = RetrievalScorer(weights, now)
        # Accelerated path (see search_topics for the full explanation):
        # pre-filter to the nearest-by-embedding candidates via the vector
        # index — workspace-scoping happened when each item was upserted there
        # (via the same messages join, at write time), so a plain id-fetch
        # is enough here, no join needed.
        semantic_lookup: dict[int, float] | None = None
        if self.vec_index.available and query_embedding and workspace_id:
            knn_hits = self.vec_index.knn("vec_memory_items", workspace_id, query_embedding, k=_knn_prefilter_k(budget.total))
            semantic_lookup = dict(knn_hits)
            if not semantic_lookup:
                rows = []
            else:
                placeholders = ",".join("?" * len(semantic_lookup))
                rows = list(self._conn.execute(
                    f"SELECT id, content, item_type, source_peer, subject_peer, "
                    f"attribution_type, topic_id, message_id, embedding, created_at "
                    f"FROM memory_items WHERE id IN ({placeholders})",
                    tuple(semantic_lookup.keys()),
                ).fetchall())
            if budget.surprise > 0 or budget.feeling > 0:
                knn_ids = set(semantic_lookup.keys())
                bypass_rows = self._conn.execute(
                    "SELECT mi.id, mi.content, mi.item_type, mi.source_peer, mi.subject_peer, "
                    "mi.attribution_type, mi.topic_id, mi.message_id, mi.embedding, mi.created_at "
                    "FROM memory_items mi JOIN messages m ON mi.message_id = m.id "
                    "WHERE m.workspace_id = ? AND mi.item_type IN ('surprise', 'feeling')",
                    (workspace_id,),
                ).fetchall()
                rows.extend(r for r in bypass_rows if r[0] not in knn_ids)
        elif workspace_id:
            rows = self._conn.execute(
                "SELECT mi.id, mi.content, mi.item_type, mi.source_peer, mi.subject_peer, "
                "mi.attribution_type, mi.topic_id, mi.message_id, mi.embedding, mi.created_at "
                "FROM memory_items mi JOIN messages m ON mi.message_id = m.id "
                "WHERE m.workspace_id = ?",
                (workspace_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, content, item_type, source_peer, subject_peer, "
                "attribution_type, topic_id, message_id, embedding, created_at "
                "FROM memory_items"
            ).fetchall()

        scored: list[tuple[float, MemoryItem, dict]] = []
        for row in rows:
            item = MemoryItem(
                id=row[0],
                content=row[1],
                item_type=row[2],
                source_peer=row[3],
                subject_peer=row[4],
                attribution_type=row[5],
                topic_id=row[6],
                message_id=row[7],
                timestamp=row[9],
            )
            # No tag_overlap (items carry no tags) and no person term (workspace_id
            # is a hard SQL filter above). row[9] is the raw created_at string
            # (item.timestamp is typed datetime but carries this string at runtime).
            # A rare-type bypass row (id not in semantic_lookup) must not
            # reuse the shared lookup dict — .get(id, 0.0) would silently
            # score it 0.0 instead of computing its real cosine similarity
            # (see _semantic_similarity's docstring), defeating the bypass.
            row_lookup = semantic_lookup if (semantic_lookup is not None and row[0] in semantic_lookup) else None
            score, breakdown = scorer.score(
                candidate_type="memory_item",
                candidate_id=row[0],
                semantic_raw=self._semantic_similarity(row_lookup, row[0], query_embedding, row[8]),
                recency_timestamp=row[9],
            )
            scored.append((score, item, breakdown))

        return scorer.finalize_tiered(scored=scored, budget=budget, tracer=tracer, request_id=request_id)

    def get_memory_items_by_turn(self, turn_id: int) -> list[MemoryItem]:
        """Return all memory items extracted from a given turn, oldest first.

        Joins through `messages.turn_id` since memory items only carry a
        `message_id` reference, not a turn id of their own.
        """
        rows = self._conn.execute(
            "SELECT mi.id, mi.content, mi.item_type, mi.source_peer, mi.subject_peer, "
            "mi.attribution_type, mi.topic_id, mi.message_id, mi.embedding, mi.created_at "
            "FROM memory_items mi JOIN messages m ON mi.message_id = m.id "
            "WHERE m.turn_id = ? ORDER BY mi.id ASC",
            (turn_id,),
        ).fetchall()
        return [
            MemoryItem(
                id=row[0], content=row[1], item_type=row[2], source_peer=row[3], subject_peer=row[4],
                attribution_type=row[5], topic_id=row[6], message_id=row[7],
                embedding=_deserialize_embedding(row[8]) if row[8] else None,
                timestamp=row[9],
            )
            for row in rows
        ]

    def get_memory_items_by_topic(self, topic_id: int) -> list[MemoryItem]:
        """Return every memory item extracted for `topic_id`, oldest first.

        No join needed — memory_items carries its own topic_id column.
        """
        rows = self._conn.execute(
            "SELECT id, content, item_type, source_peer, subject_peer, "
            "attribution_type, topic_id, message_id, created_at "
            "FROM memory_items WHERE topic_id = ? ORDER BY id ASC",
            (topic_id,),
        ).fetchall()
        return [
            MemoryItem(
                id=r[0], content=r[1], item_type=r[2], source_peer=r[3], subject_peer=r[4],
                attribution_type=r[5], topic_id=r[6], message_id=r[7], timestamp=r[8],
            )
            for r in rows
        ]

    def list_memory_items(
        self, item_type: str | None = None, topic_id: int | None = None, q: str | None = None,
    ) -> list[MemoryItem]:
        query = (
            "SELECT id, content, item_type, source_peer, subject_peer, "
            "attribution_type, topic_id, message_id, created_at FROM memory_items"
        )
        conditions = []
        params: list = []
        if item_type is not None:
            conditions.append("item_type = ?")
            params.append(item_type)
        if topic_id is not None:
            conditions.append("topic_id = ?")
            params.append(topic_id)
        if q:
            # LIKE is case-insensitive for ASCII in SQLite by default, which
            # is fine here — this is a dashboard search box, not a linguistic
            # search feature. %/_/\ are escaped so a search string containing
            # them is matched literally rather than as wildcards.
            escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append("content LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped}%")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [
            MemoryItem(
                id=r[0], content=r[1], item_type=r[2], source_peer=r[3], subject_peer=r[4],
                attribution_type=r[5], topic_id=r[6], message_id=r[7], timestamp=r[8],
            )
            for r in rows
        ]

    # ── Messages ───────────────────────────────────────────

    def store_message(
        self,
        turn_id: int,
        role: str,
        content: str,
        workspace_id: str,
        topic_id: int | None = None,
        request_id: str | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO messages (turn_id, role, content, workspace_id, topic_id, request_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (turn_id, role, content, workspace_id, topic_id, request_id),
        )
        if topic_id is not None:
            self._conn.execute(
                "UPDATE topics SET message_count = message_count + 1 WHERE id = ?",
                (topic_id,),
            )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_messages_by_topic(self, topic_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, turn_id, role, content, workspace_id, created_at "
            "FROM messages WHERE topic_id = ? ORDER BY turn_id, id",
            (topic_id,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "turn_id": r[1],
                "role": r[2],
                "content": r[3],
                "workspace_id": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    def get_recent_messages_by_topic(self, topic_id: int, limit: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, turn_id, role, content, workspace_id, created_at "
            "FROM messages WHERE topic_id = ? ORDER BY id DESC LIMIT ?",
            (topic_id, limit),
        ).fetchall()
        return [
            {
                "id": r[0],
                "turn_id": r[1],
                "role": r[2],
                "content": r[3],
                "workspace_id": r[4],
                "created_at": r[5],
            }
            for r in reversed(rows)
        ]

    def get_messages_by_request_id(self, request_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, turn_id, role, content, workspace_id, topic_id, created_at "
            "FROM messages WHERE request_id = ? ORDER BY id",
            (request_id,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "turn_id": r[1],
                "role": r[2],
                "content": r[3],
                "workspace_id": r[4],
                "topic_id": r[5],
                "created_at": r[6],
            }
            for r in rows
        ]

    def get_recent_turns(self, limit: int, workspace_id: str | None = None) -> list[dict]:
        if workspace_id is not None:
            rows = self._conn.execute(
                "SELECT request_id, workspace_id, topic_id, content, created_at FROM messages "
                "WHERE role = 'user' AND request_id IS NOT NULL AND workspace_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (workspace_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT request_id, workspace_id, topic_id, content, created_at FROM messages "
                "WHERE role = 'user' AND request_id IS NOT NULL "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"request_id": r[0], "workspace_id": r[1], "topic_id": r[2], "content": r[3], "created_at": r[4]}
            for r in rows
        ]

    def count_recent_turns(self, workspace_id: str | None = None) -> int:
        if workspace_id is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE role = 'user' AND request_id IS NOT NULL AND workspace_id = ?",
                (workspace_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE role = 'user' AND request_id IS NOT NULL",
            ).fetchone()
        return row[0]
