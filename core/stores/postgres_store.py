"""Postgres + pgvector storage backend for Eidolon.

Implements the ``StorageProvider`` protocol using Postgres, chosen as a
second backend alongside ``SqliteStore`` specifically for reliability:
SQLite's concurrency model has every client process take OS-level
advisory file locks directly on the ``.db`` file, which Docker Desktop's
virtiofs bind mount doesn't reliably forward (the same class of problem
as NFS, which SQLite's own docs warn about) — see the WAL-mode-corruption
and stuck-write-lock comments in ``core/stores/sqlite_store.py``. Postgres
sidesteps this entirely: exactly one process (the Postgres server) ever
touches the raw data files; every client talks to it over a wire
protocol, no cross-process file locking involved.

Compared to SqliteStore, this backend is simpler in one specific way:
embeddings live directly in each row's own ``VECTOR`` column (native
pgvector), so there's no separate shadow-table vector index, no
extension-loading fallback branch, and no brute-force Python cosine scan
— see ``core/stores/vec_index.py`` for the SQLite-side machinery this
backend has no equivalent of and needs none.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from core.interfaces import Tracer
from core.models import Fact, MemoryItem, MemoryItemBudget
from core.stores.postgres_migrations import migrate
from core.stores.retrieval_scorer import RetrievalScorer

logger = logging.getLogger(__name__)

# Same prefilter sizing as SqliteStore's accelerated (vec_index-available)
# path — see core/stores/sqlite_store.py's _knn_prefilter_k. PostgresStore
# has only one search path (pgvector is always available once connected),
# so this always applies, unlike SQLite's conditional use of it.
_KNN_PREFILTER_MIN = 50
_KNN_PREFILTER_MULTIPLIER = 10


def _knn_prefilter_k(limit: int) -> int:
    return max(limit * _KNN_PREFILTER_MULTIPLIER, _KNN_PREFILTER_MIN)


def _to_float_list(vec: Any) -> list[float] | None:
    """Normalize a pgvector column value — pgvector-python's psycopg
    adapter returns its own ``pgvector.Vector`` wrapper, not a plain
    list — into a plain list[float], matching SqliteStore's
    _deserialize_embedding return type."""
    if vec is None:
        return None
    if hasattr(vec, "to_list"):
        return [float(x) for x in vec.to_list()]
    return [float(x) for x in vec]


class PostgresStore:
    """Postgres + pgvector implementation of ``StorageProvider``.

    Parameters mirror ``StorageConfig``'s discrete connection fields
    (host/port/dbname/user/password) rather than a DSN string — see
    ``core/config.py``.

    Parameters
    ----------
    host, port, dbname, user, password
        Postgres connection details.
    embedding_dim : int
        Width of the embedder's vectors. pgvector's ``VECTOR(n)`` column
        width is fixed at schema-creation time, so — unlike SqliteStore,
        which never needs to know this — it must be supplied up front.
    """

    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        embedding_dim: int,
    ) -> None:
        self._host = host
        self._port = port
        self._dbname = dbname
        self._user = user
        self._password = password
        self._embedding_dim = embedding_dim
        self._db: psycopg.Connection | None = None

    # ── Lifecycle ──────────────────────────────────────────

    def connect(self) -> None:
        """Open the connection, run pending migrations, then register
        pgvector's type adapter.

        Migrations must run first: register_vector() looks up the
        `vector` type's OID in the database, which doesn't exist until
        migration 1's `CREATE EXTENSION IF NOT EXISTS vector` has run.

        Connects with ``autocommit=False`` only for this setup sequence
        — ``migrate()`` needs each migration's schema change and its
        version-bookkeeping insert to commit together atomically (see
        ``core/stores/postgres_migrations.py``). Every real operation
        after this method returns runs with autocommit on instead (see
        below): a long-lived connection like this one previously left
        every read-only method (no trailing ``.commit()``, unlike the
        write methods) sitting "idle in transaction" indefinitely once
        called — confirmed live, blocking a later schema migration that
        needed an ACCESS EXCLUSIVE lock on `messages`, and more broadly
        able to block autovacuum on frequently-written tables the
        longer it went unnoticed.
        """
        self._db = psycopg.connect(
            host=self._host,
            port=self._port,
            dbname=self._dbname,
            user=self._user,
            password=self._password,
            autocommit=False,
        )
        migrate(self._db, self._embedding_dim)
        register_vector(self._db)
        self._db.commit()  # close out register_vector()'s own implicit transaction
        self._db.autocommit = True

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def _conn(self) -> psycopg.Connection:
        if self._db is None:
            raise RuntimeError("PostgresStore not connected — call connect() first")
        return self._db

    # ── Facts ───────────────────────────────────────────────

    def get_facts(self, workspace_id: str) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT workspace_id, category, key, value, source, importance, created_at "
            "FROM facts WHERE workspace_id = %s "
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
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (workspace_id, category, key) DO UPDATE SET "
            "value=excluded.value, source=excluded.source, importance=excluded.importance",
            (workspace_id, category, key, value, source, importance),
        )

    def delete_fact(self, workspace_id: str, category: str, key: str) -> None:
        self._conn.execute(
            "DELETE FROM facts WHERE workspace_id = %s AND category = %s AND key = %s",
            (workspace_id, category, key),
        )

    # ── Facts Consolidation ────────────────────────────────

    def count_user_turns(self, workspace_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE workspace_id = %s AND role = 'user'",
            (workspace_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_fact_items_since(self, subject_peer: str, since_item_id: int) -> list[MemoryItem]:
        rows = self._conn.execute(
            "SELECT id, content, item_type, source_peer, subject_peer, "
            "attribution_type, topic_id, message_id, created_at "
            "FROM memory_items "
            "WHERE subject_peer = %s AND id > %s "
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
            "SELECT last_item_id FROM fact_consolidation_state WHERE workspace_id = %s",
            (workspace_id,),
        ).fetchone()
        return row[0] if row else 0

    def set_consolidation_checkpoint(self, workspace_id: str, last_item_id: int) -> None:
        self._conn.execute(
            "INSERT INTO fact_consolidation_state (workspace_id, last_item_id, last_run_at) "
            "VALUES (%s, %s, now()) "
            "ON CONFLICT (workspace_id) DO UPDATE SET last_item_id=excluded.last_item_id, last_run_at=excluded.last_run_at",
            (workspace_id, last_item_id),
        )

    # ── Entity Profiles ────────────────────────────────────

    def get_topics_since(self, workspace_id: str, since_topic_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, summary, named_entities FROM topics "
            "WHERE workspace_id = %s AND status = 'closed' AND id > %s "
            "ORDER BY id ASC",
            (workspace_id, since_topic_id),
        ).fetchall()
        return [
            {"id": r[0], "label": r[1], "summary": r[2], "named_entities": r[3]}
            for r in rows
        ]

    def increment_entity_mentions(self, workspace_id: str, entity_names: list[str]) -> None:
        if not entity_names:
            return
        with self._conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO entity_profiles (workspace_id, entity_name, mention_count) "
                "VALUES (%s, %s, 1) "
                "ON CONFLICT (workspace_id, entity_name) DO UPDATE SET mention_count = entity_profiles.mention_count + 1",
                [(workspace_id, name) for name in entity_names],
            )

    def materialize_entities(self, workspace_id: str, threshold: int) -> list[str]:
        rows = self._conn.execute(
            "SELECT entity_name FROM entity_profiles "
            "WHERE workspace_id = %s AND materialized_at IS NULL AND mention_count >= %s",
            (workspace_id, threshold),
        ).fetchall()
        names = [r[0] for r in rows]
        if names:
            self._conn.execute(
                "UPDATE entity_profiles SET materialized_at = now() "
                "WHERE workspace_id = %s AND entity_name = ANY(%s)",
                (workspace_id, names),
            )
        return names

    def get_materialized_entity_names(self, workspace_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT entity_name FROM entity_profiles WHERE workspace_id = %s AND materialized_at IS NOT NULL",
            (workspace_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_topics_mentioning_entity(
        self, workspace_id: str, entity_name: str, since_topic_id: int
    ) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, summary, named_entities FROM topics "
            "WHERE workspace_id = %s AND status = 'closed' AND id > %s "
            "ORDER BY id ASC",
            (workspace_id, since_topic_id),
        ).fetchall()
        result = []
        for r in rows:
            named_entities = r[3]
            if entity_name in named_entities:
                result.append({"id": r[0], "label": r[1], "summary": r[2], "named_entities": named_entities})
        return result

    def get_entity_facts(self, workspace_id: str, entity_names: list[str]) -> list[Fact]:
        if not entity_names:
            return []
        rows = self._conn.execute(
            "SELECT workspace_id, category, key, value, source, importance, created_at, entity_name "
            "FROM entity_facts WHERE workspace_id = %s AND entity_name = ANY(%s) "
            "ORDER BY entity_name, CASE importance WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, category, key",
            (workspace_id, entity_names),
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
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (workspace_id, entity_name, category, key) DO UPDATE SET "
            "value=excluded.value, source=excluded.source, importance=excluded.importance",
            (workspace_id, entity_name, category, key, value, source, importance),
        )

    def delete_entity_fact(self, workspace_id: str, entity_name: str, category: str, key: str) -> None:
        self._conn.execute(
            "DELETE FROM entity_facts WHERE workspace_id = %s AND entity_name = %s AND category = %s AND key = %s",
            (workspace_id, entity_name, category, key),
        )

    def list_entity_profiles(self, workspace_id: str | None = None) -> list[dict]:
        query = (
            "SELECT ep.workspace_id, ep.entity_name, ep.mention_count, ep.materialized_at, "
            "ep.last_consolidated_topic_id, COUNT(ef.id) AS fact_count "
            "FROM entity_profiles ep LEFT JOIN entity_facts ef "
            "ON ef.workspace_id = ep.workspace_id AND ef.entity_name = ep.entity_name"
        )
        params: list[str] = []
        if workspace_id is not None:
            query += " WHERE ep.workspace_id = %s"
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
            "SELECT last_topic_id FROM entity_scan_state WHERE workspace_id = %s",
            (workspace_id,),
        ).fetchone()
        return row[0] if row else 0

    def set_entity_scan_checkpoint(self, workspace_id: str, last_topic_id: int) -> None:
        self._conn.execute(
            "INSERT INTO entity_scan_state (workspace_id, last_topic_id, last_run_at) "
            "VALUES (%s, %s, now()) "
            "ON CONFLICT (workspace_id) DO UPDATE SET "
            "last_topic_id=excluded.last_topic_id, last_run_at=excluded.last_run_at",
            (workspace_id, last_topic_id),
        )

    def get_entity_consolidation_checkpoint(self, workspace_id: str, entity_name: str) -> int:
        row = self._conn.execute(
            "SELECT last_consolidated_topic_id FROM entity_profiles WHERE workspace_id = %s AND entity_name = %s",
            (workspace_id, entity_name),
        ).fetchone()
        return row[0] if row else 0

    def set_entity_consolidation_checkpoint(
        self, workspace_id: str, entity_name: str, last_topic_id: int
    ) -> None:
        self._conn.execute(
            "UPDATE entity_profiles SET last_consolidated_topic_id = %s "
            "WHERE workspace_id = %s AND entity_name = %s",
            (last_topic_id, workspace_id, entity_name),
        )

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
        row = self._conn.execute(
            "INSERT INTO topics (label, summary, tags, embedding, workspace_id, session_id) "
            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (label, summary, Jsonb(tags or []), embedding, workspace_id, session_id),
        ).fetchone()
        assert row is not None  # RETURNING id always yields a row on successful INSERT
        return row[0]

    def get_topic(self, topic_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, label, summary, tags, status, workspace_id, created_at, closed_at, message_count, "
            "named_entities FROM topics WHERE id = %s",
            (topic_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "label": row[1],
            "summary": row[2],
            "tags": row[3],  # jsonb round-trips to a Python list already
            "status": row[4],
            "workspace_id": row[5],
            "created_at": row[6],
            "closed_at": row[7],
            "message_count": row[8],
            "named_entities": row[9],
        }

    def close_topic(self, topic_id: int, summary: str) -> None:
        self._conn.execute(
            "UPDATE topics SET status='closed', summary=%s, closed_at=now() WHERE id=%s",
            (summary, topic_id),
        )

    def get_active_topic(self, workspace_id: str, session_id: str | None = None) -> int | None:
        """See StorageProvider.get_active_topic / SqliteStore's docstring
        for the full (workspace_id, session_id) scoping rationale. Uses
        Postgres's ``IS NOT DISTINCT FROM`` — the exact NULL-safe
        equivalent of SQLite's ``session_id IS ?`` binding."""
        row = self._conn.execute(
            "SELECT id FROM topics WHERE workspace_id = %s AND session_id IS NOT DISTINCT FROM %s AND status = 'active' "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (workspace_id, session_id),
        ).fetchone()
        return row[0] if row else None

    def get_stale_active_topics(self, cutoff: datetime, workspace_id: str | None = None) -> list[dict]:
        """See SqliteStore.get_stale_active_topics. Real TIMESTAMPTZ
        columns mean the staleness comparison can happen directly in SQL
        (via HAVING) rather than SQLite's Python-side string-timestamp
        parsing workaround."""
        query = (
            "SELECT t.id, t.label, t.summary, t.tags, t.status, t.workspace_id, t.created_at, t.closed_at, "
            "COALESCE(MAX(m.created_at), t.created_at) AS last_activity "
            "FROM topics t LEFT JOIN messages m ON m.topic_id = t.id "
            "WHERE t.status = 'active'"
        )
        params: list = []
        if workspace_id is not None:
            query += " AND t.workspace_id = %s"
            params.append(workspace_id)
        query += " GROUP BY t.id HAVING COALESCE(MAX(m.created_at), t.created_at) < %s ORDER BY last_activity ASC"
        params.append(cutoff)
        rows = self._conn.execute(query, params).fetchall()

        return [
            {
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": row[3],
                "status": row[4],
                "workspace_id": row[5],
                "created_at": row[6],
                "closed_at": row[7],
                "last_activity": row[8],
            }
            for row in rows
        ]

    def get_recent_topics(self, workspace_id: str, limit: int, min_message_count: int = -1) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, embedding, status, workspace_id, created_at, closed_at "
            "FROM topics WHERE status='closed' AND cluster_id IS NULL AND workspace_id = %s "
            "AND message_count > %s "
            "ORDER BY closed_at DESC, id DESC LIMIT %s",
            (workspace_id, min_message_count, limit),
        ).fetchall()
        return [
            {
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": row[3],
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
            "FROM topics WHERE status = 'closed' AND closed_at >= %s ORDER BY closed_at ASC",
            (since,),
        ).fetchall()
        return [
            {
                "id": row[0],
                "workspace_id": row[1],
                "label": row[2],
                "created_at": row[3],
                "closed_at": row[4],
                "message_count": row[5],
            }
            for row in rows
        ]

    def list_topics(self, status: str | None = None, workspace_id: str | None = None) -> list[dict]:
        query = (
            "SELECT id, label, status, workspace_id, cluster_id, created_at, closed_at, message_count "
            "FROM topics"
        )
        conditions = []
        params: list = []
        if status is not None:
            conditions.append("status = %s")
            params.append(status)
        if workspace_id is not None:
            conditions.append("workspace_id = %s")
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

    def update_topic_embedding(self, topic_id: int, embedding: list[float]) -> None:
        self._conn.execute("UPDATE topics SET embedding = %s WHERE id = %s", (embedding, topic_id))

    def update_topic_tags(self, topic_id: int, tags: list[str]) -> None:
        self._conn.execute("UPDATE topics SET tags = %s WHERE id = %s", (Jsonb(tags), topic_id))

    def update_topic_entities(self, topic_id: int, named_entities: list[str]) -> None:
        self._conn.execute(
            "UPDATE topics SET named_entities = %s WHERE id = %s", (Jsonb(named_entities), topic_id)
        )

    def update_topic_label(self, topic_id: int, label: str) -> None:
        self._conn.execute("UPDATE topics SET label = %s WHERE id = %s", (label, topic_id))

    def count_messages_by_topic(self, topic_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE topic_id = %s", (topic_id,)
        ).fetchone()
        return row[0] if row else 0

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
        """See StorageProvider.search_topics / SqliteStore's docstring for
        the full scoring semantics (workspace_id as a hard filter, temporal as
        a re-ranking-only signal, etc.) — unchanged here. `min_message_count`
        soft-hides "mini-topics", applied in every branch's WHERE clause
        before ranking, same as SqliteStore.

        When `query_embedding` is given, pgvector's `<=>` cosine-distance
        operator does the nearest-neighbor prefiltering directly in one
        query (native ANN index, always available — no vec_index/brute-
        force duality). This only ever returns topics that have an
        embedding, matching SqliteStore's *accelerated* (vec_index-
        available) path exactly; SQLite's brute-force *fallback* path
        additionally scores embedding-less topics at similarity=0.0 via
        tag/person alone, but that fallback only exists because
        sqlite-vec is optional there — Postgres has no such gap to fall
        back from, so this is accepted, not a regression.
        """
        scorer = RetrievalScorer(weights, now)

        if query_embedding is not None:
            k = _knn_prefilter_k(limit)
            if workspace_id:
                rows = self._conn.execute(
                    "SELECT id, label, summary, tags, status, workspace_id, created_at, closed_at, "
                    "1 - (embedding <=> %(q)s::vector) AS similarity "
                    "FROM topics WHERE status='closed' AND cluster_id IS NULL AND workspace_id = %(workspace_id)s "
                    "AND embedding IS NOT NULL AND message_count > %(min_msgs)s "
                    "ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s",
                    {"q": query_embedding, "workspace_id": workspace_id, "k": k, "min_msgs": min_message_count},
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, label, summary, tags, status, workspace_id, created_at, closed_at, "
                    "1 - (embedding <=> %(q)s::vector) AS similarity "
                    "FROM topics WHERE status='closed' AND cluster_id IS NULL AND embedding IS NOT NULL "
                    "AND message_count > %(min_msgs)s "
                    "ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s",
                    {"q": query_embedding, "k": k, "min_msgs": min_message_count},
                ).fetchall()
        elif workspace_id:
            rows = [
                (*r, 0.0)
                for r in self._conn.execute(
                    "SELECT id, label, summary, tags, status, workspace_id, created_at, closed_at "
                    "FROM topics WHERE status='closed' AND cluster_id IS NULL AND workspace_id = %s "
                    "AND message_count > %s",
                    (workspace_id, min_message_count),
                ).fetchall()
            ]
        else:
            rows = [
                (*r, 0.0)
                for r in self._conn.execute(
                    "SELECT id, label, summary, tags, status, workspace_id, created_at, closed_at "
                    "FROM topics WHERE status='closed' AND cluster_id IS NULL AND message_count > %s",
                    (min_message_count,),
                ).fetchall()
            ]

        scored: list[tuple[float, dict, dict]] = []
        for row in rows:
            topic = {
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": row[3],
                "status": row[4],
                "workspace_id": row[5],
                "created_at": row[6],
                "closed_at": row[7],
            }
            score, breakdown = scorer.score(
                candidate_type="topic",
                candidate_id=topic["id"],
                semantic_raw=row[8],
                recency_timestamp=topic["closed_at"] or topic["created_at"],
                include_tag_overlap=True,
                query_tags=tags,
                candidate_tags=topic["tags"],
                include_person=True,
                # Not workspace_id-derived: this row's workspace_id already equals the
                # query's (the WHERE clause above guarantees it), so
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
            "SELECT id, workspace_id, label, summary, tags, topic_count, created_at, updated_at "
            "FROM topic_clusters WHERE workspace_id = %s ORDER BY id",
            (workspace_id,),
        ).fetchall()
        return [
            {
                "id": row[0],
                "workspace_id": row[1],
                "label": row[2],
                "summary": row[3],
                "tags": row[4],
                "topic_count": row[5],
                "created_at": row[6],
                "updated_at": row[7],
            }
            for row in rows
        ]

    def list_clusters(self, workspace_id: str | None = None) -> list[dict]:
        query = (
            "SELECT id, workspace_id, label, summary, tags, topic_count, created_at, updated_at "
            "FROM topic_clusters"
        )
        params: list = []
        if workspace_id is not None:
            query += " WHERE workspace_id = %s"
            params.append(workspace_id)
        query += " ORDER BY created_at DESC, id DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "id": row[0],
                "workspace_id": row[1],
                "label": row[2],
                "summary": row[3],
                "tags": row[4],
                "topic_count": row[5],
                "created_at": row[6],
                "updated_at": row[7],
            }
            for row in rows
        ]

    def create_cluster(
        self,
        workspace_id: str,
        label: str,
        summary: str,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> int:
        row = self._conn.execute(
            "INSERT INTO topic_clusters (workspace_id, label, summary, tags, embedding) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (workspace_id, label, summary, Jsonb(tags or []), embedding),
        ).fetchone()
        assert row is not None  # RETURNING id always yields a row on successful INSERT
        return row[0]

    def get_cluster(self, cluster_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, workspace_id, label, summary, tags, topic_count, created_at, updated_at "
            "FROM topic_clusters WHERE id = %s",
            (cluster_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "workspace_id": row[1],
            "label": row[2],
            "summary": row[3],
            "tags": row[4],
            "topic_count": row[5],
            "created_at": row[6],
            "updated_at": row[7],
        }

    def update_cluster(
        self,
        cluster_id: int,
        summary: str,
        label: str | None = None,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        sets = ["summary = %s", "updated_at = now()"]
        params: list = [summary]
        if label is not None:
            sets.append("label = %s")
            params.append(label)
        if tags is not None:
            sets.append("tags = %s")
            params.append(Jsonb(tags))
        if embedding is not None:
            sets.append("embedding = %s")
            params.append(embedding)
        params.append(cluster_id)
        self._conn.execute(f"UPDATE topic_clusters SET {', '.join(sets)} WHERE id = %s", params)

    def assign_topic_to_cluster(self, topic_id: int, cluster_id: int) -> None:
        """Real transaction here (both statements committed together) —
        SqliteStore's equivalent commits each statement separately, a
        known non-atomic gap. Free to fix on this backend since Postgres
        makes it trivial; not backported to SQLite in this change.
        Explicit ``with conn.transaction()`` since the connection itself
        runs autocommit (see ``connect()``) — psycopg3's documented
        pattern for a real multi-statement transaction in that mode."""
        with self._conn.transaction():
            self._conn.execute("UPDATE topics SET cluster_id = %s WHERE id = %s", (cluster_id, topic_id))
            self._conn.execute(
                "UPDATE topic_clusters SET topic_count = topic_count + 1 WHERE id = %s", (cluster_id,)
            )

    def get_topics_by_cluster(self, cluster_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, status, workspace_id, created_at, closed_at "
            "FROM topics WHERE cluster_id = %s ORDER BY closed_at ASC, id ASC",
            (cluster_id,),
        ).fetchall()
        return [
            {
                "id": row[0], "label": row[1], "summary": row[2], "tags": row[3],
                "status": row[4], "workspace_id": row[5], "created_at": row[6], "closed_at": row[7],
            }
            for row in rows
        ]

    def get_recent_topics_in_cluster(self, cluster_id: int, limit: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, status, workspace_id, created_at, closed_at "
            "FROM topics WHERE cluster_id = %s ORDER BY closed_at DESC, id DESC LIMIT %s",
            (cluster_id, limit),
        ).fetchall()
        return [
            {
                "id": row[0], "label": row[1], "summary": row[2], "tags": row[3],
                "status": row[4], "workspace_id": row[5], "created_at": row[6], "closed_at": row[7],
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
        """See StorageProvider.search_clusters / SqliteStore's docstring.
        workspace_id is required (returns [] without one) and hard-filtered —
        clusters must never merge across workspaces."""
        if not workspace_id:
            return []
        scorer = RetrievalScorer(weights, now)

        if query_embedding is not None:
            rows = self._conn.execute(
                "SELECT id, workspace_id, label, summary, tags, topic_count, created_at, updated_at, "
                "1 - (embedding <=> %(q)s::vector) AS similarity "
                "FROM topic_clusters WHERE workspace_id = %(workspace_id)s AND embedding IS NOT NULL "
                "ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s",
                {"q": query_embedding, "workspace_id": workspace_id, "k": _knn_prefilter_k(limit)},
            ).fetchall()
        else:
            rows = [
                (*r, 0.0)
                for r in self._conn.execute(
                    "SELECT id, workspace_id, label, summary, tags, topic_count, created_at, updated_at "
                    "FROM topic_clusters WHERE workspace_id = %s",
                    (workspace_id,),
                ).fetchall()
            ]

        scored: list[tuple[float, dict, dict]] = []
        for row in rows:
            cluster = {
                "id": row[0], "workspace_id": row[1], "label": row[2], "summary": row[3],
                "tags": row[4], "topic_count": row[5], "created_at": row[6], "updated_at": row[7],
            }
            score, breakdown = scorer.score(
                candidate_type="cluster",
                candidate_id=cluster["id"],
                semantic_raw=row[8],
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
        """See StorageProvider.search_cluster_candidates / SqliteStore's
        docstring. Scores both raw unclustered topics and existing
        clusters against topic_id's own stored embedding/tags."""
        scorer = RetrievalScorer(weights, now)

        own_row = self._conn.execute(
            "SELECT embedding, tags FROM topics WHERE id = %s", (topic_id,)
        ).fetchone()
        if own_row is None or own_row[0] is None:
            return []
        query_embedding = _to_float_list(own_row[0])
        query_tags: list[str] = own_row[1]

        scored: list[tuple[float, dict, dict]] = []

        topic_rows = self._conn.execute(
            "SELECT id, label, summary, tags, created_at, closed_at, "
            "1 - (embedding <=> %(q)s::vector) AS similarity "
            "FROM topics WHERE status='closed' AND cluster_id IS NULL AND workspace_id = %(workspace_id)s "
            "AND id != %(topic_id)s AND embedding IS NOT NULL "
            "ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s",
            {"q": query_embedding, "workspace_id": workspace_id, "topic_id": topic_id, "k": _knn_prefilter_k(limit) + 1},
        ).fetchall()
        for row in topic_rows:
            candidate = {
                "candidate_kind": "topic", "id": row[0], "label": row[1],
                "summary": row[2], "tags": row[3], "topic_count": 1,
            }
            score, breakdown = scorer.score(
                candidate_type="cluster_candidate",
                candidate_id=candidate["id"],
                semantic_raw=row[6],
                recency_timestamp=row[5],
                include_tag_overlap=True,
                query_tags=query_tags,
                candidate_tags=candidate["tags"],
                extra_breakdown={"candidate_kind": "topic"},
            )
            scored.append((score, candidate, breakdown))

        cluster_rows = self._conn.execute(
            "SELECT id, label, summary, tags, updated_at, topic_count, "
            "1 - (embedding <=> %(q)s::vector) AS similarity "
            "FROM topic_clusters WHERE workspace_id = %(workspace_id)s AND embedding IS NOT NULL "
            "ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s",
            {"q": query_embedding, "workspace_id": workspace_id, "k": _knn_prefilter_k(limit)},
        ).fetchall()
        for row in cluster_rows:
            candidate = {
                "candidate_kind": "cluster", "id": row[0], "label": row[1],
                "summary": row[2], "tags": row[3], "topic_count": row[5],
            }
            score, breakdown = scorer.score(
                candidate_type="cluster_candidate",
                candidate_id=candidate["id"],
                semantic_raw=row[6],
                recency_timestamp=row[4],
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
        row = self._conn.execute(
            "INSERT INTO memory_items (content, item_type, source_peer, subject_peer, "
            "attribution_type, topic_id, message_id, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING id",
            (item.content, item.item_type, item.source_peer, item.subject_peer,
             item.attribution_type, item.topic_id, item.message_id, item.embedding),
        ).fetchone()
        assert row is not None  # RETURNING id always yields a row on successful INSERT
        return row[0]

    def get_memory_item(self, item_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, content, item_type, source_peer, subject_peer, attribution_type, "
            "topic_id, created_at FROM memory_items WHERE id = %s",
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
        """See StorageProvider.search_memory_items / SqliteStore's
        docstring — workspace_id, when given, is a hard filter joined through
        messages (memory_items carry no workspace_id of their own)."""
        budget = budget or MemoryItemBudget()
        scorer = RetrievalScorer(weights, now)

        if query_embedding is not None and workspace_id:
            rows = list(self._conn.execute(
                "SELECT mi.id, mi.content, mi.item_type, mi.source_peer, mi.subject_peer, "
                "mi.attribution_type, mi.topic_id, mi.message_id, mi.created_at, "
                "1 - (mi.embedding <=> %(q)s::vector) AS similarity "
                "FROM memory_items mi JOIN messages m ON mi.message_id = m.id "
                "WHERE m.workspace_id = %(workspace_id)s AND mi.embedding IS NOT NULL "
                "ORDER BY mi.embedding <=> %(q)s::vector LIMIT %(k)s",
                {"q": query_embedding, "workspace_id": workspace_id, "k": _knn_prefilter_k(budget.total)},
            ).fetchall())
            if budget.surprise > 0 or budget.feeling > 0:
                # Rare-type bypass: surprise/feeling are uncommon enough
                # that the KNN LIMIT above (embedding-similarity-only, no
                # item_type awareness) could exclude every candidate of
                # that type before type-aware selection ever runs. Fetch
                # every surprise/feeling item for this workspace directly
                # instead — real similarity via the same SQL expression,
                # just without the ORDER BY/LIMIT restriction — deduped
                # against the KNN pool's own ids.
                knn_ids = {r[0] for r in rows}
                bypass_rows = self._conn.execute(
                    "SELECT mi.id, mi.content, mi.item_type, mi.source_peer, mi.subject_peer, "
                    "mi.attribution_type, mi.topic_id, mi.message_id, mi.created_at, "
                    "1 - (mi.embedding <=> %(q)s::vector) AS similarity "
                    "FROM memory_items mi JOIN messages m ON mi.message_id = m.id "
                    "WHERE m.workspace_id = %(workspace_id)s AND mi.embedding IS NOT NULL "
                    "AND mi.item_type IN ('surprise', 'feeling')",
                    {"q": query_embedding, "workspace_id": workspace_id},
                ).fetchall()
                rows.extend(r for r in bypass_rows if r[0] not in knn_ids)
        elif workspace_id:
            rows = [
                (*r, 0.0)
                for r in self._conn.execute(
                    "SELECT mi.id, mi.content, mi.item_type, mi.source_peer, mi.subject_peer, "
                    "mi.attribution_type, mi.topic_id, mi.message_id, mi.created_at "
                    "FROM memory_items mi JOIN messages m ON mi.message_id = m.id "
                    "WHERE m.workspace_id = %s",
                    (workspace_id,),
                ).fetchall()
            ]
        else:
            rows = [
                (*r, 0.0)
                for r in self._conn.execute(
                    "SELECT id, content, item_type, source_peer, subject_peer, "
                    "attribution_type, topic_id, message_id, created_at "
                    "FROM memory_items"
                ).fetchall()
            ]

        scored: list[tuple[float, MemoryItem, dict]] = []
        for row in rows:
            item = MemoryItem(
                id=row[0], content=row[1], item_type=row[2], source_peer=row[3], subject_peer=row[4],
                attribution_type=row[5], topic_id=row[6], message_id=row[7], timestamp=row[8],
            )
            score, breakdown = scorer.score(
                candidate_type="memory_item",
                candidate_id=row[0],
                semantic_raw=row[9],
                recency_timestamp=row[8],
            )
            scored.append((score, item, breakdown))

        return scorer.finalize_tiered(scored=scored, budget=budget, tracer=tracer, request_id=request_id)

    def get_memory_items_by_turn(self, turn_id: int) -> list[MemoryItem]:
        rows = self._conn.execute(
            "SELECT mi.id, mi.content, mi.item_type, mi.source_peer, mi.subject_peer, "
            "mi.attribution_type, mi.topic_id, mi.message_id, mi.embedding, mi.created_at "
            "FROM memory_items mi JOIN messages m ON mi.message_id = m.id "
            "WHERE m.turn_id = %s ORDER BY mi.id ASC",
            (turn_id,),
        ).fetchall()
        return [
            MemoryItem(
                id=row[0], content=row[1], item_type=row[2], source_peer=row[3], subject_peer=row[4],
                attribution_type=row[5], topic_id=row[6], message_id=row[7],
                embedding=_to_float_list(row[8]), timestamp=row[9],
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
            "FROM memory_items WHERE topic_id = %s ORDER BY id ASC",
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
            conditions.append("item_type = %s")
            params.append(item_type)
        if topic_id is not None:
            conditions.append("topic_id = %s")
            params.append(topic_id)
        if q:
            # ILIKE for case-insensitivity. %/_/\ are escaped so a search
            # string containing them is matched literally, not as wildcards.
            escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append("content ILIKE %s ESCAPE '\\'")
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
        self, turn_id: int, role: str, content: str, workspace_id: str, topic_id: int | None = None,
        request_id: str | None = None,
    ) -> int:
        # with conn.transaction(): see assign_topic_to_cluster's docstring
        # — the INSERT and topics.message_count UPDATE commit together.
        with self._conn.transaction():
            row = self._conn.execute(
                "INSERT INTO messages (turn_id, role, content, workspace_id, topic_id, request_id) "
                "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
                (turn_id, role, content, workspace_id, topic_id, request_id),
            ).fetchone()
            if topic_id is not None:
                self._conn.execute(
                    "UPDATE topics SET message_count = message_count + 1 WHERE id = %s",
                    (topic_id,),
                )
        assert row is not None  # RETURNING id always yields a row on successful INSERT
        return row[0]

    def get_messages_by_topic(self, topic_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, turn_id, role, content, workspace_id, created_at "
            "FROM messages WHERE topic_id = %s ORDER BY turn_id, id",
            (topic_id,),
        ).fetchall()
        return [
            {"id": r[0], "turn_id": r[1], "role": r[2], "content": r[3], "workspace_id": r[4], "created_at": r[5]}
            for r in rows
        ]

    def get_recent_messages_by_topic(self, topic_id: int, limit: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, turn_id, role, content, workspace_id, created_at "
            "FROM messages WHERE topic_id = %s ORDER BY id DESC LIMIT %s",
            (topic_id, limit),
        ).fetchall()
        return [
            {"id": r[0], "turn_id": r[1], "role": r[2], "content": r[3], "workspace_id": r[4], "created_at": r[5]}
            for r in reversed(rows)
        ]

    def get_messages_by_request_id(self, request_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, turn_id, role, content, workspace_id, topic_id, created_at "
            "FROM messages WHERE request_id = %s ORDER BY id",
            (request_id,),
        ).fetchall()
        return [
            {
                "id": r[0], "turn_id": r[1], "role": r[2], "content": r[3],
                "workspace_id": r[4], "topic_id": r[5], "created_at": r[6],
            }
            for r in rows
        ]

    def get_recent_turns(self, limit: int, workspace_id: str | None = None) -> list[dict]:
        if workspace_id is not None:
            rows = self._conn.execute(
                "SELECT request_id, workspace_id, topic_id, content, created_at FROM messages "
                "WHERE role = 'user' AND request_id IS NOT NULL AND workspace_id = %s "
                "ORDER BY id DESC LIMIT %s",
                (workspace_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT request_id, workspace_id, topic_id, content, created_at FROM messages "
                "WHERE role = 'user' AND request_id IS NOT NULL "
                "ORDER BY id DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [
            {"request_id": r[0], "workspace_id": r[1], "topic_id": r[2], "content": r[3], "created_at": r[4]}
            for r in rows
        ]

    def count_recent_turns(self, workspace_id: str | None = None) -> int:
        if workspace_id is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE role = 'user' AND request_id IS NOT NULL AND workspace_id = %s",
                (workspace_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE role = 'user' AND request_id IS NOT NULL",
            ).fetchone()
        return row[0] if row else 0
