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
from core.models import Fact, MemoryItem
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

    def get_facts(self, peer_id: str) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT peer_id, category, key, value, source, importance, created_at "
            "FROM facts WHERE peer_id = %s "
            "ORDER BY CASE importance WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, category, key",
            (peer_id,),
        ).fetchall()
        return [
            Fact(peer_id=r[0], category=r[1], key=r[2], value=r[3], source=r[4], importance=r[5], created_at=r[6])
            for r in rows
        ]

    def set_fact(
        self, peer_id: str, category: str, key: str, value: str, source: str, importance: str = "medium"
    ) -> None:
        self._conn.execute(
            "INSERT INTO facts (peer_id, category, key, value, source, importance) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (peer_id, category, key) DO UPDATE SET "
            "value=excluded.value, source=excluded.source, importance=excluded.importance",
            (peer_id, category, key, value, source, importance),
        )
        self._conn.commit()

    def delete_fact(self, peer_id: str, category: str, key: str) -> None:
        self._conn.execute(
            "DELETE FROM facts WHERE peer_id = %s AND category = %s AND key = %s",
            (peer_id, category, key),
        )
        self._conn.commit()

    # ── Facts Consolidation ────────────────────────────────

    def count_user_turns(self, peer_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE peer_id = %s AND role = 'user'",
            (peer_id,),
        ).fetchone()
        return row[0] if row else 0

    def get_fact_items_since(self, subject_peer: str, since_item_id: int) -> list[MemoryItem]:
        rows = self._conn.execute(
            "SELECT id, content, item_type, source_peer, subject_peer, "
            "attribution_type, topic_id, message_id, created_at "
            "FROM memory_items "
            "WHERE item_type = 'fact' AND subject_peer = %s AND id > %s "
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

    def get_consolidation_checkpoint(self, peer_id: str) -> int:
        row = self._conn.execute(
            "SELECT last_item_id FROM fact_consolidation_state WHERE peer_id = %s",
            (peer_id,),
        ).fetchone()
        return row[0] if row else 0

    def set_consolidation_checkpoint(self, peer_id: str, last_item_id: int) -> None:
        self._conn.execute(
            "INSERT INTO fact_consolidation_state (peer_id, last_item_id, last_run_at) "
            "VALUES (%s, %s, now()) "
            "ON CONFLICT (peer_id) DO UPDATE SET last_item_id=excluded.last_item_id, last_run_at=excluded.last_run_at",
            (peer_id, last_item_id),
        )
        self._conn.commit()

    # ── Topics ─────────────────────────────────────────────

    def create_topic(
        self,
        label: str,
        summary: str = "",
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
        peer_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        row = self._conn.execute(
            "INSERT INTO topics (label, summary, tags, embedding, peer_id, session_id) "
            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (label, summary, Jsonb(tags or []), embedding, peer_id, session_id),
        ).fetchone()
        self._conn.commit()
        assert row is not None  # RETURNING id always yields a row on successful INSERT
        return row[0]

    def get_topic(self, topic_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, label, summary, tags, status, peer_id, created_at, closed_at, message_count "
            "FROM topics WHERE id = %s",
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
            "peer_id": row[5],
            "created_at": row[6],
            "closed_at": row[7],
            "message_count": row[8],
        }

    def close_topic(self, topic_id: int, summary: str) -> None:
        self._conn.execute(
            "UPDATE topics SET status='closed', summary=%s, closed_at=now() WHERE id=%s",
            (summary, topic_id),
        )
        self._conn.commit()

    def get_active_topic(self, peer_id: str, session_id: str | None = None) -> int | None:
        """See StorageProvider.get_active_topic / SqliteStore's docstring
        for the full (peer_id, session_id) scoping rationale. Uses
        Postgres's ``IS NOT DISTINCT FROM`` — the exact NULL-safe
        equivalent of SQLite's ``session_id IS ?`` binding."""
        row = self._conn.execute(
            "SELECT id FROM topics WHERE peer_id = %s AND session_id IS NOT DISTINCT FROM %s AND status = 'active' "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (peer_id, session_id),
        ).fetchone()
        return row[0] if row else None

    def get_stale_active_topics(self, cutoff: datetime, peer_id: str | None = None) -> list[dict]:
        """See SqliteStore.get_stale_active_topics. Real TIMESTAMPTZ
        columns mean the staleness comparison can happen directly in SQL
        (via HAVING) rather than SQLite's Python-side string-timestamp
        parsing workaround."""
        query = (
            "SELECT t.id, t.label, t.summary, t.tags, t.status, t.peer_id, t.created_at, t.closed_at, "
            "COALESCE(MAX(m.created_at), t.created_at) AS last_activity "
            "FROM topics t LEFT JOIN messages m ON m.topic_id = t.id "
            "WHERE t.status = 'active'"
        )
        params: list = []
        if peer_id is not None:
            query += " AND t.peer_id = %s"
            params.append(peer_id)
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
                "peer_id": row[5],
                "created_at": row[6],
                "closed_at": row[7],
                "last_activity": row[8],
            }
            for row in rows
        ]

    def get_recent_topics(self, peer_id: str, limit: int, min_message_count: int = -1) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, embedding, status, peer_id, created_at, closed_at "
            "FROM topics WHERE status='closed' AND cluster_id IS NULL AND peer_id = %s "
            "AND message_count > %s "
            "ORDER BY closed_at DESC, id DESC LIMIT %s",
            (peer_id, min_message_count, limit),
        ).fetchall()
        return [
            {
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": row[3],
                "status": row[5],
                "peer_id": row[6],
                "created_at": row[7],
                "closed_at": row[8],
            }
            for row in rows
        ]

    def update_topic_embedding(self, topic_id: int, embedding: list[float]) -> None:
        self._conn.execute("UPDATE topics SET embedding = %s WHERE id = %s", (embedding, topic_id))
        self._conn.commit()

    def update_topic_tags(self, topic_id: int, tags: list[str]) -> None:
        self._conn.execute("UPDATE topics SET tags = %s WHERE id = %s", (Jsonb(tags), topic_id))
        self._conn.commit()

    def update_topic_label(self, topic_id: int, label: str) -> None:
        self._conn.execute("UPDATE topics SET label = %s WHERE id = %s", (label, topic_id))
        self._conn.commit()

    def count_messages_by_topic(self, topic_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE topic_id = %s", (topic_id,)
        ).fetchone()
        return row[0] if row else 0

    def search_topics(
        self,
        query_embedding: list[float] | None = None,
        tags: list[str] | None = None,
        peer_id: str | None = None,
        limit: int = 5,
        weights: dict[str, float] | None = None,
        tracer: Tracer | None = None,
        request_id: str | None = None,
        now: datetime | None = None,
        min_message_count: int = -1,
    ) -> list[dict]:
        """See StorageProvider.search_topics / SqliteStore's docstring for
        the full scoring semantics (peer_id as a hard filter, temporal as
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
            if peer_id:
                rows = self._conn.execute(
                    "SELECT id, label, summary, tags, status, peer_id, created_at, closed_at, "
                    "1 - (embedding <=> %(q)s::vector) AS similarity "
                    "FROM topics WHERE status='closed' AND cluster_id IS NULL AND peer_id = %(peer_id)s "
                    "AND embedding IS NOT NULL AND message_count > %(min_msgs)s "
                    "ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s",
                    {"q": query_embedding, "peer_id": peer_id, "k": k, "min_msgs": min_message_count},
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT id, label, summary, tags, status, peer_id, created_at, closed_at, "
                    "1 - (embedding <=> %(q)s::vector) AS similarity "
                    "FROM topics WHERE status='closed' AND cluster_id IS NULL AND embedding IS NOT NULL "
                    "AND message_count > %(min_msgs)s "
                    "ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s",
                    {"q": query_embedding, "k": k, "min_msgs": min_message_count},
                ).fetchall()
        elif peer_id:
            rows = [
                (*r, 0.0)
                for r in self._conn.execute(
                    "SELECT id, label, summary, tags, status, peer_id, created_at, closed_at "
                    "FROM topics WHERE status='closed' AND cluster_id IS NULL AND peer_id = %s "
                    "AND message_count > %s",
                    (peer_id, min_message_count),
                ).fetchall()
            ]
        else:
            rows = [
                (*r, 0.0)
                for r in self._conn.execute(
                    "SELECT id, label, summary, tags, status, peer_id, created_at, closed_at "
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
                "peer_id": row[5],
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
                person_match=bool(peer_id) and topic["peer_id"] == peer_id,
            )
            scored.append((score, topic, breakdown))

        return scorer.finalize(
            scored=scored, candidate_type="topic", limit=limit, tracer=tracer, request_id=request_id,
        )

    # ── Topic Clusters ─────────────────────────────────────

    def get_clusters(self, peer_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, peer_id, label, summary, tags, topic_count, created_at, updated_at "
            "FROM topic_clusters WHERE peer_id = %s ORDER BY id",
            (peer_id,),
        ).fetchall()
        return [
            {
                "id": row[0],
                "peer_id": row[1],
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
        peer_id: str,
        label: str,
        summary: str,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> int:
        row = self._conn.execute(
            "INSERT INTO topic_clusters (peer_id, label, summary, tags, embedding) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (peer_id, label, summary, Jsonb(tags or []), embedding),
        ).fetchone()
        self._conn.commit()
        assert row is not None  # RETURNING id always yields a row on successful INSERT
        return row[0]

    def get_cluster(self, cluster_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, peer_id, label, summary, tags, topic_count, created_at, updated_at "
            "FROM topic_clusters WHERE id = %s",
            (cluster_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "peer_id": row[1],
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
        self._conn.commit()

    def assign_topic_to_cluster(self, topic_id: int, cluster_id: int) -> None:
        """Real transaction here (both statements share one commit) —
        SqliteStore's equivalent commits each statement separately, a
        known non-atomic gap. Free to fix on this backend since Postgres
        makes it trivial; not backported to SQLite in this change."""
        self._conn.execute("UPDATE topics SET cluster_id = %s WHERE id = %s", (cluster_id, topic_id))
        self._conn.execute(
            "UPDATE topic_clusters SET topic_count = topic_count + 1 WHERE id = %s", (cluster_id,)
        )
        self._conn.commit()

    def get_topics_by_cluster(self, cluster_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, status, peer_id, created_at, closed_at "
            "FROM topics WHERE cluster_id = %s ORDER BY closed_at ASC, id ASC",
            (cluster_id,),
        ).fetchall()
        return [
            {
                "id": row[0], "label": row[1], "summary": row[2], "tags": row[3],
                "status": row[4], "peer_id": row[5], "created_at": row[6], "closed_at": row[7],
            }
            for row in rows
        ]

    def get_recent_topics_in_cluster(self, cluster_id: int, limit: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, status, peer_id, created_at, closed_at "
            "FROM topics WHERE cluster_id = %s ORDER BY closed_at DESC, id DESC LIMIT %s",
            (cluster_id, limit),
        ).fetchall()
        return [
            {
                "id": row[0], "label": row[1], "summary": row[2], "tags": row[3],
                "status": row[4], "peer_id": row[5], "created_at": row[6], "closed_at": row[7],
            }
            for row in rows
        ]

    def search_clusters(
        self,
        query_embedding: list[float] | None = None,
        tags: list[str] | None = None,
        peer_id: str | None = None,
        limit: int = 5,
        weights: dict[str, float] | None = None,
        tracer: Tracer | None = None,
        request_id: str | None = None,
        now: datetime | None = None,
    ) -> list[dict]:
        """See StorageProvider.search_clusters / SqliteStore's docstring.
        peer_id is required (returns [] without one) and hard-filtered —
        clusters must never merge across peers."""
        if not peer_id:
            return []
        scorer = RetrievalScorer(weights, now)

        if query_embedding is not None:
            rows = self._conn.execute(
                "SELECT id, peer_id, label, summary, tags, topic_count, created_at, updated_at, "
                "1 - (embedding <=> %(q)s::vector) AS similarity "
                "FROM topic_clusters WHERE peer_id = %(peer_id)s AND embedding IS NOT NULL "
                "ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s",
                {"q": query_embedding, "peer_id": peer_id, "k": _knn_prefilter_k(limit)},
            ).fetchall()
        else:
            rows = [
                (*r, 0.0)
                for r in self._conn.execute(
                    "SELECT id, peer_id, label, summary, tags, topic_count, created_at, updated_at "
                    "FROM topic_clusters WHERE peer_id = %s",
                    (peer_id,),
                ).fetchall()
            ]

        scored: list[tuple[float, dict, dict]] = []
        for row in rows:
            cluster = {
                "id": row[0], "peer_id": row[1], "label": row[2], "summary": row[3],
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
        peer_id: str,
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
            "FROM topics WHERE status='closed' AND cluster_id IS NULL AND peer_id = %(peer_id)s "
            "AND id != %(topic_id)s AND embedding IS NOT NULL "
            "ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s",
            {"q": query_embedding, "peer_id": peer_id, "topic_id": topic_id, "k": _knn_prefilter_k(limit) + 1},
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
            "FROM topic_clusters WHERE peer_id = %(peer_id)s AND embedding IS NOT NULL "
            "ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s",
            {"q": query_embedding, "peer_id": peer_id, "k": _knn_prefilter_k(limit)},
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
        self._conn.commit()
        assert row is not None  # RETURNING id always yields a row on successful INSERT
        return row[0]

    def search_memory_items(
        self,
        query_embedding: list[float] | None = None,
        peer_id: str | None = None,
        limit: int = 5,
        weights: dict[str, float] | None = None,
        tracer: Tracer | None = None,
        request_id: str | None = None,
        now: datetime | None = None,
    ) -> list[MemoryItem]:
        """See StorageProvider.search_memory_items / SqliteStore's
        docstring — peer_id, when given, is a hard filter joined through
        messages (memory_items carry no peer_id of their own)."""
        scorer = RetrievalScorer(weights, now)

        if query_embedding is not None and peer_id:
            rows = self._conn.execute(
                "SELECT mi.id, mi.content, mi.item_type, mi.source_peer, mi.subject_peer, "
                "mi.attribution_type, mi.topic_id, mi.message_id, mi.created_at, "
                "1 - (mi.embedding <=> %(q)s::vector) AS similarity "
                "FROM memory_items mi JOIN messages m ON mi.message_id = m.id "
                "WHERE m.peer_id = %(peer_id)s AND mi.embedding IS NOT NULL "
                "ORDER BY mi.embedding <=> %(q)s::vector LIMIT %(k)s",
                {"q": query_embedding, "peer_id": peer_id, "k": _knn_prefilter_k(limit)},
            ).fetchall()
        elif peer_id:
            rows = [
                (*r, 0.0)
                for r in self._conn.execute(
                    "SELECT mi.id, mi.content, mi.item_type, mi.source_peer, mi.subject_peer, "
                    "mi.attribution_type, mi.topic_id, mi.message_id, mi.created_at "
                    "FROM memory_items mi JOIN messages m ON mi.message_id = m.id "
                    "WHERE m.peer_id = %s",
                    (peer_id,),
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

        return scorer.finalize(
            scored=scored, candidate_type="memory_item", limit=limit, tracer=tracer, request_id=request_id,
        )

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

    # ── Messages ───────────────────────────────────────────

    def store_message(
        self, turn_id: int, role: str, content: str, peer_id: str, topic_id: int | None = None,
    ) -> int:
        row = self._conn.execute(
            "INSERT INTO messages (turn_id, role, content, peer_id, topic_id) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (turn_id, role, content, peer_id, topic_id),
        ).fetchone()
        if topic_id is not None:
            self._conn.execute(
                "UPDATE topics SET message_count = message_count + 1 WHERE id = %s",
                (topic_id,),
            )
        self._conn.commit()
        assert row is not None  # RETURNING id always yields a row on successful INSERT
        return row[0]

    def get_messages_by_topic(self, topic_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, turn_id, role, content, peer_id, created_at "
            "FROM messages WHERE topic_id = %s ORDER BY turn_id, id",
            (topic_id,),
        ).fetchall()
        return [
            {"id": r[0], "turn_id": r[1], "role": r[2], "content": r[3], "peer_id": r[4], "created_at": r[5]}
            for r in rows
        ]

    def get_recent_messages_by_topic(self, topic_id: int, limit: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, turn_id, role, content, peer_id, created_at "
            "FROM messages WHERE topic_id = %s ORDER BY id DESC LIMIT %s",
            (topic_id, limit),
        ).fetchall()
        return [
            {"id": r[0], "turn_id": r[1], "role": r[2], "content": r[3], "peer_id": r[4], "created_at": r[5]}
            for r in reversed(rows)
        ]
