"""SQLite storage backend for Eidolon.

Implements the ``StorageProvider`` protocol using SQLite with:
- Auto-migration via ``PRAGMA user_version``
- JSON tags for flexible topic tagging
- BLOB embedding storage for vector search
- Cosine-similarity approximation for embedding search
"""

from __future__ import annotations

import json
import sqlite3
import struct
from datetime import UTC, datetime
from pathlib import Path

from core.interfaces import Tracer
from core.models import Fact, MemoryItem
from core.stores.migrations import migrate

# Fallback scoring weights when the caller doesn't supply an explicit
# `weights` override (e.g. from EidolonConfig.retrieval). Shared by both
# search_topics and search_memory_items, matching AssociativeRetriever's
# own defaults so a configured value means the same thing everywhere.
_DEFAULT_WEIGHTS = {"semantic": 0.4, "tag_overlap": 0.3, "person": 0.2, "temporal": 0.1}

# Recency half-life: a candidate's temporal score halves for every this-many
# days of age (see `_recency_score`). The *shape* of the decay is fixed here;
# how much it matters is the tunable "temporal" weight above. 30 days is a
# sensible default until real chat data suggests otherwise.
_TEMPORAL_HALF_LIFE_DAYS = 30.0

# How many extra scored candidates beyond `limit` to persist in a
# retrieval_scoring_summary trace event, so near-misses just below the
# cutoff are visible when tuning weights — without persisting every
# positively-scored candidate (which scales with DB size, not turn count).
_TRACE_TOP_K_PADDING = 10


def _serialize_embedding(vec: list[float]) -> bytes:
    """Pack a float list into a binary BLOB (little-endian f64)."""
    return struct.pack(f"<{len(vec)}d", *vec)


def _deserialize_embedding(blob: bytes) -> list[float]:
    """Unpack a binary BLOB back into a float list."""
    count = len(blob) // 8
    return list(struct.unpack(f"<{count}d", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    # strict=False keeps the historical behavior of truncating to the
    # shorter vector rather than raising on a length mismatch.
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _recency_score(timestamp: str | None, now: datetime, half_life_days: float) -> float:
    """Exponential recency decay in (0, 1]: ~1.0 for "just now", halving
    every ``half_life_days`` of age.

    Timestamps are the raw SQLite strings (e.g. "2026-07-20 16:57:00"),
    stored in UTC. An unknown/unparseable timestamp scores 0.0 (no recency
    signal), and a future timestamp (host clock skew) is clamped to 1.0.
    """
    if not timestamp:
        return 0.0
    try:
        ts = datetime.fromisoformat(timestamp)
    except ValueError:
        return 0.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    age_days = (now - ts).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


class SqliteStore:
    """SQLite implementation of ``StorageProvider``.

    Parameters
    ----------
    path : str
        Filesystem path to the SQLite database file.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path).expanduser().resolve()
        self._db: sqlite3.Connection | None = None

    # ── Lifecycle ──────────────────────────────────────────

    def connect(self) -> None:
        """Open the database and run pending migrations."""
        # Ensure the parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        migrate(self._db)

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

    def get_facts(self, peer_id: str) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT peer_id, category, key, value, source, created_at "
            "FROM facts WHERE peer_id = ? ORDER BY category, key",
            (peer_id,),
        ).fetchall()
        return [Fact(peer_id=r[0], category=r[1], key=r[2], value=r[3], source=r[4], created_at=r[5]) for r in rows]

    def set_fact(self, peer_id: str, category: str, key: str, value: str, source: str) -> None:
        self._conn.execute(
            "INSERT INTO facts (peer_id, category, key, value, source) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(peer_id, category, key) DO UPDATE SET value=excluded.value, source=excluded.source",
            (peer_id, category, key, value, source),
        )
        self._conn.commit()

    def delete_fact(self, peer_id: str, category: str, key: str) -> None:
        self._conn.execute(
            "DELETE FROM facts WHERE peer_id = ? AND category = ? AND key = ?",
            (peer_id, category, key),
        )
        self._conn.commit()

    # ── Facts Consolidation ────────────────────────────────

    def count_user_turns(self, peer_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE peer_id = ? AND role = 'user'",
            (peer_id,),
        ).fetchone()
        return row[0]

    def get_fact_items_since(self, subject_peer: str, since_item_id: int) -> list[MemoryItem]:
        rows = self._conn.execute(
            "SELECT id, content, item_type, source_peer, subject_peer, "
            "attribution_type, topic_id, message_id, created_at "
            "FROM memory_items "
            "WHERE item_type = 'fact' AND subject_peer = ? AND id > ? "
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
            "SELECT last_item_id FROM fact_consolidation_state WHERE peer_id = ?",
            (peer_id,),
        ).fetchone()
        return row[0] if row else 0

    def set_consolidation_checkpoint(self, peer_id: str, last_item_id: int) -> None:
        self._conn.execute(
            "INSERT INTO fact_consolidation_state (peer_id, last_item_id, last_run_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(peer_id) DO UPDATE SET last_item_id=excluded.last_item_id, last_run_at=excluded.last_run_at",
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
        tags_json = json.dumps(tags or [])
        embed_blob = _serialize_embedding(embedding) if embedding else None
        cur = self._conn.execute(
            "INSERT INTO topics (label, summary, tags, embedding, peer_id, session_id) VALUES (?, ?, ?, ?, ?, ?)",
            (label, summary, tags_json, embed_blob, peer_id, session_id),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_topic(self, topic_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, label, summary, tags, status, peer_id, created_at, closed_at "
            "FROM topics WHERE id = ?",
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
            "peer_id": row[5],
            "created_at": row[6],
            "closed_at": row[7],
        }

    def close_topic(self, topic_id: int, summary: str) -> None:
        self._conn.execute(
            "UPDATE topics SET status='closed', summary=?, closed_at=CURRENT_TIMESTAMP WHERE id=?",
            (summary, topic_id),
        )
        self._conn.commit()

    def get_active_topic(self, peer_id: str, session_id: str | None = None) -> int | None:
        """Return the id of the most recently opened 'active' topic for
        this peer *and* session.

        Scoped per (peer_id, session_id) — a peer with several concurrent
        sessions (web, Telegram, CLI, ...) gets one independent active
        topic per session, not one shared across all of them. Passing no
        session_id matches only topics created with none (e.g. legacy
        rows from before this column existed, or single-session callers
        like the CLI) — it is not a wildcard across sessions.

        If more than one is somehow left 'active' for the same
        (peer_id, session_id) (e.g. a crash mid topic transition), the
        most recent wins; older orphans stay open until explicitly
        closed — no automatic cleanup here.
        """
        # `id DESC` breaks ties from CURRENT_TIMESTAMP's second-level
        # resolution — two topics created within the same second would
        # otherwise sort ambiguously.
        row = self._conn.execute(
            "SELECT id FROM topics WHERE peer_id = ? AND session_id IS ? AND status = 'active' "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (peer_id, session_id),
        ).fetchone()
        return row[0] if row else None

    def get_recent_topics(self, peer_id: str, limit: int) -> list[dict]:
        """Return `peer_id`'s most recently closed topics, newest first.

        Ordered purely by `closed_at` — no relevance scoring, unlike
        `search_topics`. `id DESC` as a tiebreak for the same reason as
        `get_active_topic`: CURRENT_TIMESTAMP only has second resolution.
        """
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, embedding, status, peer_id, created_at, closed_at "
            "FROM topics WHERE status='closed' AND cluster_id IS NULL AND peer_id = ? "
            "ORDER BY closed_at DESC, id DESC LIMIT ?",
            (peer_id, limit),
        ).fetchall()
        return [
            {
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": json.loads(row[3]),
                "status": row[5],
                "peer_id": row[6],
                "created_at": row[7],
                "closed_at": row[8],
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

    def update_topic_tags(self, topic_id: int, tags: list[str]) -> None:
        self._conn.execute(
            "UPDATE topics SET tags = ? WHERE id = ?",
            (json.dumps(tags), topic_id),
        )
        self._conn.commit()

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
    ) -> list[dict]:
        """Search topics by embedding similarity, tag overlap, peer, and recency.

        `weights` overrides the semantic/tag_overlap/person/temporal scoring
        weights when provided (see `_DEFAULT_WEIGHTS`). The `temporal` term is
        a recency *re-ranking* signal: it's added only to candidates that
        already have some semantic/tag/person relevance, so it re-orders
        relevant topics by recency but never pulls in a merely-recent one.
        `now` is the reference time for the recency decay (defaults to the
        current UTC time); a topic's recency uses its `closed_at` (last
        active) falling back to `created_at`.

        When `tracer` is given, one `retrieval_scoring_summary` event is
        recorded for the whole call, carrying the top `limit + _TRACE_TOP_K_PADDING`
        positively-scored candidates (with breakdown) plus total/included counts —
        not one event per candidate, which doesn't scale with DB size.

        Returns topic dicts ordered by relevance (highest first).
        """
        w = weights or _DEFAULT_WEIGHTS
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, embedding, status, peer_id, created_at, closed_at "
            "FROM topics WHERE status='closed' AND cluster_id IS NULL"
        ).fetchall()

        scored: list[tuple[float, dict, dict]] = []

        for row in rows:
            topic = {
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": json.loads(row[3]),
                "status": row[5],
                "peer_id": row[6],
                "created_at": row[7],
                "closed_at": row[8],
            }
            score = 0.0
            semantic_raw = semantic_weighted = 0.0
            tag_overlap_raw = tag_overlap_weighted = 0.0
            person_weighted = 0.0

            # Embedding similarity (semantic)
            if query_embedding and row[4]:
                stored_emb = _deserialize_embedding(row[4])
                semantic_raw = _cosine_similarity(query_embedding, stored_emb)
                semantic_weighted = semantic_raw * w.get("semantic", _DEFAULT_WEIGHTS["semantic"])
                score += semantic_weighted

            # Tag overlap
            if tags:
                topic_tags: list[str] = topic["tags"]
                overlap = len(set(tags) & set(topic_tags))
                if overlap > 0:
                    tag_overlap_raw = overlap / max(len(tags), 1)
                    tag_overlap_weighted = tag_overlap_raw * w.get(
                        "tag_overlap", _DEFAULT_WEIGHTS["tag_overlap"]
                    )
                    score += tag_overlap_weighted

            # Peer filter
            if peer_id and topic["peer_id"] == peer_id:
                person_weighted = w.get("person", _DEFAULT_WEIGHTS["person"])
                score += person_weighted

            # Temporal recency — a re-ranking signal, not a relevance source:
            # only added to candidates that already have relevance (score > 0),
            # so a merely-recent-but-irrelevant topic is never pulled in.
            temporal_raw = _recency_score(
                topic["closed_at"] or topic["created_at"], now, _TEMPORAL_HALF_LIFE_DAYS
            )
            temporal_weighted = temporal_raw * w.get("temporal", _DEFAULT_WEIGHTS["temporal"])
            final_score = score + temporal_weighted if score > 0 else 0.0

            breakdown = {
                "candidate_type": "topic",
                "candidate_id": topic["id"],
                "semantic_raw": semantic_raw,
                "semantic_weighted": semantic_weighted,
                "tag_overlap_raw": tag_overlap_raw,
                "tag_overlap_weighted": tag_overlap_weighted,
                "person_weighted": person_weighted,
                "temporal_raw": temporal_raw,
                "temporal_weighted": temporal_weighted,
                "final_score": final_score,
            }
            scored.append((final_score, topic, breakdown))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [(s, t) for s, t, _ in scored if s > 0][:limit]
        included_ids = {t["id"] for _, t in results}

        if tracer is not None:
            positively_scored = [(s, b) for s, _, b in scored if s > 0]
            top = positively_scored[: limit + _TRACE_TOP_K_PADDING]
            for _, breakdown in top:
                breakdown["included"] = breakdown["candidate_id"] in included_ids
            tracer.record(
                "retrieval_scoring_summary",
                {
                    "candidate_type": "topic",
                    "total_scored": len(positively_scored),
                    "included_count": len(included_ids),
                    "top_candidates": [b for _, b in top],
                },
                request_id=request_id,
            )

        return [t for _, t in results]

    # ── Topic Clusters ─────────────────────────────────────

    def get_clusters(self, peer_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, peer_id, label, summary, tags, topic_count, created_at, updated_at "
            "FROM topic_clusters WHERE peer_id = ? ORDER BY id",
            (peer_id,),
        ).fetchall()
        return [
            {
                "id": row[0],
                "peer_id": row[1],
                "label": row[2],
                "summary": row[3],
                "tags": json.loads(row[4]),
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
        tags_json = json.dumps(tags or [])
        embed_blob = _serialize_embedding(embedding) if embedding else None
        cur = self._conn.execute(
            "INSERT INTO topic_clusters (peer_id, label, summary, tags, embedding) VALUES (?, ?, ?, ?, ?)",
            (peer_id, label, summary, tags_json, embed_blob),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_cluster(self, cluster_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT id, peer_id, label, summary, tags, topic_count, created_at, updated_at "
            "FROM topic_clusters WHERE id = ?",
            (cluster_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "peer_id": row[1],
            "label": row[2],
            "summary": row[3],
            "tags": json.loads(row[4]),
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
        sets = ["summary = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: list = [summary]
        if label is not None:
            sets.append("label = ?")
            params.append(label)
        if tags is not None:
            sets.append("tags = ?")
            params.append(json.dumps(tags))
        if embedding is not None:
            sets.append("embedding = ?")
            params.append(_serialize_embedding(embedding))
        params.append(cluster_id)
        self._conn.execute(f"UPDATE topic_clusters SET {', '.join(sets)} WHERE id = ?", params)
        self._conn.commit()

    def assign_topic_to_cluster(self, topic_id: int, cluster_id: int) -> None:
        self._conn.execute("UPDATE topics SET cluster_id = ? WHERE id = ?", (cluster_id, topic_id))
        self._conn.execute(
            "UPDATE topic_clusters SET topic_count = topic_count + 1 WHERE id = ?", (cluster_id,)
        )
        self._conn.commit()

    def get_topics_by_cluster(self, cluster_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, status, peer_id, created_at, closed_at "
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
                "peer_id": row[5],
                "created_at": row[6],
                "closed_at": row[7],
            }
            for row in rows
        ]

    def get_recent_topics_in_cluster(self, cluster_id: int, limit: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, status, peer_id, created_at, closed_at "
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
                "peer_id": row[5],
                "created_at": row[6],
                "closed_at": row[7],
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
        """See StorageProvider.search_clusters. Same scoring shape as
        search_topics, but scanning topic_clusters and hard-filtering
        peer_id in SQL instead of scoring it as a soft "person" bonus —
        clusters must never merge across peers.
        """
        w = weights or _DEFAULT_WEIGHTS
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)

        if not peer_id:
            return []

        rows = self._conn.execute(
            "SELECT id, peer_id, label, summary, tags, embedding, topic_count, created_at, updated_at "
            "FROM topic_clusters WHERE peer_id = ?",
            (peer_id,),
        ).fetchall()

        scored: list[tuple[float, dict, dict]] = []

        for row in rows:
            cluster = {
                "id": row[0],
                "peer_id": row[1],
                "label": row[2],
                "summary": row[3],
                "tags": json.loads(row[4]),
                "topic_count": row[6],
                "created_at": row[7],
                "updated_at": row[8],
            }
            score = 0.0
            semantic_raw = semantic_weighted = 0.0
            tag_overlap_raw = tag_overlap_weighted = 0.0

            if query_embedding and row[5]:
                stored_emb = _deserialize_embedding(row[5])
                semantic_raw = _cosine_similarity(query_embedding, stored_emb)
                semantic_weighted = semantic_raw * w.get("semantic", _DEFAULT_WEIGHTS["semantic"])
                score += semantic_weighted

            if tags:
                cluster_tags: list[str] = cluster["tags"]
                overlap = len(set(tags) & set(cluster_tags))
                if overlap > 0:
                    tag_overlap_raw = overlap / max(len(tags), 1)
                    tag_overlap_weighted = tag_overlap_raw * w.get(
                        "tag_overlap", _DEFAULT_WEIGHTS["tag_overlap"]
                    )
                    score += tag_overlap_weighted

            # No "person" term — peer_id is already a hard SQL filter above.
            temporal_raw = _recency_score(cluster["updated_at"], now, _TEMPORAL_HALF_LIFE_DAYS)
            temporal_weighted = temporal_raw * w.get("temporal", _DEFAULT_WEIGHTS["temporal"])
            final_score = score + temporal_weighted if score > 0 else 0.0

            breakdown = {
                "candidate_type": "cluster",
                "candidate_id": cluster["id"],
                "semantic_raw": semantic_raw,
                "semantic_weighted": semantic_weighted,
                "tag_overlap_raw": tag_overlap_raw,
                "tag_overlap_weighted": tag_overlap_weighted,
                "temporal_raw": temporal_raw,
                "temporal_weighted": temporal_weighted,
                "final_score": final_score,
            }
            scored.append((final_score, cluster, breakdown))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [(s, c) for s, c, _ in scored if s > 0][:limit]
        included_ids = {c["id"] for _, c in results}

        if tracer is not None:
            positively_scored = [(s, b) for s, _, b in scored if s > 0]
            top = positively_scored[: limit + _TRACE_TOP_K_PADDING]
            for _, breakdown in top:
                breakdown["included"] = breakdown["candidate_id"] in included_ids
            tracer.record(
                "retrieval_scoring_summary",
                {
                    "candidate_type": "cluster",
                    "total_scored": len(positively_scored),
                    "included_count": len(included_ids),
                    "top_candidates": [b for _, b in top],
                },
                request_id=request_id,
            )

        return [c for _, c in results]

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
        """See StorageProvider.search_cluster_candidates. Scores both raw
        unclustered topics and existing clusters against topic_id's own
        stored embedding/tags, so a first-ever pair can form a brand new
        cluster and not just grow an existing one.
        """
        w = weights or _DEFAULT_WEIGHTS
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)

        own_row = self._conn.execute(
            "SELECT embedding, tags FROM topics WHERE id = ?", (topic_id,)
        ).fetchone()
        if own_row is None or own_row[0] is None:
            return []
        query_embedding = _deserialize_embedding(own_row[0])
        query_tags: list[str] = json.loads(own_row[1])

        scored: list[tuple[float, dict, dict]] = []

        topic_rows = self._conn.execute(
            "SELECT id, label, summary, tags, embedding, created_at, closed_at "
            "FROM topics WHERE status='closed' AND cluster_id IS NULL AND peer_id = ? AND id != ?",
            (peer_id, topic_id),
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
            score, breakdown = self._score_cluster_candidate(
                candidate, row[4], row[6], query_embedding, query_tags, w, now
            )
            scored.append((score, candidate, breakdown))

        cluster_rows = self._conn.execute(
            "SELECT id, label, summary, tags, embedding, updated_at "
            "FROM topic_clusters WHERE peer_id = ?",
            (peer_id,),
        ).fetchall()
        for row in cluster_rows:
            cluster_full = self.get_cluster(row[0])
            candidate = {
                "candidate_kind": "cluster",
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": json.loads(row[3]),
                "topic_count": cluster_full["topic_count"] if cluster_full else 0,
            }
            score, breakdown = self._score_cluster_candidate(
                candidate, row[4], row[5], query_embedding, query_tags, w, now
            )
            scored.append((score, candidate, breakdown))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [(s, c) for s, c, _ in scored if s > 0][:limit]
        included_ids = {(c["candidate_kind"], c["id"]) for _, c in results}

        if tracer is not None:
            positively_scored = [(s, b) for s, _, b in scored if s > 0]
            top = positively_scored[: limit + _TRACE_TOP_K_PADDING]
            for _, breakdown in top:
                breakdown["included"] = (breakdown["candidate_kind"], breakdown["candidate_id"]) in included_ids
            tracer.record(
                "retrieval_scoring_summary",
                {
                    "candidate_type": "cluster_candidate",
                    "topic_id": topic_id,
                    "total_scored": len(positively_scored),
                    "included_count": len(included_ids),
                    "top_candidates": [b for _, b in top],
                },
                request_id=request_id,
            )

        return [c for _, c in results]

    @staticmethod
    def _score_cluster_candidate(
        candidate: dict,
        embedding_blob: bytes | None,
        recency_timestamp: str | None,
        query_embedding: list[float],
        query_tags: list[str],
        weights: dict[str, float],
        now: datetime,
    ) -> tuple[float, dict]:
        """Shared scoring formula for one search_cluster_candidates row —
        semantic + tag_overlap + temporal, same shape as search_topics/
        search_clusters (no "person" term; the caller already scopes the
        query to one peer_id in SQL)."""
        score = 0.0
        semantic_raw = semantic_weighted = 0.0
        tag_overlap_raw = tag_overlap_weighted = 0.0

        if embedding_blob:
            stored_emb = _deserialize_embedding(embedding_blob)
            semantic_raw = _cosine_similarity(query_embedding, stored_emb)
            semantic_weighted = semantic_raw * weights.get("semantic", _DEFAULT_WEIGHTS["semantic"])
            score += semantic_weighted

        if query_tags:
            overlap = len(set(query_tags) & set(candidate["tags"]))
            if overlap > 0:
                tag_overlap_raw = overlap / max(len(query_tags), 1)
                tag_overlap_weighted = tag_overlap_raw * weights.get(
                    "tag_overlap", _DEFAULT_WEIGHTS["tag_overlap"]
                )
                score += tag_overlap_weighted

        temporal_raw = _recency_score(recency_timestamp, now, _TEMPORAL_HALF_LIFE_DAYS)
        temporal_weighted = temporal_raw * weights.get("temporal", _DEFAULT_WEIGHTS["temporal"])
        final_score = score + temporal_weighted if score > 0 else 0.0

        breakdown = {
            "candidate_type": "cluster_candidate",
            "candidate_kind": candidate["candidate_kind"],
            "candidate_id": candidate["id"],
            "semantic_raw": semantic_raw,
            "semantic_weighted": semantic_weighted,
            "tag_overlap_raw": tag_overlap_raw,
            "tag_overlap_weighted": tag_overlap_weighted,
            "temporal_raw": temporal_raw,
            "temporal_weighted": temporal_weighted,
            "final_score": final_score,
        }
        return final_score, breakdown

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
        return cur.lastrowid  # type: ignore[return-value]

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
        """Search memory items by embedding similarity, peer, and recency.

        `weights` overrides the semantic/person/temporal scoring weights when
        provided (see `_DEFAULT_WEIGHTS`). `tag_overlap` doesn't apply (memory
        items carry no tags). The `temporal` term is a recency *re-ranking*
        signal, added only to candidates that already have semantic/person
        relevance (see `search_topics`); an item's recency uses its
        `created_at`. `now` is the reference time for the decay (defaults to
        the current UTC time).

        When `tracer` is given, one `retrieval_scoring_summary` event is
        recorded for the whole call — see `search_topics` for the shape.
        """
        w = weights or _DEFAULT_WEIGHTS
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
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
            score = 0.0
            semantic_raw = semantic_weighted = 0.0
            person_weighted = 0.0

            # Embedding similarity
            if query_embedding and row[8]:
                stored_emb = _deserialize_embedding(row[8])
                semantic_raw = _cosine_similarity(query_embedding, stored_emb)
                semantic_weighted = semantic_raw * w.get("semantic", _DEFAULT_WEIGHTS["semantic"])
                score += semantic_weighted

            # Peer filter
            if peer_id and (item.source_peer == peer_id or item.subject_peer == peer_id):
                person_weighted = w.get("person", _DEFAULT_WEIGHTS["person"])
                score += person_weighted

            # NOTE: no tag_overlap — items carry no tags.
            # Temporal recency re-ranks relevant items only (see search_topics).
            # row[9] is the raw created_at string (item.timestamp is typed
            # datetime but actually carries this same string at runtime).
            temporal_raw = _recency_score(row[9], now, _TEMPORAL_HALF_LIFE_DAYS)
            temporal_weighted = temporal_raw * w.get("temporal", _DEFAULT_WEIGHTS["temporal"])
            final_score = score + temporal_weighted if score > 0 else 0.0

            breakdown = {
                "candidate_type": "memory_item",
                "candidate_id": item.id,
                "semantic_raw": semantic_raw,
                "semantic_weighted": semantic_weighted,
                "person_weighted": person_weighted,
                "temporal_raw": temporal_raw,
                "temporal_weighted": temporal_weighted,
                "final_score": final_score,
            }
            scored.append((final_score, item, breakdown))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [(s, i) for s, i, _ in scored if s > 0][:limit]
        included_ids = {i.id for _, i in results}

        if tracer is not None:
            positively_scored = [(s, b) for s, _, b in scored if s > 0]
            top = positively_scored[: limit + _TRACE_TOP_K_PADDING]
            for _, breakdown in top:
                breakdown["included"] = breakdown["candidate_id"] in included_ids
            tracer.record(
                "retrieval_scoring_summary",
                {
                    "candidate_type": "memory_item",
                    "total_scored": len(positively_scored),
                    "included_count": len(included_ids),
                    "top_candidates": [b for _, b in top],
                },
                request_id=request_id,
            )

        return [item for _, item in results]

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

    # ── Messages ───────────────────────────────────────────

    def store_message(
        self,
        turn_id: int,
        role: str,
        content: str,
        peer_id: str,
        topic_id: int | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO messages (turn_id, role, content, peer_id, topic_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (turn_id, role, content, peer_id, topic_id),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_messages_by_topic(self, topic_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, turn_id, role, content, peer_id, created_at "
            "FROM messages WHERE topic_id = ? ORDER BY turn_id, id",
            (topic_id,),
        ).fetchall()
        return [
            {
                "id": r[0],
                "turn_id": r[1],
                "role": r[2],
                "content": r[3],
                "peer_id": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    def get_recent_messages_by_topic(self, topic_id: int, limit: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, turn_id, role, content, peer_id, created_at "
            "FROM messages WHERE topic_id = ? ORDER BY id DESC LIMIT ?",
            (topic_id, limit),
        ).fetchall()
        return [
            {
                "id": r[0],
                "turn_id": r[1],
                "role": r[2],
                "content": r[3],
                "peer_id": r[4],
                "created_at": r[5],
            }
            for r in reversed(rows)
        ]
