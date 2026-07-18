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
from pathlib import Path

from core.interfaces import Tracer
from core.models import Fact, MemoryItem
from core.stores.migrations import migrate

# Fallback scoring weights when the caller doesn't supply an explicit
# `weights` override (e.g. from EidolonConfig.retrieval). Shared by both
# search_topics and search_memory_items, matching AssociativeRetriever's
# own defaults so a configured value means the same thing everywhere.
#
# NOTE: "temporal" is accepted in the weights dict for forward-compat but is
# NOT applied anywhere below — there is no recency scoring term implemented
# yet. Don't assume configuring it has any effect until that lands.
_DEFAULT_WEIGHTS = {"semantic": 0.4, "tag_overlap": 0.3, "person": 0.2, "temporal": 0.1}


def _serialize_embedding(vec: list[float]) -> bytes:
    """Pack a float list into a binary BLOB (little-endian f64)."""
    return struct.pack(f"<{len(vec)}d", *vec)


def _deserialize_embedding(blob: bytes) -> list[float]:
    """Unpack a binary BLOB back into a float list."""
    count = len(blob) // 8
    return list(struct.unpack(f"<{count}d", blob))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


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

    # ── Topics ─────────────────────────────────────────────

    def create_topic(
        self,
        label: str,
        summary: str = "",
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
        peer_id: str | None = None,
    ) -> int:
        tags_json = json.dumps(tags or [])
        embed_blob = _serialize_embedding(embedding) if embedding else None
        cur = self._conn.execute(
            "INSERT INTO topics (label, summary, tags, embedding, peer_id) VALUES (?, ?, ?, ?, ?)",
            (label, summary, tags_json, embed_blob, peer_id),
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

    def get_active_topic(self, peer_id: str) -> int | None:
        """Return the id of the peer's most recently opened 'active' topic.

        If more than one is somehow left 'active' (e.g. a crash mid topic
        transition), the most recent wins; older orphans stay open until
        explicitly closed — no automatic cleanup here.
        """
        # `id DESC` breaks ties from CURRENT_TIMESTAMP's second-level
        # resolution — two topics created within the same second would
        # otherwise sort ambiguously.
        row = self._conn.execute(
            "SELECT id FROM topics WHERE peer_id = ? AND status = 'active' "
            "ORDER BY created_at DESC, id DESC LIMIT 1",
            (peer_id,),
        ).fetchone()
        return row[0] if row else None

    def update_topic_embedding(
        self, topic_id: int, embedding: list[float]
    ) -> None:
        embed_blob = _serialize_embedding(embedding)
        self._conn.execute(
            "UPDATE topics SET embedding = ? WHERE id = ?",
            (embed_blob, topic_id),
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
    ) -> list[dict]:
        """Search topics by embedding similarity, tag overlap, and/or peer.

        `weights` overrides the semantic/tag_overlap/person scoring weights
        when provided (see `_DEFAULT_WEIGHTS`). `temporal` is accepted
        but currently unused — no recency term exists yet.

        When `tracer` is given, one `retrieval_candidate_scored` event is
        recorded per candidate with score, once the final top-`limit` cut
        is known.

        Returns topic dicts ordered by relevance (highest first).
        """
        w = weights or _DEFAULT_WEIGHTS
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, embedding, status, peer_id, created_at, closed_at "
            "FROM topics WHERE status='closed'"
        ).fetchall()

        scored: list[tuple[float, dict, dict]] = []

        for row in rows:
            topic = {
                "id": row[0],
                "label": row[1],
                "summary": row[2],
                "tags": json.loads(row[3]),
                "status": row[4],
                "peer_id": row[5],
                "created_at": row[6],
                "closed_at": row[7],
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

            # NOTE: no temporal/recency term yet — w.get("temporal") is unused.

            breakdown = {
                "candidate_type": "topic",
                "candidate_id": topic["id"],
                "semantic_raw": semantic_raw,
                "semantic_weighted": semantic_weighted,
                "tag_overlap_raw": tag_overlap_raw,
                "tag_overlap_weighted": tag_overlap_weighted,
                "person_weighted": person_weighted,
                "final_score": score,
            }
            scored.append((score, topic, breakdown))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [(s, t) for s, t, _ in scored if s > 0][:limit]
        included_ids = {t["id"] for _, t in results}

        if tracer is not None:
            for score, topic, breakdown in scored:
                if score <= 0:
                    continue
                breakdown["included"] = topic["id"] in included_ids
                tracer.record(
                    "retrieval_candidate_scored", breakdown, request_id=request_id,
                )

        return [t for _, t in results]

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
    ) -> list[MemoryItem]:
        """Search memory items by embedding similarity and/or peer.

        `weights` overrides the semantic/person scoring weights when
        provided (see `_DEFAULT_WEIGHTS`). `tag_overlap` doesn't apply
        (memory items carry no tags) and `temporal` is unused — no recency
        term exists yet.

        When `tracer` is given, one `retrieval_candidate_scored` event is
        recorded per candidate, once the final top-`limit` cut is known.
        """
        w = weights or _DEFAULT_WEIGHTS
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

            # NOTE: no tag_overlap (items have no tags) or temporal/recency
            # term yet — w.get("tag_overlap") / w.get("temporal") are unused.

            breakdown = {
                "candidate_type": "memory_item",
                "candidate_id": item.id,
                "semantic_raw": semantic_raw,
                "semantic_weighted": semantic_weighted,
                "person_weighted": person_weighted,
                "final_score": score,
            }
            scored.append((score, item, breakdown))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [(s, i) for s, i, _ in scored if s > 0][:limit]
        included_ids = {i.id for _, i in results}

        if tracer is not None:
            for score, item, breakdown in scored:
                if score <= 0:
                    continue
                breakdown["included"] = item.id in included_ids
                tracer.record(
                    "retrieval_candidate_scored", breakdown, request_id=request_id,
                )

        return [item for _, item in results]

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
