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

from core.models import Fact, MemoryItem
from core.stores.migrations import migrate


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
        self._db = sqlite3.connect(str(self._path))
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
    ) -> list[dict]:
        """Search topics by embedding similarity, tag overlap, and/or peer.

        Returns topic dicts ordered by relevance (highest first).
        """
        rows = self._conn.execute(
            "SELECT id, label, summary, tags, embedding, status, peer_id, created_at, closed_at "
            "FROM topics WHERE status='closed'"
        ).fetchall()

        results: list[tuple[float, dict]] = []

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

            # Embedding similarity (semantic)
            if query_embedding and row[4]:
                stored_emb = _deserialize_embedding(row[4])
                score += _cosine_similarity(query_embedding, stored_emb) * 0.4

            # Tag overlap
            if tags:
                topic_tags: list[str] = topic["tags"]
                overlap = len(set(tags) & set(topic_tags))
                if overlap > 0:
                    score += (overlap / max(len(tags), 1)) * 0.3

            # Peer filter
            if peer_id and topic["peer_id"] == peer_id:
                score += 0.2

            if score > 0:
                results.append((score, topic))

        results.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in results[:limit]]

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
    ) -> list[MemoryItem]:
        rows = self._conn.execute(
            "SELECT id, content, item_type, source_peer, subject_peer, "
            "attribution_type, topic_id, message_id, embedding, created_at "
            "FROM memory_items"
        ).fetchall()

        results: list[tuple[float, MemoryItem]] = []

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

            # Embedding similarity
            if query_embedding and row[8]:
                stored_emb = _deserialize_embedding(row[8])
                score += _cosine_similarity(query_embedding, stored_emb) * 0.5

            # Peer filter
            if peer_id and (item.source_peer == peer_id or item.subject_peer == peer_id):
                score += 0.3

            if score > 0:
                results.append((score, item))

        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:limit]]

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
