"""Storage provider interface for Eidolon.

Defines the abstract contract that all storage backends must implement.
V1: SQLite. Future: pgvector, MongoDB, etc.
"""

from __future__ import annotations

from typing import Protocol

from core.models import Fact, MemoryItem


class StorageProvider(Protocol):
    """Abstract storage backend.

    V1: SQLite via ``core/stores/sqlite_store.py``.
    Future: pgvector, MongoDB, or custom backends.
    """

    # ── Facts ──────────────────────────────────────────────

    def get_facts(self, peer_id: str) -> list[Fact]:
        """Return all stored facts for a given peer."""
        ...

    def set_fact(
        self,
        peer_id: str,
        category: str,
        key: str,
        value: str,
        source: str,
    ) -> None:
        """Store or update a single fact."""
        ...

    # ── Topics ─────────────────────────────────────────────

    def create_topic(
        self,
        label: str,
        summary: str = "",
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
        peer_id: str | None = None,
    ) -> int:
        """Create a new topic and return its id."""
        ...

    def get_topic(self, topic_id: int) -> dict | None:
        """Return a topic dict by id, or None if not found."""
        ...

    def close_topic(self, topic_id: int, summary: str) -> None:
        """Mark a topic as closed and set its final summary."""
        ...

    def get_active_topic(self, peer_id: str) -> int | None:
        """Return the id of the peer's currently open ('active') topic, if any.

        `topics.status` is the source of truth for what's open — adapters
        use this to resume their in-memory active-topic tracking after a
        restart instead of assuming there's no open topic.
        """
        ...

    def search_topics(
        self,
        query_embedding: list[float] | None = None,
        tags: list[str] | None = None,
        peer_id: str | None = None,
        limit: int = 5,
        weights: dict[str, float] | None = None,
    ) -> list[dict]:
        """Search topics by embedding similarity, tag overlap, and/or peer.

        ``weights`` overrides the per-dimension scoring weights
        (semantic, tag_overlap, person, temporal) when provided.
        NOTE: ``temporal`` is accepted for forward-compat but not yet
        applied by any backend — there is no recency scoring term yet.

        Returns topic dicts ordered by relevance.
        """
        ...

    # ── Memory Items ───────────────────────────────────────

    def store_memory_item(self, item: MemoryItem) -> int:
        """Persist a single memory item and return its id."""
        ...

    def search_memory_items(
        self,
        query_embedding: list[float] | None = None,
        peer_id: str | None = None,
        limit: int = 5,
        weights: dict[str, float] | None = None,
    ) -> list[MemoryItem]:
        """Search memory items by embedding similarity and/or peer.

        ``weights`` overrides the per-dimension scoring weights when
        provided. NOTE: ``tag_overlap`` (memory items carry no tags) and
        ``temporal`` (no recency scoring yet) are accepted but currently
        have no effect.
        """
        ...
