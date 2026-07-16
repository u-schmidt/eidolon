"""Eidolon provider interfaces.

All external dependencies (storage, LLM, embedder) are defined as Protocols.
Concrete implementations live in ``providers/`` and are injected via configuration.
The core never imports a specific provider directly.
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

    def search_topics(
        self,
        query_embedding: list[float] | None = None,
        tags: list[str] | None = None,
        peer_id: str | None = None,
        limit: int = 5,
    ) -> list[dict]:
        """Search topics by embedding similarity, tag overlap, and/or peer.

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
    ) -> list[MemoryItem]:
        """Search memory items by embedding similarity and/or peer."""
        ...


class LLMProvider(Protocol):
    """Abstract LLM backend.

    V1: wraps Hermes' existing provider via a callable passed by the adapter.
    """

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 100,
    ) -> str:
        """Send a completion request and return the response text."""
        ...


class EmbedderProvider(Protocol):
    """Abstract embedding backend.

    V1: LM Studio (primary) or sentence-transformers (fallback).
    """

    def embed(self, text: str) -> list[float]:
        """Return a vector embedding for the given text."""
        ...
