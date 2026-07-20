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

    def delete_fact(self, peer_id: str, category: str, key: str) -> None:
        """Remove a single fact, if present. No-op if it doesn't exist."""
        ...

    def count_user_turns(self, peer_id: str) -> int:
        """Return how many user messages this peer has sent in total.

        Peer-scoped, not session-scoped — used to drive the facts
        consolidation trigger (every N turns), matching how facts
        themselves are peer-scoped.
        """
        ...

    def get_fact_items_since(self, subject_peer: str, since_item_id: int) -> list[MemoryItem]:
        """Return fact-type memory items about ``subject_peer`` with
        id > ``since_item_id``, ordered oldest-first.

        Bounds the window for the next facts-consolidation pass — see
        ``core/consolidation/fact_consolidator.py``.
        """
        ...

    def get_consolidation_checkpoint(self, peer_id: str) -> int:
        """Return the highest memory_item id already incorporated into
        ``peer_id``'s facts profile. 0 if consolidation has never run."""
        ...

    def set_consolidation_checkpoint(self, peer_id: str, last_item_id: int) -> None:
        """Record the highest memory_item id incorporated into
        ``peer_id``'s facts profile as of the most recent consolidation."""
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

    def update_topic_embedding(
        self, topic_id: int, embedding: list[float]
    ) -> None:
        """Store or update the embedding vector for a topic."""
        ...

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

        ``weights`` overrides the per-dimension scoring weights
        (semantic, tag_overlap, person, temporal) when provided.
        NOTE: ``temporal`` is accepted for forward-compat but not yet
        applied by any backend — there is no recency scoring term yet.

        When ``tracer`` is given, one ``retrieval_scoring_summary`` event
        is recorded for the whole call, carrying the top scored candidates
        (with per-dimension breakdown) plus total/included counts.

        Returns topic dicts ordered by relevance.
        """
        ...

    def get_recent_topics(self, peer_id: str, limit: int) -> list[dict]:
        """Return `peer_id`'s most recently closed topics, newest first.

        Pure recency, not relevance — used to surface "what we discussed
        last" alongside (and independent of) ``search_topics``'s
        relevance ranking, which has no recency term.
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
        tracer: Tracer | None = None,
        request_id: str | None = None,
    ) -> list[MemoryItem]:
        """Search memory items by embedding similarity and/or peer.

        ``weights`` overrides the per-dimension scoring weights when
        provided. NOTE: ``tag_overlap`` (memory items carry no tags) and
        ``temporal`` (no recency scoring yet) are accepted but currently
        have no effect.

        When ``tracer`` is given, one ``retrieval_scoring_summary`` event
        is recorded for the whole call — see ``search_topics``.
        """
        ...

    # ── Messages ───────────────────────────────────────────

    def store_message(
        self,
        turn_id: int,
        role: str,
        content: str,
        peer_id: str,
        topic_id: int | None = None,
    ) -> int:
        """Store a single message and return its id."""
        ...

    def get_messages_by_topic(self, topic_id: int) -> list[dict]:
        """Return all messages for a given topic, ordered by turn_id."""
        ...

    def get_recent_messages_by_topic(self, topic_id: int, limit: int) -> list[dict]:
        """Return the most recent `limit` messages for a topic, oldest first."""
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


class Tracer(Protocol):
    """Abstract debug-tracing backend.

    V1: ``NullTracer`` (no-op, default) or ``SqliteTracer`` (writes to a
    separate trace database) via ``core/trace/``. Disabled by default —
    production behavior and performance are unaffected unless a real
    tracer is configured.
    """

    def connect(self) -> None:
        """Open any resources needed to record events."""
        ...

    def close(self) -> None:
        """Release resources opened by ``connect()``."""
        ...

    def record(
        self,
        event_type: str,
        payload: dict,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Record one trace event.

        ``request_id`` groups the events produced by a single top-level
        ``Eidolon`` call (e.g. one ``store_turn`` invocation) — see
        ``core/__init__.py``. ``session_id`` is the adapter-level
        conversation identifier, when available.
        """
        ...
