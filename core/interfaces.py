"""Eidolon provider interfaces.

All external dependencies (storage, LLM, embedder) are defined as Protocols.
Concrete implementations live in ``providers/`` and are injected via configuration.
The core never imports a specific provider directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from core.models import Fact, MemoryItem


class StorageProvider(Protocol):
    """Abstract storage backend.

    V1: SQLite via ``core/stores/sqlite_store.py``.
    Future: pgvector, MongoDB, or custom backends.
    """

    # ── Lifecycle ──────────────────────────────────────────

    def connect(self) -> None:
        """Open the backend and run any pending setup/migrations."""
        ...

    def close(self) -> None:
        """Release resources opened by ``connect()``."""
        ...

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
        session_id: str | None = None,
    ) -> int:
        """Create a new topic and return its id.

        ``session_id`` tags which conversation session opened this topic —
        see ``get_active_topic`` for why that matters.
        """
        ...

    def get_topic(self, topic_id: int) -> dict | None:
        """Return a topic dict by id, or None if not found."""
        ...

    def close_topic(self, topic_id: int, summary: str) -> None:
        """Mark a topic as closed and set its final summary."""
        ...

    def get_active_topic(self, peer_id: str, session_id: str | None = None) -> int | None:
        """Return the id of the currently open ('active') topic for
        ``peer_id`` *and* ``session_id``, if any.

        A peer can have multiple concurrent sessions (web, Telegram, CLI,
        ...), each continuing its own topic thread — the active topic is
        scoped per (peer_id, session_id), not per peer_id alone, so one
        session's topic detection never clobbers or orphans another
        session's still-open topic. `topics.status` is the source of
        truth for what's open — adapters use this to resume their
        in-memory active-topic tracking after a restart instead of
        assuming there's no open topic.
        """
        ...

    def update_topic_embedding(
        self, topic_id: int, embedding: list[float]
    ) -> None:
        """Store or update the embedding vector for a topic."""
        ...

    def update_topic_tags(self, topic_id: int, tags: list[str]) -> None:
        """Replace a topic's tags with the given list."""
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
        now: datetime | None = None,
    ) -> list[dict]:
        """Search topics by embedding similarity, tag overlap, peer, and recency.

        ``weights`` overrides the per-dimension scoring weights
        (semantic, tag_overlap, person, temporal) when provided. The
        ``temporal`` term is a recency *re-ranking* signal — it only
        re-orders candidates that already have relevance, never surfacing a
        merely-recent one. ``now`` is the reference time for the recency
        decay (defaults to the current UTC time).

        When ``tracer`` is given, one ``retrieval_scoring_summary`` event
        is recorded for the whole call, carrying the top scored candidates
        (with per-dimension breakdown) plus total/included counts.

        Only considers topics not yet assigned to a cluster (see
        ``search_clusters``) — a clustered topic is represented by its
        cluster's consolidated summary instead of competing individually.

        Returns topic dicts ordered by relevance.
        """
        ...

    def get_recent_topics(self, peer_id: str, limit: int) -> list[dict]:
        """Return `peer_id`'s most recently closed, unclustered topics,
        newest first.

        Pure recency, not relevance — used to surface "what we discussed
        last" alongside (and independent of) ``search_topics``'s
        relevance ranking, which has no recency term. Excludes clustered
        topics for the same reason ``search_topics`` does.
        """
        ...

    # ── Topic Clusters ─────────────────────────────────────

    def get_clusters(self, peer_id: str) -> list[dict]:
        """Return every cluster for ``peer_id``, unscored — the full set,
        not just ones relevant to some query. Mirrors ``get_facts``."""
        ...

    def create_cluster(
        self,
        peer_id: str,
        label: str,
        summary: str,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> int:
        """Create a new topic cluster and return its id. ``topic_count``
        starts at 0 — the caller assigns member topics separately via
        ``assign_topic_to_cluster``."""
        ...

    def get_cluster(self, cluster_id: int) -> dict | None:
        """Return a cluster dict by id, or None if not found."""
        ...

    def update_cluster(
        self,
        cluster_id: int,
        summary: str,
        label: str | None = None,
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        """Rewrite a cluster's summary in place (and optionally its label/
        tags/embedding). Bumps ``updated_at``. Mirrors
        ``FactConsolidator``'s update-in-place promise — the summary is
        replaced, not appended to."""
        ...

    def assign_topic_to_cluster(self, topic_id: int, cluster_id: int) -> None:
        """Assign ``topic_id`` to ``cluster_id`` and increment the
        cluster's ``topic_count``. Once assigned, the topic is excluded
        from ``search_topics``/``get_recent_topics``."""
        ...

    def get_topics_by_cluster(self, cluster_id: int) -> list[dict]:
        """Return every topic assigned to ``cluster_id``, oldest first —
        the full member list, for the ``eidolon_inspect_cluster`` tool."""
        ...

    def get_recent_topics_in_cluster(self, cluster_id: int, limit: int) -> list[dict]:
        """Return ``cluster_id``'s most recently closed member topics,
        newest first — the short "newest: ..." tail shown alongside a
        cluster's summary in context injection."""
        ...

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
        """Search topic clusters by embedding similarity, tag overlap, and
        recency — same scoring formula as ``search_topics``, scanning
        ``topic_clusters`` instead of ``topics``.

        Unlike ``search_topics``, ``peer_id`` is a hard SQL filter, not a
        soft scoring bonus — clusters never merge across peers.

        Returns cluster dicts ordered by relevance.
        """
        ...

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
        """Shortlist candidates for ``topic_id`` to join or seed a
        cluster — the bootstrap step run by ``ClusterConsolidator`` at
        topic-close time. Looks up ``topic_id``'s own stored embedding/
        tags internally (set by ``TopicSummarizer`` when it closed).

        Scores BOTH still-unclustered closed topics (excluding
        ``topic_id`` itself) and existing ``topic_clusters`` rows with
        the same formula as ``search_topics``/``search_clusters``, so a
        first-ever match between two never-before-clustered topics is
        possible, not just growth of an existing cluster. Each result
        dict carries ``candidate_kind: "topic" | "cluster"`` alongside
        its normal fields, so the caller knows how to apply an LLM's
        match decision. Hard-filtered to ``peer_id`` in SQL, same
        reasoning as ``search_clusters``.

        Returns candidate dicts ordered by relevance; empty if nothing
        scores above zero (including when ``topic_id`` has no stored
        embedding) — the caller's cue to skip the LLM call entirely for
        a topic with nothing to merge into.
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
        now: datetime | None = None,
    ) -> list[MemoryItem]:
        """Search memory items by embedding similarity, peer, and recency.

        ``weights`` overrides the per-dimension scoring weights when
        provided. ``tag_overlap`` has no effect (memory items carry no
        tags); ``temporal`` is a recency *re-ranking* signal over already-
        relevant items (see ``search_topics``). ``now`` is the reference
        time for the decay (defaults to the current UTC time).

        When ``tracer`` is given, one ``retrieval_scoring_summary`` event
        is recorded for the whole call — see ``search_topics``.
        """
        ...

    def get_memory_items_by_turn(self, turn_id: int) -> list[MemoryItem]:
        """Return all memory items extracted from a given turn, oldest first.

        Joins through ``messages.turn_id`` since memory items only carry a
        ``message_id`` reference, not a turn id of their own. Each item's
        ``embedding`` is populated from storage, so callers that need it
        (e.g. comparing against a freshly embedded query) don't have to
        re-embed the content themselves.
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
        max_tokens: int | None = None,
    ) -> str:
        """Send a completion request and return the response text.

        ``max_tokens`` is the cap on the response. ``None`` (the default)
        means "use the provider's own configured cap" — wired from
        ``ModelConfig.max_tokens`` via the provider factory — so callers
        don't each hard-code a guess. A caller only passes an explicit
        value to deliberately override that cap for one call (e.g. the
        fragmentary extractor, whose JSON output grows with turn richness
        and needs more headroom than ordinary calls).
        """
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
