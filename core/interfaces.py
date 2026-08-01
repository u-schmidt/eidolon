"""Eidolon provider interfaces.

All external dependencies (storage, LLM, embedder) are defined as Protocols.
Concrete implementations live in ``providers/`` and are injected via configuration.
The core never imports a specific provider directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from core.models import Fact, MemoryItem, MemoryItemBudget


class StorageProvider(Protocol):
    """Abstract storage backend.

    Implementations: SQLite (``core/stores/sqlite_store.py``, the
    zero-config default for the CLI and tests) and Postgres + pgvector
    (``core/stores/postgres_store.py``, recommended for production —
    see its module docstring for why). Future: MongoDB or other custom
    backends.
    """

    # ── Lifecycle ──────────────────────────────────────────

    def connect(self) -> None:
        """Open the backend and run any pending setup/migrations."""
        ...

    def close(self) -> None:
        """Release resources opened by ``connect()``."""
        ...

    # ── Facts ──────────────────────────────────────────────

    def get_facts(self, workspace_id: str) -> list[Fact]:
        """Return all stored facts for a given workspace."""
        ...

    def get_all_facts(self) -> list[Fact]:
        """Return every stored fact across every workspace, ordered by
        ``workspace_id``/``category``/``key``.

        Bulk, unscoped counterpart to ``get_facts`` — powers dashboard
        views that need to scan the whole facts table (e.g. flagging
        categories outside the configured list) rather than one workspace's
        profile.
        """
        ...

    def set_fact(
        self,
        workspace_id: str,
        category: str,
        key: str,
        value: str,
        source: str,
        importance: str = "medium",
    ) -> None:
        """Store or update a single fact."""
        ...

    def delete_fact(self, workspace_id: str, category: str, key: str) -> None:
        """Remove a single fact, if present. No-op if it doesn't exist."""
        ...

    def count_user_turns(self, workspace_id: str) -> int:
        """Return how many user messages this workspace has sent in total.

        Workspace-scoped, not session-scoped — used to drive the facts
        consolidation trigger (every N turns), matching how facts
        themselves are workspace-scoped.
        """
        ...

    def get_fact_items_since(self, subject_peer: str, since_item_id: int) -> list[MemoryItem]:
        """Return every memory item about ``subject_peer`` (any item_type)
        with id > ``since_item_id``, ordered oldest-first.

        Bounds the window for the next facts-consolidation pass — see
        ``core/consolidation/fact_consolidator.py``. Deliberately not
        filtered to item_type == "fact": that turn-level label is decided
        per-turn with no cross-turn context and is empirically unreliable
        (a concrete fact is often labeled "essence" instead), so filtering
        on it here would permanently exclude mislabeled facts from ever
        reaching the profile. FactConsolidator's own LLM pass already
        judges durability from each item's content, so it doesn't need
        item_type to have already done that filtering for it.
        """
        ...

    def get_consolidation_checkpoint(self, workspace_id: str) -> int:
        """Return the highest memory_item id already incorporated into
        ``workspace_id``'s facts profile. 0 if consolidation has never run."""
        ...

    def set_consolidation_checkpoint(self, workspace_id: str, last_item_id: int) -> None:
        """Record the highest memory_item id incorporated into
        ``workspace_id``'s facts profile as of the most recent consolidation."""
        ...

    # ── Entity Profiles (per-named-entity fact cards) ───────
    # See core/consolidation/entity_profile_consolidator.py. Mirrors the
    # Facts/Facts Consolidation sections above, keyed by an extra
    # entity_name dimension instead of (or alongside) workspace_id.

    def get_topics_since(self, workspace_id: str, since_topic_id: int) -> list[dict]:
        """Return ``workspace_id``'s closed topics with id > ``since_topic_id``,
        oldest first — id/label/summary/named_entities.

        Workspace-scoped, id-cursored counterpart to
        ``get_fact_items_since``, feeding
        ``EntityProfileConsolidator``'s mention-scanning pass. Not
        ``get_closed_topics_since`` (time-windowed, cross-workspace, and
        doesn't return named_entities — built for the dashboard, not
        this).
        """
        ...

    def increment_entity_mentions(self, workspace_id: str, entity_names: list[str]) -> None:
        """Upsert an ``entity_profiles`` row for each name, incrementing
        ``mention_count`` by 1. No-op for an empty list."""
        ...

    def materialize_entities(self, workspace_id: str, threshold: int) -> list[str]:
        """Mark every not-yet-materialized entity at/above ``threshold``
        mentions as materialized (sets ``materialized_at``), returning
        the names just materialized."""
        ...

    def get_materialized_entity_names(self, workspace_id: str) -> list[str]:
        """Return every materialized entity name for ``workspace_id`` —
        the small candidate list retrieval regex-matches a turn's
        message against before pulling any cards."""
        ...

    def get_topics_mentioning_entity(
        self, workspace_id: str, entity_name: str, since_topic_id: int
    ) -> list[dict]:
        """Return closed topics for ``workspace_id`` whose
        ``named_entities`` includes ``entity_name``, with id >
        ``since_topic_id``, oldest first — id/label/summary.
        ``since_topic_id=0`` (an entity's default checkpoint) returns its
        entire mention history, which matters the first time an entity
        materializes — its card shouldn't only reflect the batch of
        topics that happened to cross the threshold."""
        ...

    def get_entity_facts(self, workspace_id: str, entity_names: list[str]) -> list[Fact]:
        """Return every stored entity-profile fact for the given entity
        names, with ``Fact.entity_name`` set on each. Empty list in,
        empty list out — no query issued."""
        ...

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
        """Store or update a single entity-profile fact."""
        ...

    def delete_entity_fact(self, workspace_id: str, entity_name: str, category: str, key: str) -> None:
        """Remove a single entity-profile fact, if present. No-op if it doesn't exist."""
        ...

    def list_entity_profiles(self, workspace_id: str | None = None) -> list[dict]:
        """Return every ``entity_profiles`` row, across every workspace or
        one workspace's if given, most-mentioned first — workspace_id/
        entity_name/mention_count/materialized_at/
        last_consolidated_topic_id/fact_count (a count against
        ``entity_facts``, not a stored column).

        Bulk, optionally-scoped read mirroring ``list_clusters``'s
        relationship to ``get_clusters`` — powers the dashboard's entity-
        profiles browse page. Includes not-yet-materialized entities
        (``materialized_at`` is ``None``) deliberately, so "why hasn't
        this name gotten a card yet" is visible, not just finished cards.
        """
        ...

    def get_entity_scan_checkpoint(self, workspace_id: str) -> int:
        """Return the highest topic id already scanned for named-entity
        mentions in ``workspace_id``. 0 if never scanned."""
        ...

    def set_entity_scan_checkpoint(self, workspace_id: str, last_topic_id: int) -> None:
        """Record the highest topic id already scanned for named-entity
        mentions as of the most recent entity-profile pass."""
        ...

    def get_entity_consolidation_checkpoint(self, workspace_id: str, entity_name: str) -> int:
        """Return the highest topic id already incorporated into this
        entity's profile. 0 if never consolidated — which, for a just-
        materialized entity, means its next pass pulls its whole
        history via ``get_topics_mentioning_entity``."""
        ...

    def set_entity_consolidation_checkpoint(
        self, workspace_id: str, entity_name: str, last_topic_id: int
    ) -> None:
        """Record the highest topic id incorporated into this entity's
        profile as of the most recent consolidation pass."""
        ...

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

    def get_active_topic(self, workspace_id: str, session_id: str | None = None) -> int | None:
        """Return the id of the currently open ('active') topic for
        ``workspace_id`` *and* ``session_id``, if any.

        A workspace can have multiple concurrent sessions (web, Telegram, CLI,
        ...), each continuing its own topic thread — the active topic is
        scoped per (workspace_id, session_id), not per workspace_id alone, so one
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

    def update_topic_entities(self, topic_id: int, named_entities: list[str]) -> None:
        """Replace a topic's named-entity list with the given list.

        Separate from ``update_topic_tags`` deliberately — ``tags`` is
        capped for retrieval relevance ("3-5 key tags"), while
        ``named_entities`` is uncapped, so every person/place/organization
        TopicSummarizer found survives here even when only some of them
        also get folded into ``tags``. See core/stores/migrations.py's
        version 12 for the full rationale.
        """
        ...

    def update_topic_label(self, topic_id: int, label: str) -> None:
        """Replace a topic's label. Mirrors update_topic_tags."""
        ...

    def count_messages_by_topic(self, topic_id: int) -> int:
        """Return how many messages a topic has accumulated so far.

        Lightweight COUNT query — unlike get_messages_by_topic, doesn't
        load message content. Drives the mid-conversation topic-relabel
        threshold check in Eidolon.store_turn (see
        core/summarization/topic_relabeler.py) without paying for a
        full transcript load on every turn of every still-pending topic.
        """
        ...

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

        ``workspace_id``, when given, is a hard filter, not just a scoring
        bonus — only that workspace's topics are ever scanned, both to bound
        the scan cost and so a different workspace's topic can never surface
        in these results regardless of how well it scores. ``workspace_id=None``
        searches every workspace.

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

        ``min_message_count`` soft-hides "mini-topics" (see
        ``core/config.py``'s ``RetrievalConfig.min_topic_messages``) —
        only topics with more than this many messages are returned.
        -1 (default) disables the filter, since every real topic has
        ``message_count >= 0``. Applied before ranking, not as a
        post-fetch discard, so it never eats into the result count.

        Returns topic dicts ordered by relevance.
        """
        ...

    def get_stale_active_topics(self, cutoff: datetime, workspace_id: str | None = None) -> list[dict]:
        """Return 'active' topics whose last message predates `cutoff`,
        oldest first — candidates for graceful cleanup (see
        `Eidolon.close_stale_topics`) when a topic was never properly
        closed (a crashed session, an adapter that stopped calling
        `store_turn`).

        Each dict carries a `last_activity` timestamp alongside the usual
        topic fields — the topic's own `created_at` when it has no
        messages yet. `workspace_id=None` scans every workspace's active topics.
        """
        ...

    def get_recent_topics(self, workspace_id: str, limit: int, min_message_count: int = -1) -> list[dict]:
        """Return `workspace_id`'s most recently closed, unclustered topics,
        newest first.

        Pure recency, not relevance — used to surface "what we discussed
        last" alongside (and independent of) ``search_topics``'s
        relevance ranking, which has no recency term. Excludes clustered
        topics for the same reason ``search_topics`` does.

        ``min_message_count`` soft-hides "mini-topics" — see
        ``search_topics``'s docstring for the same parameter.
        """
        ...

    def get_closed_topics_since(self, since: datetime) -> list[dict]:
        """Return every topic closed at/after ``since``, across every
        workspace, oldest first — id/workspace_id/label/created_at/closed_at/
        message_count.

        Bulk, time-windowed, cross-workspace counterpart to
        ``get_recent_topics``/``get_stale_active_topics`` (both scoped
        differently) — powers the topic-lifecycle dashboard view; not
        used by any core orchestration path.
        """
        ...

    def list_topics(self, status: str | None = None, workspace_id: str | None = None) -> list[dict]:
        """Return every topic across every workspace, newest first, optionally
        filtered by ``status`` (``'active'``/``'closed'``) and/or
        ``workspace_id``.

        Bulk, unscoped, no time constraint — none of the existing
        topic-read methods cover this: ``get_recent_topics`` is
        recency-limited and unclustered-only, ``get_stale_active_topics``
        is active-only, ``get_closed_topics_since`` is closed-and-time-
        windowed. Powers the dashboard's topic browse page.
        """
        ...

    # ── Topic Clusters ─────────────────────────────────────

    def get_clusters(self, workspace_id: str) -> list[dict]:
        """Return every cluster for ``workspace_id``, unscored — the full set,
        not just ones relevant to some query. Mirrors ``get_facts``."""
        ...

    def list_clusters(self, workspace_id: str | None = None) -> list[dict]:
        """Return every cluster, across every workspace or one workspace's if
        given, newest first.

        Bulk, optionally-scoped counterpart to ``get_clusters`` (always
        ``workspace_id``-required) — mirrors ``get_all_facts``'s relationship
        to ``get_facts``. Powers the dashboard's cluster browse page.
        """
        ...

    def create_cluster(
        self,
        workspace_id: str,
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
        workspace_id: str | None = None,
        limit: int = 5,
        weights: dict[str, float] | None = None,
        tracer: Tracer | None = None,
        request_id: str | None = None,
        now: datetime | None = None,
    ) -> list[dict]:
        """Search topic clusters by embedding similarity, tag overlap, and
        recency — same scoring formula as ``search_topics``, scanning
        ``topic_clusters`` instead of ``topics``.

        Unlike ``search_topics``, ``workspace_id`` is a hard SQL filter, not a
        soft scoring bonus — clusters never merge across workspaces.

        Returns cluster dicts ordered by relevance.
        """
        ...

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
        match decision. Hard-filtered to ``workspace_id`` in SQL, same
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

    def get_memory_item(self, item_id: int) -> dict | None:
        """Return a memory item dict by id, or None if not found.

        Mirrors ``get_topic``/``get_cluster`` — a plain dict, not a
        ``MemoryItem``, since this is a read-path lookup (e.g. resolving a
        retrieval candidate to its content), not a write-path object.
        """
        ...

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

        ``workspace_id``, when given, is a hard filter joined through
        ``messages`` (memory items carry no ``workspace_id`` of their own, only
        a ``message_id`` — ``messages.workspace_id`` is the real identifier).
        ``workspace_id=None`` searches every workspace. There is no ``person``
        scoring dimension here (unlike ``search_topics``) — workspace-scoping
        is a hard prerequisite via the join, not a ranking signal.

        ``budget`` (defaults to ``MemoryItemBudget()``) replaces a flat
        result-count limit with a per-item-type reserved budget — see
        ``MemoryItemBudget`` and ``RetrievalScorer.finalize_tiered``.
        ``fact``/``surprise``/``feeling`` each fill from their own
        embedding-similarity-ranked candidate pool; ``surprise``/
        ``feeling`` additionally get an un-prefiltered fetch scoped to
        just those two types (bypassing the vector-index KNN prefilter),
        since they're rare enough that a flat top-K-by-similarity prefilter
        could otherwise exclude every candidate of that type before
        type-aware selection ever runs.

        ``weights`` overrides the per-dimension scoring weights when
        provided. ``tag_overlap`` has no effect (memory items carry no
        tags); ``temporal`` is a recency *re-ranking* signal over already-
        relevant items (see ``search_topics``). ``now`` is the reference
        time for the decay (defaults to the current UTC time).

        When ``tracer`` is given, one ``retrieval_scoring_summary`` event
        is recorded for the whole call — see ``RetrievalScorer.finalize_tiered``.
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

    def get_memory_items_by_topic(self, topic_id: int) -> list[MemoryItem]:
        """Return every memory item extracted for ``topic_id``, oldest first.

        No join needed — memory items carry their own ``topic_id`` column,
        set at extraction time. Feeds both ``TopicSummarizer``'s primary
        summarization input and the ``eidolon_inspect_topic`` tool.
        """
        ...

    def list_memory_items(
        self, item_type: str | None = None, topic_id: int | None = None, q: str | None = None,
    ) -> list[MemoryItem]:
        """Return every memory item across every workspace/topic, newest
        first, optionally filtered by ``item_type``/``topic_id``/``q``.

        ``q`` is a plain case-insensitive substring match against
        ``content`` — a dashboard search box, not embedding similarity
        (that's ``search_memory_items``). Bulk, unscoped counterpart to
        ``get_memory_items_by_topic`` (topic-scoped) and
        ``search_memory_items`` (relevance-ranked, not a plain listing).
        Powers the dashboard's memory-item browse page.
        """
        ...

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
        """Store a single message and return its id.

        ``request_id`` correlates this message back to the turn that
        produced it — the same id threaded through every step of
        ``Eidolon.store_turn`` (see core/__init__.py) — so a turn's
        actual message text can be looked up without the debug-gated
        trace (see ``get_messages_by_request_id``).
        """
        ...

    def get_messages_by_topic(self, topic_id: int) -> list[dict]:
        """Return all messages for a given topic, ordered by turn_id."""
        ...

    def get_recent_messages_by_topic(self, topic_id: int, limit: int) -> list[dict]:
        """Return the most recent `limit` messages for a topic, oldest first."""
        ...

    def get_messages_by_request_id(self, request_id: str) -> list[dict]:
        """Return every message stored with this ``request_id`` (typically
        the user + AI message pair from one turn), ordered by id."""
        ...

    def get_recent_turns(self, limit: int, workspace_id: str | None = None) -> list[dict]:
        """Return up to ``limit`` recent turns, most recent first — one
        row per ``role='user'`` message that has a ``request_id``, with
        its ``content`` as a preview. Always-on (no debug trace needed):
        the data source for the turn-detail dashboard's picker list
        (``tools/dashboard/turn_reader.py``)."""
        ...

    def count_recent_turns(self, workspace_id: str | None = None) -> int:
        """Return the total count of turns matching ``get_recent_turns``'s
        same criteria (``role='user'`` messages with a ``request_id``,
        optionally scoped to one workspace) — without paging through the rows
        themselves. Powers the dashboard landing page's turn count."""
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


class TraceReader(Protocol):
    """Read-only counterpart to ``Tracer`` — fetches back rich trace
    events previously written by a ``Tracer`` implementation, keyed by
    ``request_id``.

    Kept separate from ``Tracer`` itself (rather than adding read
    methods to it) so the write-side interface — on the hot path of
    every instrumented class — stays single-purpose. Reading is always
    safe regardless of ``debug.enabled``: historical rows may exist from
    a period when debug tracing was on even if it's off now.
    Implementations: ``core/trace/``.
    """

    def connect(self) -> None:
        """Open any resources needed to read events."""
        ...

    def close(self) -> None:
        """Release resources opened by ``connect()``."""
        ...

    def events_for_request(self, request_id: str) -> list[dict]:
        """Return every trace event recorded for ``request_id``, oldest
        first. Each dict has ``event_type``/``payload``/``duration_ms``/
        ``session_id``/``created_at`` keys. Returns ``[]`` if no rich
        trace data exists for this request (e.g. debug was off, or the
        backing store has never been created) — never raises for that
        reason."""
        ...

    def events_of_type_since(self, event_type: str, since: datetime) -> list[dict]:
        """Return every trace event of ``event_type`` with ``created_at
        >= since``, oldest first. Bulk/time-windowed counterpart to
        ``events_for_request`` — powers aggregate dashboard views (e.g.
        retrieval-quality analysis) rather than single-turn replay. Each
        dict has ``payload``/``duration_ms``/``session_id``/
        ``request_id``/``created_at`` keys."""
        ...


class MetricsRecorder(Protocol):
    """Abstract always-on operational-metrics backend.

    Unlike ``Tracer`` (rich, verbose, gated by ``debug.enabled`` — off by
    default), this is never gated: every instrumented class always
    records duration/error data here, typed and cheap, so health/decay
    reporting works whether or not debug tracing is turned on. Backed by
    three tables living in the same database as production storage (see
    ``core/stores/migrations.py``/``postgres_migrations.py``) —
    ``events`` (errors/warnings), ``llm_calls`` (duration per LLM call
    type), ``embedding_calls`` (duration per embedding purpose).
    Implementations: ``core/metrics/``.
    """

    def connect(self) -> None:
        """Open any resources needed to record metrics."""
        ...

    def close(self) -> None:
        """Release resources opened by ``connect()``."""
        ...

    def record_event(
        self,
        severity: str,
        step: str,
        message: str,
        *,
        workspace_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Record one error/warning. ``severity`` is ``"error"`` or
        ``"warning"``; ``step`` names the process that raised it (e.g.
        ``"fact_consolidation"``), matching the corresponding ``Tracer``
        event type where one exists."""
        ...

    def record_llm_call(
        self,
        call_type: str,
        duration_ms: float,
        succeeded: bool,
        *,
        workspace_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Record one LLM completion call. ``call_type`` is one of
        ``topic_detection``, ``extraction``, ``topic_summarization``,
        ``topic_relabel``, ``fact_consolidation``, ``cluster_consolidation``
        — matching the corresponding ``Tracer`` event type."""
        ...

    def record_embedding_call(
        self,
        purpose: str,
        duration_ms: float,
        dimension: int,
        succeeded: bool,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        workspace_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Record one embedding call. ``purpose`` is one of
        ``retrieval_query``, ``memory_item``, ``topic_summary``,
        ``cluster_summary``. ``entity_type``/``entity_id`` identify what
        got embedded (e.g. ``"topic"``/``42``) — nullable and polymorphic,
        not a hard foreign key, since the id space varies by purpose."""
        ...


class MetricsReader(Protocol):
    """Read-only counterpart to ``MetricsRecorder`` — fetches back
    ``events``/``llm_calls``/``embedding_calls`` rows, keyed by
    ``request_id``, or listed by recency for a picker UI. Kept separate
    from ``MetricsRecorder`` for the same reason ``TraceReader`` is kept
    separate from ``Tracer`` — the write side stays single-purpose.
    Implementations: ``core/metrics/``.
    """

    def connect(self) -> None:
        """Open any resources needed to read metrics."""
        ...

    def close(self) -> None:
        """Release resources opened by ``connect()``."""
        ...

    def events_for_request(self, request_id: str) -> list[dict]:
        """Return every ``events`` row for ``request_id``, oldest first."""
        ...

    def llm_calls_for_request(self, request_id: str) -> list[dict]:
        """Return every ``llm_calls`` row for ``request_id``, oldest first."""
        ...

    def embedding_calls_for_request(self, request_id: str) -> list[dict]:
        """Return every ``embedding_calls`` row for ``request_id``, oldest first."""
        ...

    def events_since(self, since: datetime) -> list[dict]:
        """Return every ``events`` row with ``created_at >= since``, oldest first.

        Bulk/time-windowed counterpart to ``events_for_request`` — the
        data-access layer behind aggregate dashboard views (error rates,
        top error sources) rather than a single turn's replay.
        """
        ...

    def llm_calls_since(self, since: datetime) -> list[dict]:
        """Return every ``llm_calls`` row with ``created_at >= since``, oldest first."""
        ...

    def embedding_calls_since(self, since: datetime) -> list[dict]:
        """Return every ``embedding_calls`` row with ``created_at >= since``, oldest first."""
        ...
