"""Eidolon core facade.

The main entry point for the Eidolon topic memory system.
Wires together all core components and exposes the public API.

Usage:
    from core import Eidolon

    eidolon = Eidolon(storage=store, llm=llm, embedder=embedder)

    # Before LLM responds (sync):
    bundle = eidolon.prepare_context(
        message="Lass uns über Bello sprechen",
        peer_id="user",
        environment=env,
        active_topic_id=42,
    )

    # After LLM responds (async):
    eidolon.store_turn(
        user_message="Lass uns über Bello sprechen",
        ai_response="Gerne! Was möchtest du wissen?",
        peer_id="user",
        environment=env,
        old_topic_id=42,
        new_topic_label="Bello Features",
    )
"""

from __future__ import annotations

import time
import uuid

from core.consolidation.fact_consolidator import FactConsolidator
from core.detection.topic_detector import TopicDetector
from core.extraction.fragmentary_extractor import FragmentaryExtractor
from core.interfaces import EmbedderProvider, LLMProvider, StorageProvider, Tracer
from core.models import ContextBundle, EnvironmentSnapshot, MemoryItem, TopicChange
from core.retrieval.associative_retriever import AssociativeRetriever
from core.summarization.topic_summarizer import TopicSummarizer
from core.trace.null_tracer import NullTracer


def _new_request_id() -> str:
    """Short id grouping the trace events of one top-level Eidolon call.

    Independent of ``messages.turn_id`` (which adapters currently don't
    populate consistently) — this is purely for correlating this call's
    own sub-steps (detection, retrieval, extraction, ...) in a trace.
    """
    return uuid.uuid4().hex[:8]


# How many of a topic's most recent messages to show the detector alongside
# its (possibly still-generic, since it's only ever refreshed on close)
# label — real, evolving content instead of a frozen string. ~2 exchanges.
_TOPIC_DETECTION_MESSAGE_WINDOW = 4


class Eidolon:
    """Eidolon topic memory system — main facade.

    Parameters
    ----------
    storage : StorageProvider
        Backend for persisting facts, topics, and memory items.
    llm : LLMProvider
        Provider for LLM calls (detection, extraction, summarization).
    embedder : EmbedderProvider
        Provider for generating text embeddings.
    language : str
        Language code for stop word filtering (default: "en").
        Prompts to the LLM are always in English regardless of this setting.
    retrieval_weights : dict | None
        Optional weight overrides for associative retrieval dimensions.
    retrieval_limits : dict | None
        Optional result-count overrides for associative retrieval — see
        ``AssociativeRetriever``.
    facts_consolidation_interval : int
        Reconcile the durable-facts profile every N user turns (see
        ``store_turn``'s step 7 and ``FactConsolidator``).
    facts_target_max : int
        Soft guidance passed to ``FactConsolidator`` for the target
        profile size — not enforced in code, only steered via its prompt.
    tracer : Tracer | None
        Debug tracer recording per-step prompts/responses/scores/durations
        and otherwise-silent failures. Defaults to a no-op tracer — off
        unless explicitly configured (see ``core/trace/``).
    """

    def __init__(
        self,
        storage: StorageProvider,
        llm: LLMProvider,
        embedder: EmbedderProvider,
        language: str = "en",
        retrieval_weights: dict[str, float] | None = None,
        retrieval_limits: dict[str, int] | None = None,
        facts_consolidation_interval: int = 20,
        facts_target_max: int = 100,
        tracer: Tracer | None = None,
    ) -> None:
        self._storage = storage
        self._tracer = tracer or NullTracer()
        self._detector = TopicDetector(llm, tracer=self._tracer)
        self._retriever = AssociativeRetriever(
            storage, embedder, retrieval_weights, language, retrieval_limits, tracer=self._tracer
        )
        self._extractor = FragmentaryExtractor(llm, language, tracer=self._tracer)
        self._summarizer = TopicSummarizer(storage, llm, embedder, tracer=self._tracer)
        self._fact_consolidator = FactConsolidator(
            storage, llm, target_max=facts_target_max, tracer=self._tracer
        )
        self._consolidation_interval = facts_consolidation_interval
        self._embedder = embedder
        self._language = language

    # ── Public API ─────────────────────────────────────────

    def prepare_context(
        self,
        message: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        active_topic_id: int | None = None,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> ContextBundle:
        """Prepare context for an LLM call.

        Combines topic detection and context retrieval into a single sync call.
        Called BEFORE the LLM responds.

        Parameters
        ----------
        message : str
            The user's message.
        peer_id : str
            Who is speaking (e.g. "user", "ai:neuromancer").
        environment : EnvironmentSnapshot
            Current situational context.
        active_topic_id : int | None
            The current active topic ID from the adapter's state, if any.
        session_id : str | None
            Adapter-level conversation identifier, for trace filtering.
        request_id : str | None
            Caller-supplied id to correlate this call's trace events with
            others from the same turn (e.g. a matching store_turn() call
            made later). Defaults to a fresh id if not given.

        Returns
        -------
        ContextBundle
            Facts, environment, topic info, related topics, and memory items
            for prompt injection.
        """
        request_id = request_id or _new_request_id()

        # 1. Detect topic change
        current_topic_summary = self._get_topic_summary(active_topic_id)
        recent_messages = self._recent_messages_for_topic(active_topic_id)
        topic_change = self._detector.detect(
            message, current_topic_summary, recent_messages,
            request_id=request_id, session_id=session_id,
        )

        # 2. Retrieve context
        is_new_topic = topic_change.type == "NEW_TOPIC"
        bundle = self._retriever.retrieve(
            message=message,
            peer_id=peer_id,
            environment=environment,
            topic_label=topic_change.label,
            is_new_topic=is_new_topic,
            active_topic_id=active_topic_id,
            request_id=request_id,
            session_id=session_id,
        )

        return bundle

    def detect_topic(
        self,
        message: str,
        active_topic_id: int | None = None,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> TopicChange:
        """Detect whether a message continues or changes the topic.

        Parameters
        ----------
        message : str
            The user's message.
        active_topic_id : int | None
            The current active topic ID, if any.
        session_id : str | None
            Adapter-level conversation identifier, for trace filtering.
        request_id : str | None
            Caller-supplied id to correlate this call's trace events with
            others from the same turn. Defaults to a fresh id if not given.

        Returns
        -------
        TopicChange
            CONTINUE or NEW_TOPIC with a label.
        """
        current_topic_summary = self._get_topic_summary(active_topic_id)
        recent_messages = self._recent_messages_for_topic(active_topic_id)
        return self._detector.detect(
            message, current_topic_summary, recent_messages,
            request_id=request_id or _new_request_id(), session_id=session_id,
        )

    def get_active_topic(self, peer_id: str, session_id: str | None = None) -> int | None:
        """Look up the currently open topic for this peer *and* session, if any.

        Adapters call this whenever a session is first seen (not just once
        at startup) to resume its own in-memory active-topic tracking —
        otherwise a restart, or a session that was never seen before,
        silently forgets about an open topic, leaving it open (and
        unretrievable, since search only considers closed topics) forever.
        Scoped per session so a peer's concurrent sessions (web, Telegram,
        CLI, ...) each track their own topic thread independently.

        Parameters
        ----------
        peer_id : str
            Who the active topic belongs to.
        session_id : str | None
            Which session's topic thread to resume. None matches only
            topics created with no session (legacy rows, single-session
            callers) — not a wildcard across all of the peer's sessions.

        Returns
        -------
        int | None
            The open topic's id, or None if there's no open topic for
            this (peer_id, session_id).
        """
        return self._storage.get_active_topic(peer_id, session_id)

    def retrieve_context(
        self,
        message: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        topic_label: str | None = None,
        is_new_topic: bool = False,
        active_topic_id: int | None = None,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> ContextBundle:
        """Retrieve relevant context for a message.

        Parameters
        ----------
        message : str
            The user's message.
        peer_id : str
            Who is speaking.
        environment : EnvironmentSnapshot
            Current situational context.
        topic_label : str | None
            Detected topic label, if any.
        is_new_topic : bool
            True if this is a NEW_TOPIC, False if CONTINUE.
        active_topic_id : int | None
            The topic being continued, if this is a CONTINUE turn — used
            to keep ``current_topic_label`` populated for the whole
            topic's lifetime (see ``AssociativeRetriever.retrieve``).
        session_id : str | None
            Adapter-level conversation identifier, for trace filtering.
        request_id : str | None
            Caller-supplied id to correlate this call's trace events with
            others from the same turn (e.g. a preceding detect_topic()
            call). Defaults to a fresh id if not given.

        Returns
        -------
        ContextBundle
            Facts, related/recent topics, and memory items.
        """
        return self._retriever.retrieve(
            message=message,
            peer_id=peer_id,
            environment=environment,
            topic_label=topic_label,
            is_new_topic=is_new_topic,
            active_topic_id=active_topic_id,
            request_id=request_id or _new_request_id(),
            session_id=session_id,
        )

    def store_turn(
        self,
        user_message: str,
        ai_response: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        turn_id: int | None = None,
        old_topic_id: int | None = None,
        new_topic_label: str | None = None,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> int | None:
        """Store a conversation turn after the LLM has responded.

        Called AFTER the LLM responds (async, does not block the turn).

        Performs:
        1. Close old topic (if topic changed)
        2. Create new topic (if topic changed)
        3. Store user message and AI response as messages
        4. Extract memory fragments from the turn
        5. Store memory items with perspective attribution + message reference
        6. Generate embedding for the new topic

        Parameters
        ----------
        user_message : str
            The user's message from this turn.
        ai_response : str
            The AI's response from this turn.
        peer_id : str
            Who is speaking (the user).
        environment : EnvironmentSnapshot
            The environment context at the time of this turn.
        turn_id : int | None
            Turn counter for grouping user+ai messages. Managed by the adapter.
        old_topic_id : int | None
            The topic ID to close (if a topic change occurred).
        new_topic_label : str | None
            The label for the new topic (if a topic change occurred).
        session_id : str | None
            Adapter-level conversation identifier, for trace filtering.
        request_id : str | None
            Caller-supplied id to correlate this call's trace events with
            others from the same turn (e.g. a preceding prefetch's
            detect_topic()/retrieve_context() calls). Defaults to a fresh
            id if not given. Unrelated to `turn_id` above.

        Returns
        -------
        int | None
            The new topic ID if a topic was created, None otherwise.
        """
        request_id = request_id or _new_request_id()
        turn_started = time.perf_counter()

        # 1. Close old topic if topic changed — and generate summary + tags.
        # _summarize_topic() closes the topic itself once it has a real
        # summary; the placeholder here is only a fallback for when it
        # couldn't (no messages, LLM failure, unparseable response), so a
        # topic is never left stuck 'active' forever.
        if old_topic_id is not None and new_topic_label is not None:
            summarized = self._summarizer.summarize(old_topic_id, request_id=request_id, session_id=session_id)
            if not summarized:
                self._storage.close_topic(old_topic_id, f"Continued to: {new_topic_label}")

        # 2. Create new topic if topic changed
        new_topic_id: int | None = None
        if new_topic_label is not None:
            new_topic_id = self._storage.create_topic(
                label=new_topic_label,
                peer_id=peer_id,
                session_id=session_id,
            )

        # 3. Store messages
        resolved_topic_id = new_topic_id or old_topic_id
        tid = turn_id or 0

        user_msg_id = self._storage.store_message(
            turn_id=tid, role="user", content=user_message,
            peer_id=peer_id, topic_id=resolved_topic_id,
        )
        ai_msg_id = self._storage.store_message(
            turn_id=tid, role="ai", content=ai_response,
            peer_id="ai", topic_id=resolved_topic_id,
        )

        # 4. Generate embedding for the new topic
        if new_topic_id is not None and new_topic_label:
            try:
                embed_started = time.perf_counter()
                topic_embedding = self._embedder.embed(new_topic_label)
                embed_duration_ms = (time.perf_counter() - embed_started) * 1000
                self._storage.update_topic_embedding(new_topic_id, topic_embedding)
                self._tracer.record(
                    "embedding",
                    {"purpose": "topic_label", "text": new_topic_label, "dimension": len(topic_embedding)},
                    request_id=request_id, session_id=session_id, duration_ms=embed_duration_ms,
                )
            except Exception as exc:
                self._tracer.record(
                    "error",
                    {"step": "topic_label_embedding", "exception": repr(exc)},
                    request_id=request_id, session_id=session_id,
                )
                pass  # Embedding failure is non-critical

        # 5. Extract memory fragments
        items = self._extractor.extract(
            user_message=user_message,
            ai_response=ai_response,
            source_peer=peer_id,
            topic_label=new_topic_label,
            environment=environment,
            request_id=request_id,
            session_id=session_id,
        )

        # 6. Store memory items with topic + message association + embeddings
        fact_items: list[MemoryItem] = []
        for item in items:
            if resolved_topic_id is not None:
                item.topic_id = resolved_topic_id
            # Link items to the message they were extracted from
            if item.source_peer == "ai":
                item.message_id = ai_msg_id
            else:
                item.message_id = user_msg_id
            # Generate embedding for the item content
            if item.content:
                try:
                    embed_started = time.perf_counter()
                    item.embedding = self._embedder.embed(item.content)
                    embed_duration_ms = (time.perf_counter() - embed_started) * 1000
                    self._tracer.record(
                        "embedding",
                        {"purpose": "memory_item", "text": item.content, "dimension": len(item.embedding)},
                        request_id=request_id, session_id=session_id, duration_ms=embed_duration_ms,
                    )
                except Exception as exc:
                    self._tracer.record(
                        "error",
                        {"step": "memory_item_embedding", "exception": repr(exc)},
                        request_id=request_id, session_id=session_id,
                    )
                    pass  # Embedding failure is non-critical
            self._storage.store_memory_item(item)
            # Collect fact-type items — no longer converted to facts
            # immediately (see step 7); kept here only for the
            # turn_complete trace field below.
            if item.item_type == "fact":
                fact_items.append(item)

        # 7. Periodically reconcile the durable facts profile — a batch
        # review every N turns instead of a per-turn conversion, so
        # updates replace stale values in place and the LLM judges what's
        # actually durable rather than everything fact-shaped becoming a
        # permanent row. See FactConsolidator.
        if self._storage.count_user_turns(peer_id) % self._consolidation_interval == 0:
            self.consolidate_facts(peer_id, request_id=request_id, session_id=session_id)

        turn_duration_ms = (time.perf_counter() - turn_started) * 1000
        self._tracer.record(
            "turn_complete",
            {
                "user_message": user_message,
                "ai_response": ai_response,
                "peer_id": peer_id,
                "old_topic_id": old_topic_id,
                "new_topic_id": new_topic_id,
                "new_topic_label": new_topic_label,
                "items_extracted": len(items),
                "fact_items": len(fact_items),
            },
            request_id=request_id, session_id=session_id, duration_ms=turn_duration_ms,
        )

        return new_topic_id

    def consolidate_facts(
        self,
        peer_id: str,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Reconcile `peer_id`'s durable-facts profile now.

        Normally triggered automatically from `store_turn` every N
        turns (see step 7); exposed publicly so adapters/tools can also
        trigger it manually (e.g. a `/consolidate` CLI command) without
        waiting for the interval. No-ops if nothing new has accumulated
        since the last successful pass — see `FactConsolidator`.
        """
        self._fact_consolidator.consolidate(
            peer_id, request_id=request_id or _new_request_id(), session_id=session_id
        )

    def correct_fact(
        self,
        peer_id: str,
        category: str,
        key: str,
        value: str | None = None,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Directly upsert or delete one fact, bypassing consolidation.

        For explicit user-driven corrections (e.g. the eidolon_correct_fact
        tool). `value=None` (or blank) deletes the fact; otherwise upserts
        it with source="user". A later consolidation pass may still revise
        or remove this fact based on other accumulated evidence — that's
        accepted, not a bug.
        """
        category = category.strip().lower()
        key = key.strip().lower()
        if value is None or not value.strip():
            self._storage.delete_fact(peer_id, category, key)
            action = "delete"
        else:
            value = value.strip()
            self._storage.set_fact(peer_id, category, key, value, source="user")
            action = "upsert"
        self._tracer.record(
            "fact_correction",
            {"peer_id": peer_id, "category": category, "key": key, "value": value, "action": action},
            request_id=request_id or _new_request_id(), session_id=session_id,
        )

    # ── Internal Helpers ────────────────────────────────────

    def _get_topic_summary(self, topic_id: int | None) -> str | None:
        """Get the summary of a topic by its ID.

        Parameters
        ----------
        topic_id : int | None
            The topic ID to look up.

        Returns
        -------
        str | None
            The topic summary, or None if not found.
        """
        if topic_id is None:
            return None
        topic = self._storage.get_topic(topic_id)
        if topic is None:
            return None
        return topic.get("summary") or topic.get("label")

    def _recent_messages_for_topic(self, topic_id: int | None) -> list[dict]:
        """Fetch a small window of `topic_id`'s most recent messages, for
        the topic detector — see `_TOPIC_DETECTION_MESSAGE_WINDOW`.

        Runs before the current turn's message is stored, so there's no
        risk of it leaking into its own "recent history".
        """
        if topic_id is None:
            return []
        return self._storage.get_recent_messages_by_topic(topic_id, _TOPIC_DETECTION_MESSAGE_WINDOW)
