"""Associative retrieval for Eidolon.

Multi-dimensional weighted search across topics and memory items.
Combines semantic similarity (embedding), tag overlap, person-bound, and temporal closeness.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from core.interfaces import EmbedderProvider, MetricsRecorder, StorageProvider, Tracer
from core.metrics.null_metrics_recorder import NullMetricsRecorder
from core.models import ClusterSummary, ContextBundle, EnvironmentSnapshot, MemoryItemBudget, RelatedTopic
from core.stop_words import get_stop_words
from core.trace.null_tracer import NullTracer


class AssociativeRetriever:
    """Multi-dimensional associative search.

    Combines semantic (embedding), tag, person, and temporal dimensions
    with configurable weights to find relevant past topics and memory items.

    Parameters
    ----------
    storage : StorageProvider
        Backend for topic and memory item storage.
    embedder : EmbedderProvider
        Provider for generating query embeddings.
    weights : dict | None
        Optional weight overrides for retrieval dimensions, passed through
        to the storage backend's search methods.
        Defaults: semantic=0.4, tag_overlap=0.3, person=0.2, temporal=0.1
        ``temporal`` is a recency re-ranking signal applied by the storage
        backend over already-relevant candidates — set it to 0 to disable
        recency weighting entirely.
    language : str
        Language code for stop word filtering (default: "en").
        Prompts to the LLM are always in English regardless of this setting.
    limits : dict | None
        Optional result-count overrides, keyed by "related_topics",
        "memory_items_fact", "memory_items_surprise", "memory_items_feeling",
        "memory_items_free", "recent_topics", "cluster_summaries",
        "cluster_recent_topics", "min_topic_messages". Defaults: 5 for
        topics, 4/1/1/4 for the memory-item tiers (fact/surprise/feeling/
        free — see `MemoryItemBudget`), 5 for recent_topics, 3 for
        cluster_summaries, 2 for cluster_recent_topics, -1 (no
        filtering) for min_topic_messages. A caller passing its own
        dict here should include this key explicitly (see
        `RetrievalConfig.min_topic_messages`) — a missing key falls
        back to -1 at the call site, not this constructor default.
        ``min_topic_messages`` is applied to the relevance-scored
        related-topics search only, never to ``recent_topics`` — that
        surface is unconditional recency by design (see step 4b below).
        Unlike the older `_new`/`_continue` split this replaced, a single
        flat budget is used regardless of whether the turn is a new topic
        or a continuation — most of what's surfaced comes from other
        topics/history either way, so the distinction bought little.
    tracer : Tracer | None
        Debug tracer recording the retrieval summary and (when the storage
        backend supports it) per-candidate score breakdowns. Defaults to a
        no-op tracer.
    metrics : MetricsRecorder | None
        Always-on operational-metrics recorder. Defaults to a no-op
        recorder.
    """

    def __init__(
        self,
        storage: StorageProvider,
        embedder: EmbedderProvider,
        weights: dict[str, float] | None = None,
        language: str = "en",
        limits: dict[str, int] | None = None,
        tracer: Tracer | None = None,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._weights = weights or {
            "semantic": 0.4,
            "tag_overlap": 0.3,
            "person": 0.2,
            "temporal": 0.1,
        }
        self._limits = limits or {
            "related_topics": 5,
            "memory_items_fact": 4,
            "memory_items_surprise": 1,
            "memory_items_feeling": 1,
            "memory_items_free": 4,
            "recent_topics": 5,
            "cluster_summaries": 3,
            "cluster_recent_topics": 2,
            "min_topic_messages": -1,  # -1 = no mini-topic filtering
        }
        self._stop_words = get_stop_words(language)
        self._tracer = tracer or NullTracer()
        self._metrics = metrics or NullMetricsRecorder()

    def retrieve(
        self,
        message: str,
        workspace_id: str,
        environment: EnvironmentSnapshot,
        topic_label: str | None,
        is_new_topic: bool,
        active_topic_id: int | None = None,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> ContextBundle:
        """Retrieve relevant context for a user message.

        Parameters
        ----------
        message : str
            The user's message to find context for.
        workspace_id : str
            Who is speaking (e.g. "user", "ai:neuromancer").
        environment : EnvironmentSnapshot
            Current situational context.
        topic_label : str | None
            Detected topic label (from detect_topic). None if CONTINUE.
        is_new_topic : bool
            True if this is a NEW_TOPIC, False if CONTINUE.
        active_topic_id : int | None
            The topic being continued, if this is a CONTINUE turn — used to
            look up its summary so ``current_topic_label`` stays populated
            for the whole topic's lifetime instead of only on the turn it
            was detected as new.
        request_id : str | None
            Correlates this call's trace events with the enclosing
            top-level ``Eidolon`` call.
        session_id : str | None
            Adapter-level conversation identifier, for trace filtering.

        Returns
        -------
        ContextBundle
            Facts, related/recent topics, and memory items for prompt
            injection.
        """
        started = time.perf_counter()

        current_topic_label = topic_label
        if not is_new_topic and active_topic_id is not None:
            active_topic = self._storage.get_topic(active_topic_id)
            if active_topic:
                current_topic_label = active_topic.get("summary") or active_topic.get("label")

        # 1. Load facts for the current workspace
        facts = self._storage.get_facts(workspace_id)

        # 2. Generate query embedding for semantic search
        embed_started = time.perf_counter()
        query_embedding = self._embedder.embed(message)
        embed_duration_ms = (time.perf_counter() - embed_started) * 1000
        self._tracer.record(
            "embedding",
            {"purpose": "retrieval_query", "text": message, "dimension": len(query_embedding)},
            request_id=request_id,
            session_id=session_id,
            duration_ms=embed_duration_ms,
        )
        self._metrics.record_embedding_call(
            "retrieval_query", embed_duration_ms, len(query_embedding), True,
            workspace_id=workspace_id, request_id=request_id, session_id=session_id,
        )

        # 3. Extract potential tags from the message (simple keyword extraction)
        tags = self._extract_tags(message, stop_words=self._stop_words)

        # Reference time for recency scoring — the turn's own timestamp when
        # the adapter supplied one (keeps it consistent with the item-age the
        # ContextFormatter later displays), otherwise the current UTC time.
        now = environment.time or datetime.now(UTC)

        # 4. Search topics
        topic_limit = self._limits.get("related_topics", 5)
        topic_results = self._storage.search_topics(
            query_embedding=query_embedding,
            tags=tags,
            workspace_id=workspace_id,
            limit=topic_limit,
            weights=self._weights,
            tracer=self._tracer,
            request_id=request_id,
            now=now,
            min_message_count=self._limits.get("min_topic_messages", -1),
        )

        related_topics = [
            RelatedTopic(
                topic_id=t["id"],
                summary=t["summary"],
                tags=t.get("tags", []),
                timestamp=t.get("created_at"),
                label=t.get("label"),
            )
            for t in topic_results
        ]

        # 4b. Fetch the workspace's most-recently-discussed topics (pure
        # recency, not relevance) — padded so that even if every one of
        # them also showed up in the relevance search above, dedup still
        # leaves `recent_topics` limit worth of genuinely distinct ones.
        # Deliberately NOT passing min_topic_messages here (unlike the
        # search_topics call above) — this surface is documented to the
        # LLM as "regardless of relevance" (see ContextFormatter's
        # preamble), so filtering it by size would silently break that
        # promise for a workspace's most recent exchange, however short.
        recent_limit = self._limits.get("recent_topics", 5)
        related_ids = {t.topic_id for t in related_topics}
        recent_rows = self._storage.get_recent_topics(
            workspace_id=workspace_id, limit=recent_limit + len(related_ids),
        )
        recent_topics = [
            RelatedTopic(
                topic_id=t["id"],
                summary=t["summary"],
                tags=t.get("tags", []),
                timestamp=t.get("closed_at"),
                label=t.get("label"),
            )
            for t in recent_rows
            if t["id"] not in related_ids
        ][:recent_limit]

        # 5. Search memory items
        memory_budget = MemoryItemBudget(
            fact=self._limits.get("memory_items_fact", 4),
            surprise=self._limits.get("memory_items_surprise", 1),
            feeling=self._limits.get("memory_items_feeling", 1),
            free=self._limits.get("memory_items_free", 4),
        )
        memory_items = self._storage.search_memory_items(
            query_embedding=query_embedding,
            workspace_id=workspace_id,
            budget=memory_budget,
            weights=self._weights,
            tracer=self._tracer,
            request_id=request_id,
            now=now,
        )

        # 6. Search recurring-topic clusters — same query embedding/tags,
        # one limit regardless of new-vs-continue (clusters are longer-
        # lived threads, not as turn-sensitive as raw topic counts).
        cluster_results = self._storage.search_clusters(
            query_embedding=query_embedding,
            tags=tags,
            workspace_id=workspace_id,
            limit=self._limits.get("cluster_summaries", 3),
            weights=self._weights,
            tracer=self._tracer,
            request_id=request_id,
            now=now,
        )
        cluster_recent_limit = self._limits.get("cluster_recent_topics", 2)
        cluster_summaries = [
            ClusterSummary(
                cluster_id=c["id"],
                label=c["label"],
                summary=c["summary"],
                topic_count=c["topic_count"],
                recent_topics=[
                    RelatedTopic(
                        topic_id=t["id"],
                        summary=t["summary"],
                        tags=t.get("tags", []),
                        timestamp=t.get("closed_at"),
                        label=t.get("label"),
                    )
                    for t in self._storage.get_recent_topics_in_cluster(c["id"], cluster_recent_limit)
                ],
            )
            for c in cluster_results
        ]

        self._tracer.record(
            "context_retrieval",
            {
                "message": message,
                "workspace_id": workspace_id,
                "topic_label": topic_label,
                "is_new_topic": is_new_topic,
                "tags": tags,
                "environment": {
                    "situation": environment.situation,
                    "location": environment.location,
                    "time": environment.time.isoformat() if environment.time else None,
                    "day_of_week": environment.day_of_week,
                },
                "facts": [
                    {"category": f.category, "key": f.key, "value": f.value, "created_at": f.created_at}
                    for f in facts
                ],
                "related_topics": [
                    {"id": t.topic_id, "label": t.label, "summary": t.summary,
                     "tags": t.tags, "timestamp": t.timestamp}
                    for t in related_topics
                ],
                "recent_topics": [
                    {"id": t.topic_id, "label": t.label, "summary": t.summary,
                     "tags": t.tags, "timestamp": t.timestamp}
                    for t in recent_topics
                ],
                "memory_items": [
                    {
                        "content": mi.content,
                        "item_type": mi.item_type,
                        "source_peer": mi.source_peer,
                        "subject_peer": mi.subject_peer,
                        "timestamp": mi.timestamp,
                    }
                    for mi in memory_items
                ],
                "cluster_summaries": [
                    {"id": c.cluster_id, "label": c.label, "summary": c.summary, "topic_count": c.topic_count}
                    for c in cluster_summaries
                ],
            },
            request_id=request_id,
            session_id=session_id,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

        return ContextBundle(
            facts=facts,
            environment=environment,
            current_topic_label=current_topic_label,
            related_topics=related_topics,
            recent_topics=recent_topics,
            memory_items=memory_items,
            cluster_summaries=cluster_summaries,
        )

    @staticmethod
    def _extract_tags(message: str, stop_words: set[str] | None = None) -> list[str]:
        """Simple tag/keyword extraction from a message.

        Parameters
        ----------
        message : str
            The message to extract tags from.
        stop_words : set[str] | None
            Set of stop words to filter out. Uses English defaults if None.

        Returns
        -------
        list[str]
            Extracted tag candidates.
        """
        if stop_words is None:
            from core.stop_words import get_stop_words
            stop_words = get_stop_words("en")

        words = message.split()
        tags: list[str] = []

        for word in words:
            clean = word.strip(".,!?;:\"'()[]{}").lower()
            if len(clean) < 3 or clean in stop_words:
                continue
            tags.append(clean)

        return tags
