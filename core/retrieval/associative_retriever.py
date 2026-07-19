"""Associative retrieval for Eidolon.

Multi-dimensional weighted search across topics and memory items.
Combines semantic similarity (embedding), tag overlap, person-bound, and temporal closeness.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from core.interfaces import EmbedderProvider, StorageProvider, Tracer
from core.models import ContextBundle, EnvironmentSnapshot, Fact, MemoryItem, RelatedTopic
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
        NOTE: ``temporal`` is not yet applied by any storage backend —
        there is no recency scoring term implemented yet. It is kept
        here (rather than dropped) so the dimension isn't forgotten.
    language : str
        Language code for stop word filtering (default: "en").
        Prompts to the LLM are always in English regardless of this setting.
    tracer : Tracer | None
        Debug tracer recording the retrieval summary and (when the storage
        backend supports it) per-candidate score breakdowns. Defaults to a
        no-op tracer.
    """

    def __init__(
        self,
        storage: StorageProvider,
        embedder: EmbedderProvider,
        weights: dict[str, float] | None = None,
        language: str = "en",
        tracer: Tracer | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._weights = weights or {
            "semantic": 0.4,
            "tag_overlap": 0.3,
            "person": 0.2,
            "temporal": 0.1,
        }
        self._stop_words = get_stop_words(language)
        self._tracer = tracer or NullTracer()

    def retrieve(
        self,
        message: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        topic_label: str | None,
        is_new_topic: bool,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> ContextBundle:
        """Retrieve relevant context for a user message.

        Parameters
        ----------
        message : str
            The user's message to find context for.
        peer_id : str
            Who is speaking (e.g. "user", "ai:neuromancer").
        environment : EnvironmentSnapshot
            Current situational context.
        topic_label : str | None
            Detected topic label (from detect_topic). None if CONTINUE.
        is_new_topic : bool
            True if this is a NEW_TOPIC, False if CONTINUE.
        request_id : str | None
            Correlates this call's trace events with the enclosing
            top-level ``Eidolon`` call.
        session_id : str | None
            Adapter-level conversation identifier, for trace filtering.

        Returns
        -------
        ContextBundle
            Facts, related topics, and memory items for prompt injection.
        """
        started = time.perf_counter()

        # 1. Load facts for the current peer
        facts = self._storage.get_facts(peer_id)

        # 2. Generate query embedding for semantic search
        embed_started = time.perf_counter()
        query_embedding = self._embedder.embed(message)
        self._tracer.record(
            "embedding",
            {"purpose": "retrieval_query", "text": message, "dimension": len(query_embedding)},
            request_id=request_id,
            session_id=session_id,
            duration_ms=(time.perf_counter() - embed_started) * 1000,
        )

        # 3. Extract potential tags from the message (simple keyword extraction)
        tags = self._extract_tags(message, stop_words=self._stop_words)

        # 4. Search topics
        topic_limit = 5 if is_new_topic else 2
        topic_results = self._storage.search_topics(
            query_embedding=query_embedding,
            tags=tags,
            peer_id=peer_id,
            limit=topic_limit,
            weights=self._weights,
            tracer=self._tracer,
            request_id=request_id,
        )

        related_topics = [
            RelatedTopic(
                topic_id=t["id"],
                summary=t["summary"],
                tags=t.get("tags", []),
                timestamp=t.get("created_at"),
            )
            for t in topic_results
        ]

        # 5. Search memory items
        memory_limit = 3 if is_new_topic else 1
        memory_items = self._storage.search_memory_items(
            query_embedding=query_embedding,
            peer_id=peer_id,
            limit=memory_limit,
            weights=self._weights,
            tracer=self._tracer,
            request_id=request_id,
        )

        self._tracer.record(
            "context_retrieval",
            {
                "message": message,
                "peer_id": peer_id,
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
                    {"id": t.topic_id, "summary": t.summary, "tags": t.tags, "timestamp": t.timestamp}
                    for t in related_topics
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
            },
            request_id=request_id,
            session_id=session_id,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

        return ContextBundle(
            facts=facts,
            environment=environment,
            current_topic_label=topic_label,
            related_topics=related_topics,
            memory_items=memory_items,
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
