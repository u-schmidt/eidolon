"""Associative retrieval for Eidolon.

Multi-dimensional weighted search across topics and memory items.
Combines semantic similarity (embedding), tag overlap, person-bound, and temporal closeness.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.interfaces import EmbedderProvider, StorageProvider
from core.models import ContextBundle, EnvironmentSnapshot, Fact, MemoryItem, RelatedTopic
from core.stop_words import get_stop_words


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
        Optional weight overrides for retrieval dimensions.
        Defaults: semantic=0.4, tag_overlap=0.3, person=0.2, temporal=0.1
    language : str
        Language code for stop word filtering (default: "en").
        Prompts to the LLM are always in English regardless of this setting.
    """

    def __init__(
        self,
        storage: StorageProvider,
        embedder: EmbedderProvider,
        weights: dict[str, float] | None = None,
        language: str = "en",
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

    def retrieve(
        self,
        message: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        topic_label: str | None,
        is_new_topic: bool,
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

        Returns
        -------
        ContextBundle
            Facts, related topics, and memory items for prompt injection.
        """
        # 1. Load facts for the current peer
        facts = self._storage.get_facts(peer_id)

        # 2. Generate query embedding for semantic search
        query_embedding = self._embedder.embed(message)

        # 3. Extract potential tags from the message (simple keyword extraction)
        tags = self._extract_tags(message)

        # 4. Search topics
        topic_limit = 5 if is_new_topic else 2
        topic_results = self._storage.search_topics(
            query_embedding=query_embedding,
            tags=tags,
            peer_id=peer_id,
            limit=topic_limit,
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
