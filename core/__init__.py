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

from core.detection.topic_detector import TopicDetector
from core.extraction.fragmentary_extractor import FragmentaryExtractor
from core.interfaces import EmbedderProvider, LLMProvider, StorageProvider
from core.models import ContextBundle, EnvironmentSnapshot, MemoryItem, TopicChange
from core.retrieval.associative_retriever import AssociativeRetriever


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
    """

    def __init__(
        self,
        storage: StorageProvider,
        llm: LLMProvider,
        embedder: EmbedderProvider,
        language: str = "en",
        retrieval_weights: dict[str, float] | None = None,
    ) -> None:
        self._storage = storage
        self._detector = TopicDetector(llm)
        self._retriever = AssociativeRetriever(storage, embedder, retrieval_weights, language)
        self._extractor = FragmentaryExtractor(llm, language)
        self._embedder = embedder
        self._llm = llm
        self._language = language

    # ── Public API ─────────────────────────────────────────

    def prepare_context(
        self,
        message: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        active_topic_id: int | None = None,
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

        Returns
        -------
        ContextBundle
            Facts, environment, topic info, related topics, and memory items
            for prompt injection.
        """
        # 1. Detect topic change
        current_topic_summary = self._get_topic_summary(active_topic_id)
        topic_change = self._detector.detect(message, current_topic_summary)

        # 2. Retrieve context
        is_new_topic = topic_change.type == "NEW_TOPIC"
        bundle = self._retriever.retrieve(
            message=message,
            peer_id=peer_id,
            environment=environment,
            topic_label=topic_change.label,
            is_new_topic=is_new_topic,
        )

        return bundle

    def detect_topic(
        self,
        message: str,
        active_topic_id: int | None = None,
    ) -> TopicChange:
        """Detect whether a message continues or changes the topic.

        Parameters
        ----------
        message : str
            The user's message.
        active_topic_id : int | None
            The current active topic ID, if any.

        Returns
        -------
        TopicChange
            CONTINUE or NEW_TOPIC with a label.
        """
        current_topic_summary = self._get_topic_summary(active_topic_id)
        return self._detector.detect(message, current_topic_summary)

    def retrieve_context(
        self,
        message: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        topic_label: str | None = None,
        is_new_topic: bool = False,
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

        Returns
        -------
        ContextBundle
            Facts, related topics, and memory items.
        """
        return self._retriever.retrieve(
            message=message,
            peer_id=peer_id,
            environment=environment,
            topic_label=topic_label,
            is_new_topic=is_new_topic,
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

        Returns
        -------
        int | None
            The new topic ID if a topic was created, None otherwise.
        """
        # 1. Close old topic if topic changed — and generate summary + tags
        if old_topic_id is not None and new_topic_label is not None:
            self._summarize_topic(old_topic_id)
            self._storage.close_topic(old_topic_id, f"Continued to: {new_topic_label}")

        # 2. Create new topic if topic changed
        new_topic_id: int | None = None
        if new_topic_label is not None:
            new_topic_id = self._storage.create_topic(
                label=new_topic_label,
                peer_id=peer_id,
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
                topic_embedding = self._embedder.embed(new_topic_label)
                self._storage.update_topic_embedding(new_topic_id, topic_embedding)
            except Exception:
                pass  # Embedding failure is non-critical

        # 5. Extract memory fragments
        items = self._extractor.extract(
            user_message=user_message,
            ai_response=ai_response,
            source_peer=peer_id,
            topic_label=new_topic_label,
            environment=environment,
        )

        # 6. Store memory items with topic + message association + embeddings
        fact_items: list[MemoryItem] = []
        for item in items:
            if new_topic_id is not None:
                item.topic_id = new_topic_id
            # Link items to the message they were extracted from
            if item.source_peer == "ai":
                item.message_id = ai_msg_id
            else:
                item.message_id = user_msg_id
            # Generate embedding for the item content
            if item.content:
                try:
                    item.embedding = self._embedder.embed(item.content)
                except Exception:
                    pass  # Embedding failure is non-critical
            self._storage.store_memory_item(item)
            # Collect fact-type items for structured fact extraction
            if item.item_type == "fact":
                fact_items.append(item)

        # 7. Extract structured facts from fact-type memory items
        if fact_items:
            self._extract_facts(fact_items, peer_id)

        return new_topic_id

    # ── Internal Helpers ────────────────────────────────────

    def _extract_facts(
        self, fact_items: list[MemoryItem], peer_id: str
    ) -> None:
        """Extract structured facts from fact-type memory items.

        Takes memory items with item_type="fact" and asks the LLM to
        convert them into structured (category, key, value) triples
        that are stored in the facts table.

        Called async after fragmentary extraction.

        Parameters
        ----------
        fact_items : list[MemoryItem]
            Memory items with item_type="fact" from the current turn.
        peer_id : str
            The peer these facts are about.
        """
        if not fact_items:
            return

        try:
            # Build a list of fact statements
            statements = "\n".join(
                f"- {item.content}" for item in fact_items
            )

            prompt = (
                "Convert these facts about a user into structured (category, key, value) triples.\n\n"
                f"Facts:\n{statements}\n\n"
                "Rules:\n"
                "- category: a broad domain like 'preferences', 'health', 'work', 'fitness', 'lifestyle'\n"
                "- key: a specific, concise identifier (snake_case, 2-4 words)\n"
                "- value: the fact itself (1 sentence)\n\n"
                "Respond with a JSON array only. Example:\n"
                '[\n'
                '  {\n'
                '    "category": "fitness",\n'
                '    "key": "cardio_clothing_preference",\n'
                '    "value": "dislikes heavy cotton shirts during cardio"\n'
                '  }\n'
                ']\n\n'
                "If nothing to extract, respond with: []"
            )

            response = self._llm.complete(
                prompt=prompt,
                system_prompt=(
                    "You extract structured facts from conversation fragments. "
                    "Output JSON only."
                ),
                max_tokens=1000,
            )

            # Parse JSON response
            import json
            text = response.strip()
            if text.startswith("```"):
                start = text.find("[")
                if start != -1:
                    text = text[start:]
                end = text.rfind("```")
                if end != -1:
                    text = text[:end]
                text = text.strip()

            structured = json.loads(text)
            if not isinstance(structured, list):
                return

            for fact in structured:
                category = fact.get("category", "").strip()
                key = fact.get("key", "").strip()
                value = fact.get("value", "").strip()
                if category and key and value:
                    self._storage.set_fact(
                        peer_id=peer_id,
                        category=category,
                        key=key,
                        value=value,
                        source="ai",
                    )
        except Exception:
            pass  # Fact extraction failure is non-critical

    def _summarize_topic(self, topic_id: int) -> None:
        """Generate a summary and tags for a topic based on its messages.

        Called async when a topic is closed. Loads all messages for the topic,
        asks the LLM for a concise summary and relevant tags, then updates
        the topic in storage.

        Parameters
        ----------
        topic_id : int
            The topic ID to summarize.
        """
        try:
            messages = self._storage.get_messages_by_topic(topic_id)
            if not messages:
                return

            # Build a compact conversation transcript
            transcript = "\n".join(
                f"{m['role']}: {m['content'][:200]}"
                for m in messages
            )

            prompt = (
                f"Summarize this conversation topic in 1-2 sentences "
                f"and extract 3-5 key tags (comma-separated, lowercase).\n\n"
                f"{transcript}\n\n"
                f"Format:\n"
                f"Summary: <1-2 sentences>\n"
                f"Tags: <tag1, tag2, tag3>"
            )

            response = self._llm.complete(
                prompt=prompt,
                system_prompt=(
                    "You are a conversation summarizer. "
                    "Output a concise summary and relevant tags only."
                ),
                max_tokens=1000,
            )

            # Parse response
            summary = ""
            tags: list[str] = []
            for line in response.strip().split("\n"):
                if line.lower().startswith("summary:"):
                    summary = line[len("summary:"):].strip()
                elif line.lower().startswith("tags:"):
                    raw_tags = line[len("tags:"):].strip()
                    tags = [t.strip().lower() for t in raw_tags.split(",") if t.strip()]

            if summary:
                self._storage.close_topic(topic_id, summary)
                if tags:
                    # Update tags via raw SQL (no dedicated method yet)
                    import json
                    self._storage._conn.execute(
                        "UPDATE topics SET tags = ? WHERE id = ?",
                        (json.dumps(tags), topic_id),
                    )
                    self._storage._conn.commit()
        except Exception:
            pass  # Summarization failure is non-critical

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
