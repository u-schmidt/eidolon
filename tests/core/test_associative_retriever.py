"""Tests for the associative retriever."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.models import (
    ContextBundle,
    EnvironmentSnapshot,
    Fact,
    MemoryItem,
    RelatedTopic,
)
from core.retrieval.associative_retriever import AssociativeRetriever
from tests.mocks import SpyTracer


class MockStorage:
    """Mock storage that returns predefined data."""

    def __init__(self) -> None:
        self.facts: list[Fact] = []
        self.topics: list[dict] = []
        self.memory_items: list[MemoryItem] = []
        self.last_query_embedding: list[float] | None = None
        self.last_tags: list[str] | None = None
        self.last_peer_id: str | None = None
        self.last_weights: dict[str, float] | None = None
        self.last_tracer = None
        self.last_request_id: str | None = None

    def get_facts(self, peer_id: str) -> list[Fact]:
        self.last_peer_id = peer_id
        return self.facts

    def set_fact(self, peer_id: str, category: str, key: str, value: str, source: str) -> None:
        pass

    def create_topic(self, *args, **kwargs) -> int:
        return 1

    def get_topic(self, topic_id: int) -> dict | None:
        return None

    def close_topic(self, topic_id: int, summary: str) -> None:
        pass

    def update_topic_embedding(self, topic_id: int, embedding: list[float]) -> None:
        pass

    def search_topics(
        self,
        query_embedding: list[float] | None = None,
        tags: list[str] | None = None,
        peer_id: str | None = None,
        limit: int = 5,
        weights: dict[str, float] | None = None,
        tracer=None,
        request_id: str | None = None,
    ) -> list[dict]:
        self.last_query_embedding = query_embedding
        self.last_tags = tags
        self.last_peer_id = peer_id
        self.last_weights = weights
        self.last_tracer = tracer
        self.last_request_id = request_id
        return self.topics[:limit]

    def store_memory_item(self, item: MemoryItem) -> int:
        return 1

    def search_memory_items(
        self,
        query_embedding: list[float] | None = None,
        peer_id: str | None = None,
        limit: int = 5,
        weights: dict[str, float] | None = None,
        tracer=None,
        request_id: str | None = None,
    ) -> list[MemoryItem]:
        self.last_weights = weights
        self.last_tracer = tracer
        self.last_request_id = request_id
        return self.memory_items[:limit]


class MockEmbedder:
    """Mock embedder that returns a fixed vector."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]
        self.last_text: str | None = None

    def embed(self, text: str) -> list[float]:
        self.last_text = text
        return self.vector


class TestAssociativeRetriever:
    def test_retrieve_with_facts(self):
        storage = MockStorage()
        storage.facts = [
            Fact(peer_id="user", category="basic", key="name", value="Alex", source="user"),
        ]
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        env = EnvironmentSnapshot(location="home", situation="morning")
        bundle = retriever.retrieve(
            message="Lass uns über Bello sprechen",
            peer_id="user",
            environment=env,
            topic_label="Bello Features",
            is_new_topic=True,
        )

        assert isinstance(bundle, ContextBundle)
        assert len(bundle.facts) == 1
        assert bundle.facts[0].key == "name"
        assert bundle.current_topic_label == "Bello Features"
        assert bundle.environment is not None
        assert bundle.environment.location == "home"

    def test_retrieve_new_topic_returns_more_results(self):
        storage = MockStorage()
        storage.topics = [
            {"id": 1, "label": "Bello", "summary": "About Bello", "tags": ["bello", "fitness"]},
            {"id": 2, "label": "Mampfi", "summary": "About Mampfi", "tags": ["mampfi", "shopping"]},
            {"id": 3, "label": "Fitness", "summary": "Fitness talk", "tags": ["fitness", "health"]},
        ]
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        # NEW_TOPIC: should request up to 5 topics
        bundle = retriever.retrieve(
            message="Lass uns über Bello sprechen",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label="Bello",
            is_new_topic=True,
        )
        assert len(bundle.related_topics) == 3  # all 3 returned

    def test_retrieve_continue_returns_fewer_results(self):
        storage = MockStorage()
        storage.topics = [
            {"id": 1, "label": "Bello", "summary": "About Bello", "tags": ["bello", "fitness"]},
            {"id": 2, "label": "Mampfi", "summary": "About Mampfi", "tags": ["mampfi", "shopping"]},
            {"id": 3, "label": "Fitness", "summary": "Fitness talk", "tags": ["fitness", "health"]},
        ]
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        # CONTINUE: should request only 2 topics
        bundle = retriever.retrieve(
            message="jep, mach weiter",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label=None,
            is_new_topic=False,
        )
        assert len(bundle.related_topics) == 2  # limited to 2

    def test_retrieve_with_memory_items(self):
        storage = MockStorage()
        storage.memory_items = [
            MemoryItem(content="User talked about Bello", item_type="essence",
                       source_peer="user", subject_peer="user", attribution_type="self_statement"),
        ]
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        bundle = retriever.retrieve(
            message="Lass uns über Bello sprechen",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label="Bello",
            is_new_topic=True,
        )
        assert len(bundle.memory_items) == 1

    def test_retrieve_empty(self):
        storage = MockStorage()
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        bundle = retriever.retrieve(
            message="Hello",
            peer_id="unknown",
            environment=EnvironmentSnapshot(),
            topic_label=None,
            is_new_topic=False,
        )
        assert len(bundle.facts) == 0
        assert len(bundle.related_topics) == 0
        assert len(bundle.memory_items) == 0
        assert bundle.current_topic_label is None

    def test_embedder_called_with_message(self):
        storage = MockStorage()
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        retriever.retrieve(
            message="Lass uns über Bello sprechen",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label="Bello",
            is_new_topic=True,
        )
        assert embedder.last_text == "Lass uns über Bello sprechen"

    def test_storage_search_called_with_embedding(self):
        storage = MockStorage()
        embedder = MockEmbedder(vector=[0.5, 0.5, 0.5])
        retriever = AssociativeRetriever(storage, embedder)

        retriever.retrieve(
            message="test",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label="Test",
            is_new_topic=True,
        )
        assert storage.last_query_embedding == [0.5, 0.5, 0.5]

    def test_extract_tags(self):
        from core.stop_words import get_stop_words
        stop_words = get_stop_words("de")
        tags = AssociativeRetriever._extract_tags("Lass uns über Bello und Fitness sprechen", stop_words=stop_words)
        assert "bello" in tags
        assert "fitness" in tags
        assert "lass" not in tags  # stop word
        assert "uns" not in tags  # stop word
        assert "über" not in tags  # stop word

    def test_extract_tags_empty_message(self):
        tags = AssociativeRetriever._extract_tags("")
        assert tags == []

    def test_extract_tags_short_words_filtered(self):
        tags = AssociativeRetriever._extract_tags("yes no ok")
        assert tags == []

    def test_extract_tags_english_default(self):
        tags = AssociativeRetriever._extract_tags("Let's talk about Bello and fitness")
        assert "bello" in tags
        assert "fitness" in tags
        assert "the" not in tags  # English stop word
        assert "about" not in tags  # English stop word

    def test_custom_weights(self):
        storage = MockStorage()
        embedder = MockEmbedder()
        weights = {"semantic": 0.5, "tag_overlap": 0.3, "person": 0.1, "temporal": 0.1}
        retriever = AssociativeRetriever(storage, embedder, weights=weights)
        assert retriever._weights == weights

    def test_retrieve_uses_configured_language_for_tags(self):
        """`language` passed to the retriever must actually reach tag
        extraction during retrieve() — self._stop_words was previously
        computed but never used, so German conversations always got
        English (i.e. no-op) stop-word filtering regardless of config."""
        storage = MockStorage()
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder, language="de")

        retriever.retrieve(
            message="Lass uns über Bello und Fitness sprechen",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label="Bello",
            is_new_topic=True,
        )

        # German stop words ("lass", "uns", "über") must be filtered out;
        # with the English default they would have leaked through as tags.
        assert "lass" not in storage.last_tags
        assert "uns" not in storage.last_tags
        assert "über" not in storage.last_tags
        assert "bello" in storage.last_tags
        assert "fitness" in storage.last_tags

    def test_custom_weights_passed_to_storage(self):
        """Configured weights must reach the storage backend's search calls,
        not just sit unused on the retriever."""
        storage = MockStorage()
        embedder = MockEmbedder()
        weights = {"semantic": 0.5, "tag_overlap": 0.3, "person": 0.1, "temporal": 0.1}
        retriever = AssociativeRetriever(storage, embedder, weights=weights)

        retriever.retrieve(
            message="Lass uns über Bello sprechen",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label="Bello",
            is_new_topic=True,
        )
        assert storage.last_weights == weights

    def test_tracer_passed_to_storage(self):
        storage = MockStorage()
        embedder = MockEmbedder()
        tracer = SpyTracer()
        retriever = AssociativeRetriever(storage, embedder, tracer=tracer)

        retriever.retrieve(
            message="test",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label="Test",
            is_new_topic=True,
            request_id="req1",
        )
        assert storage.last_tracer is tracer
        assert storage.last_request_id == "req1"

    def test_records_context_retrieval_event(self):
        storage = MockStorage()
        storage.topics = [
            {"id": 1, "label": "Bello", "summary": "About Bello", "tags": ["bello"]},
        ]
        embedder = MockEmbedder()
        tracer = SpyTracer()
        retriever = AssociativeRetriever(storage, embedder, tracer=tracer)

        retriever.retrieve(
            message="Lass uns über Bello sprechen",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label="Bello",
            is_new_topic=True,
            request_id="req1",
            session_id="sess1",
        )

        events = tracer.events_of_type("context_retrieval")
        assert len(events) == 1
        event = events[0]
        assert event["request_id"] == "req1"
        assert event["session_id"] == "sess1"
        assert event["duration_ms"] is not None
        assert event["payload"]["message"] == "Lass uns über Bello sprechen"
        assert event["payload"]["related_topics"] == [
            {"id": 1, "summary": "About Bello", "tags": ["bello"], "timestamp": None}
        ]

    def test_context_retrieval_event_carries_full_content_not_counts(self):
        """The payload must carry actual fact/memory-item/environment content
        for a debug UI to display — not just counts, which was the previous
        (insufficient) shape."""
        storage = MockStorage()
        storage.facts = [
            Fact(peer_id="user", category="fitness", key="gym", value="FitZone", source="user"),
        ]
        storage.memory_items = [
            MemoryItem(content="User likes FitZone", item_type="essence",
                       source_peer="user", subject_peer="user", attribution_type="self_statement"),
        ]
        embedder = MockEmbedder()
        tracer = SpyTracer()
        retriever = AssociativeRetriever(storage, embedder, tracer=tracer)

        retriever.retrieve(
            message="hello",
            peer_id="user",
            environment=EnvironmentSnapshot(situation="conversation", location="home"),
            topic_label=None,
            is_new_topic=False,
            request_id="req1",
        )

        payload = tracer.events_of_type("context_retrieval")[0]["payload"]
        assert "facts_count" not in payload
        assert "memory_items_count" not in payload
        assert payload["facts"] == [
            {"category": "fitness", "key": "gym", "value": "FitZone", "created_at": None}
        ]
        assert payload["memory_items"] == [
            {"content": "User likes FitZone", "item_type": "essence",
             "source_peer": "user", "subject_peer": "user", "timestamp": None}
        ]
        assert payload["environment"]["situation"] == "conversation"
        assert payload["environment"]["location"] == "home"

    def test_records_query_embedding_event(self):
        storage = MockStorage()
        embedder = MockEmbedder(vector=[0.1, 0.2, 0.3])
        tracer = SpyTracer()
        retriever = AssociativeRetriever(storage, embedder, tracer=tracer)

        retriever.retrieve(
            message="test message",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label="Test",
            is_new_topic=True,
            request_id="req1",
        )

        events = tracer.events_of_type("embedding")
        assert len(events) == 1
        assert events[0]["payload"]["purpose"] == "retrieval_query"
        assert events[0]["payload"]["text"] == "test message"
        assert events[0]["payload"]["dimension"] == 3
