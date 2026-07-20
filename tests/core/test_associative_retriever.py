"""Tests for the associative retriever."""

from __future__ import annotations

from core.models import (
    ContextBundle,
    EnvironmentSnapshot,
    Fact,
    MemoryItem,
)
from core.retrieval.associative_retriever import AssociativeRetriever
from tests.mocks import SpyTracer


class MockStorage:
    """Mock storage that returns predefined data."""

    def __init__(self) -> None:
        self.facts: list[Fact] = []
        self.topics: list[dict] = []
        self.topics_by_id: dict[int, dict] = {}
        self.recent_topics_data: list[dict] = []
        self.memory_items: list[MemoryItem] = []
        self.last_recent_topics_peer_id: str | None = None
        self.last_recent_topics_limit: int | None = None
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
        return self.topics_by_id.get(topic_id)

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
        now=None,
    ) -> list[dict]:
        self.last_query_embedding = query_embedding
        self.last_tags = tags
        self.last_peer_id = peer_id
        self.last_weights = weights
        self.last_tracer = tracer
        self.last_request_id = request_id
        self.last_now = now
        return self.topics[:limit]

    def get_recent_topics(self, peer_id: str, limit: int) -> list[dict]:
        self.last_recent_topics_peer_id = peer_id
        self.last_recent_topics_limit = limit
        return self.recent_topics_data[:limit]

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
        now=None,
    ) -> list[MemoryItem]:
        self.last_weights = weights
        self.last_tracer = tracer
        self.last_request_id = request_id
        self.last_now = now
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

        # NEW_TOPIC: should request up to related_topics_new (default 6)
        bundle = retriever.retrieve(
            message="Lass uns über Bello sprechen",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label="Bello",
            is_new_topic=True,
        )
        assert len(bundle.related_topics) == 3  # all 3 returned

    def test_retrieve_populates_related_topic_label(self):
        storage = MockStorage()
        storage.topics = [
            {"id": 1, "label": "Bello", "summary": "About Bello", "tags": ["bello"]},
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
        assert bundle.related_topics[0].label == "Bello"

    def test_retrieve_continue_returns_fewer_results(self):
        storage = MockStorage()
        storage.topics = [
            {"id": 1, "label": "Bello", "summary": "About Bello", "tags": ["bello", "fitness"]},
            {"id": 2, "label": "Mampfi", "summary": "About Mampfi", "tags": ["mampfi", "shopping"]},
            {"id": 3, "label": "Fitness", "summary": "Fitness talk", "tags": ["fitness", "health"]},
        ]
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        # CONTINUE: AssociativeRetriever's own built-in default (unrelated
        # to RetrievalConfig's higher default) stays at 2 for backward
        # compatibility with callers that construct it without `limits=`.
        bundle = retriever.retrieve(
            message="jep, mach weiter",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label=None,
            is_new_topic=False,
        )
        assert len(bundle.related_topics) == 2  # limited to 2

    def test_retrieve_continue_limit_actually_applied(self):
        """With more candidates than the built-in CONTINUE limit, the limit
        must actually cut the result count — not just happen to match the
        candidate count, which the test above alone wouldn't catch."""
        storage = MockStorage()
        storage.topics = [
            {"id": i, "label": f"Topic {i}", "summary": f"Summary {i}", "tags": []}
            for i in range(10)
        ]
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        bundle = retriever.retrieve(
            message="jep, mach weiter",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label=None,
            is_new_topic=False,
        )
        assert len(bundle.related_topics) == 2  # built-in default related_topics_continue

    def test_custom_limits_override_related_topics(self):
        storage = MockStorage()
        storage.topics = [
            {"id": i, "label": f"Topic {i}", "summary": f"Summary {i}", "tags": []}
            for i in range(10)
        ]
        embedder = MockEmbedder()
        limits = {
            "related_topics_new": 8, "related_topics_continue": 4,
            "memory_items_new": 3, "memory_items_continue": 1,
        }
        retriever = AssociativeRetriever(storage, embedder, limits=limits)

        new_bundle = retriever.retrieve(
            message="hi", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label="Test", is_new_topic=True,
        )
        assert len(new_bundle.related_topics) == 8

        continue_bundle = retriever.retrieve(
            message="hi", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label=None, is_new_topic=False,
        )
        assert len(continue_bundle.related_topics) == 4

    def test_custom_limits_override_memory_items(self):
        storage = MockStorage()
        storage.memory_items = [
            MemoryItem(content=f"item {i}", item_type="essence", source_peer="user",
                       subject_peer="user", attribution_type="self_statement")
            for i in range(10)
        ]
        embedder = MockEmbedder()
        limits = {
            "related_topics_new": 5, "related_topics_continue": 2,
            "memory_items_new": 7, "memory_items_continue": 5,
        }
        retriever = AssociativeRetriever(storage, embedder, limits=limits)

        new_bundle = retriever.retrieve(
            message="hi", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label="Test", is_new_topic=True,
        )
        assert len(new_bundle.memory_items) == 7

        continue_bundle = retriever.retrieve(
            message="hi", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label=None, is_new_topic=False,
        )
        assert len(continue_bundle.memory_items) == 5

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
            {"id": 1, "label": "Bello", "summary": "About Bello", "tags": ["bello"], "timestamp": None}
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


class TestAssociativeRetrieverCurrentTopicLabel:
    def test_new_topic_uses_detected_label_not_active_topic(self):
        """On a NEW_TOPIC turn, current_topic_label is the freshly detected
        label — active_topic_id (the topic just being left) must not
        override it, even if given."""
        storage = MockStorage()
        storage.topics_by_id[1] = {"id": 1, "label": "Old", "summary": "Old summary"}
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        bundle = retriever.retrieve(
            message="hi", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label="New Topic", is_new_topic=True, active_topic_id=1,
        )
        assert bundle.current_topic_label == "New Topic"

    def test_continue_resolves_label_from_active_topic_summary(self):
        """On CONTINUE, topic_label is always None (nothing was just
        detected) — current_topic_label must instead persist by looking
        up the active topic's own summary, so it doesn't go blank for the
        rest of the topic's lifetime."""
        storage = MockStorage()
        storage.topics_by_id[5] = {"id": 5, "label": "Bello", "summary": "Talking about Bello's diet"}
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        bundle = retriever.retrieve(
            message="mach weiter", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label=None, is_new_topic=False, active_topic_id=5,
        )
        assert bundle.current_topic_label == "Talking about Bello's diet"

    def test_continue_falls_back_to_label_when_summary_missing(self):
        storage = MockStorage()
        storage.topics_by_id[5] = {"id": 5, "label": "Bello", "summary": ""}
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        bundle = retriever.retrieve(
            message="mach weiter", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label=None, is_new_topic=False, active_topic_id=5,
        )
        assert bundle.current_topic_label == "Bello"

    def test_continue_without_active_topic_id_stays_none(self):
        storage = MockStorage()
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        bundle = retriever.retrieve(
            message="mach weiter", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label=None, is_new_topic=False,
        )
        assert bundle.current_topic_label is None

    def test_continue_with_unknown_active_topic_id_stays_none(self):
        storage = MockStorage()  # topics_by_id has no entry for 999
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        bundle = retriever.retrieve(
            message="mach weiter", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label=None, is_new_topic=False, active_topic_id=999,
        )
        assert bundle.current_topic_label is None


class TestAssociativeRetrieverRecentTopics:
    def test_recent_topics_populated_from_storage(self):
        storage = MockStorage()
        storage.recent_topics_data = [
            {"id": 9, "label": "Last week", "summary": "About last week", "tags": [], "closed_at": "t1"},
        ]
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        bundle = retriever.retrieve(
            message="hi", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label="Test", is_new_topic=True,
        )
        assert len(bundle.recent_topics) == 1
        assert bundle.recent_topics[0].topic_id == 9
        assert bundle.recent_topics[0].summary == "About last week"

    def test_recent_topics_excludes_ones_already_in_related_topics(self):
        """The whole point of the feature: don't show the same topic twice
        under [Recently discussed] and [Related topics]."""
        storage = MockStorage()
        storage.topics = [
            {"id": 1, "label": "Bello", "summary": "About Bello", "tags": ["bello"]},
        ]
        storage.recent_topics_data = [
            {"id": 1, "label": "Bello", "summary": "About Bello", "tags": ["bello"], "closed_at": "t1"},
            {"id": 2, "label": "Other", "summary": "About something else", "tags": [], "closed_at": "t2"},
        ]
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        bundle = retriever.retrieve(
            message="hi", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label="Bello", is_new_topic=True,
        )
        assert [t.topic_id for t in bundle.related_topics] == [1]
        assert [t.topic_id for t in bundle.recent_topics] == [2]

    def test_recent_topics_respects_limit(self):
        storage = MockStorage()
        storage.recent_topics_data = [
            {"id": i, "label": f"Topic {i}", "summary": f"Summary {i}", "tags": [], "closed_at": f"t{i}"}
            for i in range(5)
        ]
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder, limits={
            "related_topics_new": 5, "related_topics_continue": 2,
            "memory_items_new": 3, "memory_items_continue": 1,
            "recent_topics": 2,
        })

        bundle = retriever.retrieve(
            message="hi", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label="Test", is_new_topic=True,
        )
        assert len(bundle.recent_topics) == 2

    def test_recent_topics_limit_defaults_when_not_in_custom_limits(self):
        """A caller-supplied `limits` dict that predates this feature
        (missing the "recent_topics" key) must not crash — it should fall
        back to the built-in default instead of a KeyError."""
        storage = MockStorage()
        storage.recent_topics_data = [
            {"id": i, "label": f"Topic {i}", "summary": f"Summary {i}", "tags": [], "closed_at": f"t{i}"}
            for i in range(5)
        ]
        embedder = MockEmbedder()
        limits = {
            "related_topics_new": 5, "related_topics_continue": 2,
            "memory_items_new": 3, "memory_items_continue": 1,
        }
        retriever = AssociativeRetriever(storage, embedder, limits=limits)

        bundle = retriever.retrieve(
            message="hi", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label="Test", is_new_topic=True,
        )
        assert len(bundle.recent_topics) == 2  # built-in default

    def test_recent_topics_query_scoped_to_peer(self):
        storage = MockStorage()
        embedder = MockEmbedder()
        retriever = AssociativeRetriever(storage, embedder)

        retriever.retrieve(
            message="hi", peer_id="alice", environment=EnvironmentSnapshot(),
            topic_label="Test", is_new_topic=True,
        )
        assert storage.last_recent_topics_peer_id == "alice"

    def test_records_recent_topics_in_context_retrieval_event(self):
        storage = MockStorage()
        storage.recent_topics_data = [
            {"id": 9, "label": "Last week", "summary": "About last week", "tags": [], "closed_at": "t1"},
        ]
        embedder = MockEmbedder()
        tracer = SpyTracer()
        retriever = AssociativeRetriever(storage, embedder, tracer=tracer)

        retriever.retrieve(
            message="hi", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label="Test", is_new_topic=True, request_id="req1",
        )

        payload = tracer.events_of_type("context_retrieval")[0]["payload"]
        assert payload["recent_topics"] == [
            {"id": 9, "label": "Last week", "summary": "About last week", "tags": [], "timestamp": "t1"}
        ]
