"""Tests for the Eidolon core facade."""

from __future__ import annotations

from core import Eidolon
from core.models import ContextBundle, EnvironmentSnapshot, Fact, MemoryItem, TopicChange


class MockStorage:
    """Mock storage for facade testing."""

    def __init__(self) -> None:
        self.facts: list[Fact] = []
        self.topics: dict[int, dict] = {}
        self.memory_items: list[MemoryItem] = []
        self._next_topic_id = 1
        self.closed_topics: list[int] = []
        self.stored_items: list[MemoryItem] = []

    def get_facts(self, peer_id: str) -> list[Fact]:
        return self.facts

    def set_fact(self, peer_id: str, category: str, key: str, value: str, source: str) -> None:
        self.facts.append(Fact(peer_id=peer_id, category=category, key=key, value=value, source=source))

    def create_topic(self, label: str, summary: str = "", tags: list[str] | None = None,
                     embedding: list[float] | None = None, peer_id: str | None = None) -> int:
        tid = self._next_topic_id
        self._next_topic_id += 1
        self.topics[tid] = {"id": tid, "label": label, "summary": summary, "tags": tags or [],
                           "status": "active", "peer_id": peer_id}
        return tid

    def get_topic(self, topic_id: int) -> dict | None:
        return self.topics.get(topic_id)

    def close_topic(self, topic_id: int, summary: str) -> None:
        if topic_id in self.topics:
            self.topics[topic_id]["status"] = "closed"
            self.topics[topic_id]["summary"] = summary
            self.closed_topics.append(topic_id)

    def update_topic_embedding(self, topic_id: int, embedding: list[float]) -> None:
        pass

    def search_topics(self, query_embedding=None, tags=None, peer_id=None, limit=5):
        return [t for t in self.topics.values() if t["status"] == "closed"][:limit]

    def store_memory_item(self, item: MemoryItem) -> int:
        self.stored_items.append(item)
        return len(self.stored_items)

    def search_memory_items(self, query_embedding=None, peer_id=None, limit=5):
        return self.memory_items[:limit]

    def store_message(self, turn_id, role, content, peer_id, topic_id=None):
        return 1  # dummy id


class MockLLM:
    def __init__(self, detection_response: str = "CONTINUE", extraction_response: str = "[]"):
        self.detection_response = detection_response
        self.extraction_response = extraction_response

    def complete(self, prompt: str, system_prompt: str | None = None, max_tokens: int = 100) -> str:
        if "topic detection" in (system_prompt or "").lower():
            return self.detection_response
        if "memory extraction" in (system_prompt or "").lower():
            return self.extraction_response
        return ""


class MockEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class TestEidolon:
    def test_prepare_context_returns_bundle(self):
        storage = MockStorage()
        storage.facts = [Fact(peer_id="user", category="basic", key="name", value="Alex", source="user")]
        llm = MockLLM(detection_response="CONTINUE")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        env = EnvironmentSnapshot(location="home")
        bundle = eidolon.prepare_context(
            message="jep, mach weiter",
            peer_id="user",
            environment=env,
            active_topic_id=1,
        )

        assert isinstance(bundle, ContextBundle)
        assert len(bundle.facts) == 1
        assert bundle.environment is not None
        assert bundle.environment.location == "home"

    def test_prepare_context_new_topic(self):
        storage = MockStorage()
        llm = MockLLM(detection_response="NEW_TOPIC:Bello Features")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        bundle = eidolon.prepare_context(
            message="Lass uns über Bello sprechen",
            peer_id="user",
            environment=EnvironmentSnapshot(),
        )

        assert isinstance(bundle, ContextBundle)
        # NEW_TOPIC detected → label is set
        assert bundle.current_topic_label == "Bello Features"

    def test_detect_topic_continue(self):
        storage = MockStorage()
        llm = MockLLM(detection_response="CONTINUE")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        result = eidolon.detect_topic("jep, mach weiter", active_topic_id=1)
        assert result.type == "CONTINUE"

    def test_detect_topic_new(self):
        storage = MockStorage()
        llm = MockLLM(detection_response="NEW_TOPIC:Bello")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        result = eidolon.detect_topic("Lass uns über Bello sprechen")
        assert result.type == "NEW_TOPIC"
        assert result.label == "Bello"

    def test_retrieve_context(self):
        storage = MockStorage()
        storage.facts = [Fact(peer_id="user", category="basic", key="name", value="Alex", source="user")]
        llm = MockLLM()
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        bundle = eidolon.retrieve_context(
            message="test",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label="Test",
            is_new_topic=True,
        )

        assert len(bundle.facts) == 1
        assert bundle.current_topic_label == "Test"

    def test_store_turn_closes_old_topic(self):
        storage = MockStorage()
        tid = storage.create_topic("Old Topic")
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        eidolon.store_turn(
            user_message="Lass uns über Bello sprechen",
            ai_response="Gerne!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            old_topic_id=tid,
            new_topic_label="Bello Features",
        )

        assert tid in storage.closed_topics

    def test_store_turn_creates_new_topic(self):
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        eidolon.store_turn(
            user_message="Lass uns über Bello sprechen",
            ai_response="Gerne!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            new_topic_label="Bello Features",
        )

        # Should have created a topic
        assert len(storage.topics) == 1
        topic = list(storage.topics.values())[0]
        assert topic["label"] == "Bello Features"

    def test_store_turn_extracts_and_stores_items(self):
        storage = MockStorage()
        llm = MockLLM(
            extraction_response='[{"content": "User wants to discuss Bello", '
            '"item_type": "essence", "source_peer": "user", '
            '"subject_peer": "user", "attribution_type": "self_statement"}]'
        )
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        eidolon.store_turn(
            user_message="Lass uns über Bello sprechen",
            ai_response="Gerne!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            new_topic_label="Bello",
        )

        assert len(storage.stored_items) == 1
        assert storage.stored_items[0].content == "User wants to discuss Bello"
        assert storage.stored_items[0].topic_id is not None

    def test_store_turn_no_topic_change(self):
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        # No old_topic_id and no new_topic_label = no topic change
        eidolon.store_turn(
            user_message="jep",
            ai_response="ok",
            peer_id="user",
            environment=EnvironmentSnapshot(),
        )

        assert len(storage.topics) == 0
        assert len(storage.closed_topics) == 0

    def test_language_parameter(self):
        storage = MockStorage()
        llm = MockLLM()
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder, language="de")
        assert eidolon._language == "de"

    def test_custom_retrieval_weights(self):
        storage = MockStorage()
        llm = MockLLM()
        embedder = MockEmbedder()
        weights = {"semantic": 0.5, "tag_overlap": 0.3, "person": 0.1, "temporal": 0.1}
        eidolon = Eidolon(storage, llm, embedder, retrieval_weights=weights)
        assert eidolon._retriever._weights == weights
