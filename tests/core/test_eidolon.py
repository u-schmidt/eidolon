"""Tests for the Eidolon core facade."""

from __future__ import annotations

from unittest.mock import MagicMock

from core import Eidolon
from core.models import ContextBundle, EnvironmentSnapshot, Fact, MemoryItem, TopicChange
from tests.mocks import SpyTracer


class MockStorage:
    """Mock storage for facade testing."""

    def __init__(self) -> None:
        self.facts: list[Fact] = []
        self.topics: dict[int, dict] = {}
        self.memory_items: list[MemoryItem] = []
        self._next_topic_id = 1
        self.closed_topics: list[int] = []
        self.stored_items: list[MemoryItem] = []
        self.messages: list[dict] = []
        self.topic_embeddings: dict[int, list[float]] = {}
        # _summarize_topic updates tags via raw SQL (see core/__init__.py) —
        # not modeled in this mock beyond not raising.
        self._conn = MagicMock()

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

    def get_active_topic(self, peer_id: str) -> int | None:
        active = [
            t for t in self.topics.values()
            if t["status"] == "active" and t.get("peer_id") == peer_id
        ]
        return active[-1]["id"] if active else None

    def update_topic_embedding(self, topic_id: int, embedding: list[float]) -> None:
        self.topic_embeddings[topic_id] = embedding

    def search_topics(self, query_embedding=None, tags=None, peer_id=None, limit=5, weights=None,
                       tracer=None, request_id=None):
        return [t for t in self.topics.values() if t["status"] == "closed"][:limit]

    def store_memory_item(self, item: MemoryItem) -> int:
        self.stored_items.append(item)
        return len(self.stored_items)

    def search_memory_items(self, query_embedding=None, peer_id=None, limit=5, weights=None,
                             tracer=None, request_id=None):
        return self.memory_items[:limit]

    def store_message(self, turn_id, role, content, peer_id, topic_id=None):
        msg = {
            "id": len(self.messages) + 1,
            "turn_id": turn_id,
            "role": role,
            "content": content,
            "peer_id": peer_id,
            "topic_id": topic_id,
        }
        self.messages.append(msg)
        return msg["id"]

    def get_messages_by_topic(self, topic_id: int) -> list[dict]:
        return [m for m in self.messages if m["topic_id"] == topic_id]


class MockLLM:
    def __init__(
        self,
        detection_response: str = "CONTINUE",
        extraction_response: str = "[]",
        summary_response: str = "",
        fact_response: str = "[]",
    ):
        self.detection_response = detection_response
        self.extraction_response = extraction_response
        self.summary_response = summary_response
        self.fact_response = fact_response

    def complete(self, prompt: str, system_prompt: str | None = None, max_tokens: int = 100) -> str:
        sp = (system_prompt or "").lower()
        if "topic detection" in sp:
            return self.detection_response
        if "memory extraction" in sp:
            return self.extraction_response
        if "summarizer" in sp:
            return self.summary_response
        if "structured facts" in sp:
            return self.fact_response
        return ""


class MockEmbedder:
    def __init__(self, should_fail: bool = False) -> None:
        self.embedded_texts: list[str] = []
        self.should_fail = should_fail

    def embed(self, text: str) -> list[float]:
        if self.should_fail:
            raise RuntimeError("embedding backend unavailable")
        self.embedded_texts.append(text)
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

    def test_store_turn_closes_old_topic_with_real_summary_when_available(self):
        """store_turn must keep the LLM-generated summary from
        _summarize_topic instead of overwriting it with the
        'Continued to: ...' placeholder."""
        storage = MockStorage()
        tid = storage.create_topic("Old Topic", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded", peer_id="user", topic_id=tid)
        llm = MockLLM(extraction_response="[]", summary_response="Summary: Crowded gym frustration\nTags: fitness")
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

        assert storage.topics[tid]["summary"] == "Crowded gym frustration"

    def test_store_turn_falls_back_to_placeholder_when_summarization_fails(self):
        """If _summarize_topic can't produce a real summary (no messages,
        LLM failure, unparseable response), the topic must still end up
        closed — never stuck 'active' forever."""
        storage = MockStorage()
        tid = storage.create_topic("Old Topic", peer_id="user")
        # No messages stored under this topic -> _summarize_topic returns
        # False without closing it.
        llm = MockLLM(extraction_response="[]", summary_response="")
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

        assert storage.topics[tid]["status"] == "closed"
        assert storage.topics[tid]["summary"] == "Continued to: Bello Features"

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

    def test_summarize_topic_reembeds_with_summary(self):
        """The embedding generated at topic creation is based on the short
        provisional label, not the richer summary produced when the topic
        closes — _summarize_topic must re-embed so semantic retrieval isn't
        stuck matching against stale text forever."""
        storage = MockStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded again", peer_id="user", topic_id=tid)
        storage.store_message(turn_id=1, role="ai", content="That sounds frustrating!", peer_id="ai", topic_id=tid)

        llm = MockLLM(summary_response="Summary: User frustrated about crowded gym\nTags: fitness, gym")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        eidolon._summarize_topic(tid)

        assert tid in storage.topic_embeddings
        assert "User frustrated about crowded gym" in embedder.embedded_texts

    def test_summarize_topic_returns_true_on_success(self):
        storage = MockStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded", peer_id="user", topic_id=tid)
        llm = MockLLM(summary_response="Summary: Crowded gym frustration\nTags: fitness")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        assert eidolon._summarize_topic(tid) is True

    def test_summarize_topic_returns_false_when_no_messages(self):
        storage = MockStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        llm = MockLLM(summary_response="Summary: Crowded gym frustration\nTags: fitness")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        assert eidolon._summarize_topic(tid) is False

    def test_summarize_topic_returns_false_on_exception(self):
        class FailingStorage(MockStorage):
            def get_messages_by_topic(self, topic_id):
                raise RuntimeError("db unavailable")

        storage = FailingStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        llm = MockLLM(summary_response="Summary: x\nTags: y")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        assert eidolon._summarize_topic(tid) is False

    def test_get_active_topic_delegates_to_storage(self):
        storage = MockStorage()
        tid = storage.create_topic("Open Topic", peer_id="user")
        llm = MockLLM()
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        assert eidolon.get_active_topic("user") == tid

    def test_get_active_topic_none_when_no_open_topic(self):
        storage = MockStorage()
        llm = MockLLM()
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        assert eidolon.get_active_topic("user") is None


class TestEidolonTracing:
    """Verifies the tracer actually gets called at the right points with
    the right data — not just that a tracer can be passed in."""

    def test_store_turn_records_turn_complete_event(self):
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.store_turn(
            user_message="Lass uns über Bello sprechen",
            ai_response="Gerne!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            new_topic_label="Bello Features",
            session_id="sess1",
        )

        events = tracer.events_of_type("turn_complete")
        assert len(events) == 1
        event = events[0]
        assert event["session_id"] == "sess1"
        assert event["duration_ms"] is not None
        assert event["payload"]["user_message"] == "Lass uns über Bello sprechen"
        assert event["payload"]["new_topic_label"] == "Bello Features"
        assert event["payload"]["new_topic_id"] is not None

    def test_detect_topic_honors_caller_supplied_request_id(self):
        storage = MockStorage()
        llm = MockLLM()
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.detect_topic("hello", request_id="fixed-id")

        assert {e["request_id"] for e in tracer.events} == {"fixed-id"}

    def test_retrieve_context_honors_caller_supplied_request_id(self):
        storage = MockStorage()
        llm = MockLLM()
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.retrieve_context(
            message="hello", peer_id="user", environment=EnvironmentSnapshot(),
            request_id="fixed-id",
        )

        assert {e["request_id"] for e in tracer.events} == {"fixed-id"}

    def test_store_turn_honors_caller_supplied_request_id(self):
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.store_turn(
            user_message="hi", ai_response="hey", peer_id="user",
            environment=EnvironmentSnapshot(), request_id="fixed-id",
        )

        assert {e["request_id"] for e in tracer.events} == {"fixed-id"}

    def test_detect_topic_and_retrieve_context_share_caller_request_id(self):
        """The core mechanism the Hermes adapter relies on to correlate a
        whole turn: passing the same request_id into both of prefetch()'s
        calls must produce one unified group of trace events, not two."""
        storage = MockStorage()
        llm = MockLLM()
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.detect_topic("hello", request_id="turn-1")
        eidolon.retrieve_context(
            message="hello", peer_id="user", environment=EnvironmentSnapshot(),
            request_id="turn-1",
        )

        assert {e["request_id"] for e in tracer.events} == {"turn-1"}

    def test_store_turn_events_share_request_id(self):
        storage = MockStorage()
        llm = MockLLM(
            extraction_response='[{"content": "fact-worthy", "item_type": "fact", '
            '"source_peer": "user", "subject_peer": "user", "attribution_type": "self_statement"}]',
            fact_response='[{"category": "fitness", "key": "gym_frustration", "value": "crowded gym"}]',
        )
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.store_turn(
            user_message="FitZone ist voll",
            ai_response="Nervig!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            new_topic_label="Bello Features",
        )

        request_ids = {e["request_id"] for e in tracer.events}
        assert len(request_ids) == 1
        assert None not in request_ids

    def test_store_turn_records_topic_label_embedding_event(self):
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.store_turn(
            user_message="Hi",
            ai_response="Hello",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            new_topic_label="Bello Features",
        )

        events = tracer.events_of_type("embedding")
        purposes = [e["payload"]["purpose"] for e in events]
        assert "topic_label" in purposes

    def test_store_turn_records_memory_item_embedding_event(self):
        storage = MockStorage()
        llm = MockLLM(
            extraction_response='[{"content": "User likes hiking", "item_type": "essence", '
            '"source_peer": "user", "subject_peer": "user", "attribution_type": "self_statement"}]'
        )
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.store_turn(
            user_message="I love hiking",
            ai_response="Nice!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
        )

        events = tracer.events_of_type("embedding")
        item_events = [e for e in events if e["payload"]["purpose"] == "memory_item"]
        assert len(item_events) == 1
        assert item_events[0]["payload"]["text"] == "User likes hiking"

    def test_store_turn_records_error_on_topic_embedding_failure(self):
        """An embedding failure must not vanish silently — it should show
        up as an `error` trace event even though store_turn still succeeds
        (embedding failure is non-critical, per the existing except/pass)."""
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder(should_fail=True)
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        new_topic_id = eidolon.store_turn(
            user_message="Hi",
            ai_response="Hello",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            new_topic_label="Bello Features",
        )

        assert new_topic_id is not None  # store_turn still succeeds
        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "topic_label_embedding" for e in error_events)

    def test_store_turn_records_error_on_memory_item_embedding_failure(self):
        storage = MockStorage()
        llm = MockLLM(
            extraction_response='[{"content": "User likes hiking", "item_type": "essence", '
            '"source_peer": "user", "subject_peer": "user", "attribution_type": "self_statement"}]'
        )
        embedder = MockEmbedder(should_fail=True)
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.store_turn(
            user_message="I love hiking",
            ai_response="Nice!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
        )

        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "memory_item_embedding" for e in error_events)

    def test_extract_facts_records_fact_extraction_event(self):
        storage = MockStorage()
        llm = MockLLM(
            fact_response='[{"category": "fitness", "key": "gym_frustration", "value": "crowded gym"}]'
        )
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        fact_item = MemoryItem(
            content="User is frustrated with crowded gym", item_type="fact",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
        )
        eidolon._extract_facts([fact_item], "user", request_id="req1", session_id="sess1")

        events = tracer.events_of_type("fact_extraction")
        assert len(events) == 1
        event = events[0]
        assert event["request_id"] == "req1"
        assert event["session_id"] == "sess1"
        assert event["payload"]["stored_facts"] == [
            {"category": "fitness", "key": "gym_frustration", "value": "crowded gym"}
        ]
        assert storage.facts  # actually persisted too

    def test_extract_facts_records_error_on_malformed_json(self):
        storage = MockStorage()
        llm = MockLLM(fact_response="not json at all")
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        fact_item = MemoryItem(
            content="User is frustrated with crowded gym", item_type="fact",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
        )
        eidolon._extract_facts([fact_item], "user")

        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "fact_extraction" for e in error_events)
        assert tracer.events_of_type("fact_extraction") == []  # never got that far
        assert storage.facts == []

    def test_summarize_topic_records_topic_summarization_event(self):
        storage = MockStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded", peer_id="user", topic_id=tid)

        llm = MockLLM(summary_response="Summary: Crowded gym frustration\nTags: fitness, gym")
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon._summarize_topic(tid, request_id="req1", session_id="sess1")

        events = tracer.events_of_type("topic_summarization")
        assert len(events) == 1
        event = events[0]
        assert event["request_id"] == "req1"
        assert event["session_id"] == "sess1"
        assert event["payload"]["topic_id"] == tid
        assert event["payload"]["message_count"] == 1
        assert event["payload"]["summary"] == "Crowded gym frustration"
        assert event["payload"]["tags"] == ["fitness", "gym"]

    def test_summarize_topic_records_error_on_failure(self):
        class FailingStorage(MockStorage):
            def get_messages_by_topic(self, topic_id):
                raise RuntimeError("db unavailable")

        storage = FailingStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        llm = MockLLM(summary_response="Summary: x\nTags: y")
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon._summarize_topic(tid)  # must not raise

        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "topic_summarization" for e in error_events)

    def test_default_tracer_is_noop(self):
        # No tracer passed anywhere — must not raise.
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        eidolon.store_turn(
            user_message="Hi",
            ai_response="Hello",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            new_topic_label="Test",
        )
