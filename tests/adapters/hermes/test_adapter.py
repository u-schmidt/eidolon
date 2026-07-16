"""Tests for the Eidolon Hermes adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.models import ContextBundle, EnvironmentSnapshot, Fact, MemoryItem, RelatedTopic


class MockEidolon:
    """Mock Eidolon core for adapter testing."""

    def __init__(self) -> None:
        self.last_message: str | None = None
        self.last_peer_id: str | None = None
        self.last_environment: EnvironmentSnapshot | None = None
        self.last_active_topic_id: int | None = None
        self.last_ai_response: str | None = None
        self.last_old_topic_id: int | None = None
        self.last_new_topic_label: str | None = None
        self.prepare_context_calls = 0
        self.store_turn_calls = 0

    def prepare_context(
        self,
        message: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        active_topic_id: int | None = None,
    ) -> ContextBundle:
        self.prepare_context_calls += 1
        self.last_message = message
        self.last_peer_id = peer_id
        self.last_environment = environment
        self.last_active_topic_id = active_topic_id
        return ContextBundle(
            current_topic_label="Test Topic",
            facts=[Fact(peer_id="user", category="basic", key="name", value="Alex", source="user")],
            related_topics=[
                RelatedTopic(topic_id=1, summary="Previous conversation", tags=["test"])
            ],
            memory_items=[
                MemoryItem(content="User said something", item_type="essence",
                          source_peer="user", subject_peer="user", attribution_type="self_statement")
            ],
        )

    def detect_topic(self, message: str, active_topic_id: int | None = None):
        from core.models import TopicChange
        return TopicChange(type="NEW_TOPIC", label="Test Topic")

    def retrieve_context(
        self,
        message: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        topic_label: str | None = None,
        is_new_topic: bool = False,
    ) -> ContextBundle:
        return ContextBundle(current_topic_label=topic_label)

    def store_turn(
        self,
        user_message: str,
        ai_response: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        old_topic_id: int | None = None,
        new_topic_label: str | None = None,
    ) -> None:
        self.store_turn_calls += 1
        self.last_message = user_message
        self.last_ai_response = ai_response
        self.last_peer_id = peer_id
        self.last_environment = environment
        self.last_old_topic_id = old_topic_id
        self.last_new_topic_label = new_topic_label


class MockStore:
    """Mock SQLite store for adapter testing."""

    def __init__(self) -> None:
        self.facts: list[Fact] = []
        self._conn = MagicMock()

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def get_facts(self, peer_id: str) -> list[Fact]:
        return self.facts


class TestEidolonMemoryProvider:
    """Tests for the EidolonMemoryProvider (Full Mode)."""

    @pytest.fixture
    def provider(self):
        from adapters.hermes import EidolonMemoryProvider
        p = EidolonMemoryProvider(config={
            "db_path": "/tmp/test_eidolon.db",
            "language": "en",
        })
        # Inject mock Eidolon core
        p._eidolon = MockEidolon()
        p._store = MockStore()
        return p

    def test_initialize(self):
        from adapters.hermes import EidolonMemoryProvider
        p = EidolonMemoryProvider()
        # Without Hermes installed, initialize should not crash
        # but Eidolon won't be set up (Hermes imports fail)
        # This just tests the code path doesn't throw
        assert p._eidolon is None

    def test_prefetch_returns_facts(self, provider):
        provider._store.facts = [
            Fact(peer_id="user", category="basic", key="name", value="Alex", source="user"),
        ]
        result = provider.prefetch(session_id="sess_1", peer_id="user")
        assert result["eidolon_ready"] is True
        assert len(result["facts"]) == 1
        assert result["facts"][0]["key"] == "name"
        assert result["facts"][0]["value"] == "Alex"

    def test_prefetch_resets_active_topic(self, provider):
        provider._active_topic_id = 42
        provider.prefetch(session_id="sess_1", peer_id="user")
        assert provider._active_topic_id is None

    def test_sync_turn_calls_prepare_context(self, provider):
        result = provider.sync_turn(
            user_message="Lass uns über Bello sprechen",
            session_id="sess_1",
            peer_id="user",
        )
        assert provider._eidolon.prepare_context_calls == 1
        assert provider._eidolon.last_message == "Lass uns über Bello sprechen"
        assert provider._eidolon.last_peer_id == "user"
        assert "eidolon_context" in result
        assert result["eidolon_context"]["current_topic"] == "Test Topic"

    def test_sync_turn_with_environment(self, provider):
        result = provider.sync_turn(
            user_message="Hello",
            session_id="sess_1",
            peer_id="user",
            environment={"location": "home", "situation": "morning"},
        )
        env = provider._eidolon.last_environment
        assert env is not None
        assert env.location == "home"
        assert env.situation == "morning"

    def test_sync_turn_with_active_topic(self, provider):
        provider._active_topic_id = 42
        provider.sync_turn(
            user_message="Hello",
            session_id="sess_1",
            peer_id="user",
        )
        assert provider._eidolon.last_active_topic_id == 42

    def test_sync_turn_sets_last_topic_label(self, provider):
        provider.sync_turn(
            user_message="Hello",
            session_id="sess_1",
            peer_id="user",
        )
        assert provider._last_topic_label == "Test Topic"

    def test_store_turn_called_after_sync(self, provider):
        # Simulate a full turn cycle
        provider.sync_turn(
            user_message="Lass uns über Bello sprechen",
            session_id="sess_1",
            peer_id="user",
        )
        provider.store_turn(
            user_message="Lass uns über Bello sprechen",
            ai_response="Gerne! Was möchtest du wissen?",
            session_id="sess_1",
            peer_id="user",
        )
        assert provider._eidolon.store_turn_calls == 1
        assert provider._eidolon.last_ai_response == "Gerne! Was möchtest du wissen?"

    def test_store_turn_with_topic_change(self, provider):
        provider._active_topic_id = 1
        provider._last_topic_label = "New Topic"
        provider.store_turn(
            user_message="Lass uns über was neues sprechen",
            ai_response="Klar!",
            session_id="sess_1",
            peer_id="user",
        )
        assert provider._eidolon.last_old_topic_id == 1
        assert provider._eidolon.last_new_topic_label == "New Topic"

    def test_store_turn_without_topic_change(self, provider):
        provider._active_topic_id = 1
        # No _last_topic_label set = no topic change
        provider.store_turn(
            user_message="jep",
            ai_response="ok",
            session_id="sess_1",
            peer_id="user",
        )
        assert provider._eidolon.last_old_topic_id is None
        assert provider._eidolon.last_new_topic_label is None

    def test_get_tool_schemas(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "search_memory"
        assert "query" in schemas[0]["parameters"]["properties"]

    def test_build_environment_empty(self, provider):
        env = provider._build_environment(None)
        assert env.location is None
        assert env.situation is None

    def test_build_environment_with_data(self, provider):
        env = provider._build_environment({
            "location": "kitchen",
            "time": "morning",
            "situation": "cooking",
            "rules": ["no phone"],
        })
        assert env.location == "kitchen"
        assert env.situation == "cooking"
        assert env.rules == ["no phone"]

    def test_bundle_to_dict(self, provider):
        bundle = ContextBundle(
            current_topic_label="Test",
            facts=[Fact(peer_id="user", category="basic", key="name", value="Alex", source="user")],
            related_topics=[RelatedTopic(topic_id=1, summary="Old topic", tags=["old"])],
            memory_items=[MemoryItem(content="Remember this", item_type="essence",
                                    source_peer="user", subject_peer="user",
                                    attribution_type="self_statement")],
        )
        result = provider._bundle_to_dict(bundle)
        assert result["eidolon_context"]["current_topic"] == "Test"
        assert len(result["eidolon_context"]["facts"]) == 1
        assert len(result["eidolon_context"]["related_topics"]) == 1
        assert len(result["eidolon_context"]["memory_items"]) == 1


class TestEidolonTopicHook:
    """Tests for the EidolonTopicHook (Topic-Only Mode)."""

    @pytest.fixture
    def hook(self):
        from adapters.hermes.llm_hook import EidolonTopicHook
        h = EidolonTopicHook(config={
            "db_path": "/tmp/test_eidolon_topic.db",
            "language": "en",
        })
        # Inject mock Eidolon core
        h._eidolon = MockEidolon()
        h._store = MockStore()
        return h

    def test_initialize(self):
        from adapters.hermes.llm_hook import EidolonTopicHook
        h = EidolonTopicHook()
        # Without Hermes installed, should not crash
        assert h._eidolon is None

    def test_pre_llm_call_returns_topic_context(self, hook):
        result = hook.pre_llm_call(
            user_message="Hello",
            session_id="sess_1",
            peer_id="user",
        )
        assert "eidolon_topics" in result
        assert result["eidolon_topics"]["current_topic"] == "Test Topic"

    def test_pre_llm_call_sets_last_topic_label(self, hook):
        hook.pre_llm_call(
            user_message="Hello",
            session_id="sess_1",
            peer_id="user",
        )
        assert hook._last_topic_label == "Test Topic"

    def test_pre_llm_call_with_environment(self, hook):
        result = hook.pre_llm_call(
            user_message="Hello",
            session_id="sess_1",
            peer_id="user",
            environment={"location": "office"},
        )
        assert "eidolon_topics" in result

    def test_post_llm_call_stores_turn(self, hook):
        hook._last_topic_label = "New Topic"
        hook._last_topic_change = MagicMock()
        hook._last_topic_change.type = "NEW_TOPIC"
        hook._active_topic_id = 1

        hook.post_llm_call(
            user_message="Lass uns über was neues sprechen",
            ai_response="Klar!",
            session_id="sess_1",
            peer_id="user",
        )
        assert hook._eidolon.store_turn_calls == 1
        assert hook._eidolon.last_old_topic_id == 1
        assert hook._eidolon.last_new_topic_label == "New Topic"

    def test_post_llm_call_without_topic_change(self, hook):
        # No _last_topic_label set
        hook.post_llm_call(
            user_message="jep",
            ai_response="ok",
            session_id="sess_1",
            peer_id="user",
        )
        assert hook._eidolon.store_turn_calls == 1
        assert hook._eidolon.last_old_topic_id is None
        assert hook._eidolon.last_new_topic_label is None

    def test_build_environment(self, hook):
        env = hook._build_environment({"location": "home"})
        assert env.location == "home"
        assert env.situation is None
