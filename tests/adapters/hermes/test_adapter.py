"""Tests for the Eidolon Hermes adapter (updated for new API)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.models import ContextBundle, EnvironmentSnapshot, Fact, MemoryItem, RelatedTopic


class MockEidolon:
    """Mock Eidolon core for adapter testing (new API)."""

    def __init__(self) -> None:
        self.detect_topic_calls = 0
        self.retrieve_context_calls = 0
        self.store_turn_calls = 0
        self.last_message: str | None = None
        self.last_ai_response: str | None = None
        self.last_peer_id: str | None = None
        self.last_active_topic_id: int | None = None
        self.last_old_topic_id: int | None = None
        self.last_new_topic_label: str | None = None
        self.active_topic_to_resume: int | None = None
        self.get_active_topic_calls = 0
        self.last_get_active_topic_peer_id: str | None = None
        self.last_session_id: str | None = None

    def get_active_topic(self, peer_id: str) -> int | None:
        self.get_active_topic_calls += 1
        self.last_get_active_topic_peer_id = peer_id
        return self.active_topic_to_resume

    def detect_topic(self, message: str, active_topic_id: int | None = None, *, session_id: str | None = None):
        from core.models import TopicChange
        self.detect_topic_calls += 1
        self.last_message = message
        self.last_active_topic_id = active_topic_id
        self.last_session_id = session_id
        return TopicChange(type="NEW_TOPIC", label="Test Topic")

    def retrieve_context(
        self,
        message: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        topic_label: str | None = None,
        is_new_topic: bool = False,
        *,
        session_id: str | None = None,
    ) -> ContextBundle:
        self.retrieve_context_calls += 1
        self.last_session_id = session_id
        return ContextBundle(
            current_topic_label=topic_label or "Test Topic",
            facts=[Fact(peer_id="user", category="basic", key="name", value="Alex", source="user")],
            related_topics=[
                RelatedTopic(topic_id=1, summary="Previous conversation", tags=["test"])
            ],
            memory_items=[
                MemoryItem(content="User said something", item_type="essence",
                          source_peer="user", subject_peer="user", attribution_type="self_statement")
            ],
        )

    def store_turn(
        self,
        user_message: str,
        ai_response: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        old_topic_id: int | None = None,
        new_topic_label: str | None = None,
        *,
        session_id: str | None = None,
    ) -> int | None:
        self.store_turn_calls += 1
        self.last_message = user_message
        self.last_ai_response = ai_response
        self.last_peer_id = peer_id
        self.last_environment = environment
        self.last_old_topic_id = old_topic_id
        self.last_new_topic_label = new_topic_label
        self.last_session_id = session_id
        return 42  # new topic id


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
    """Tests for the EidolonMemoryProvider (new API)."""

    @pytest.fixture
    def provider(self):
        from adapters.hermes import EidolonMemoryProvider
        p = EidolonMemoryProvider()
        # Inject mock Eidolon core + store (skip real initialize)
        p._eidolon = MockEidolon()
        p._store = MockStore()
        p._initialized = True
        return p

    def test_name(self, provider):
        assert provider.name == "eidolon"

    def test_is_available(self, provider):
        # Without Hermes installed, is_available returns False
        assert provider.is_available() is False

    def test_prefetch_returns_string(self, provider):
        result = provider.prefetch("hello", session_id="sess_1")
        assert isinstance(result, str)
        assert "Test Topic" in result
        assert "Alex" in result

    def test_prefetch_empty_when_not_initialized(self):
        from adapters.hermes import EidolonMemoryProvider
        p = EidolonMemoryProvider()
        result = p.prefetch("hello", session_id="sess_1")
        assert result == ""

    def test_prefetch_detects_topic(self, provider):
        provider.prefetch("hello", session_id="sess_1")
        assert provider._eidolon.detect_topic_calls >= 1

    def test_prefetch_retrieves_context(self, provider):
        provider.prefetch("hello", session_id="sess_1")
        assert provider._eidolon.retrieve_context_calls >= 1

    def test_prefetch_passes_session_id_through(self, provider):
        provider.prefetch("hello", session_id="sess_1")
        assert provider._eidolon.last_session_id == "sess_1"

    def test_sync_turn_stores_turn(self, provider):
        provider.sync_turn(
            user_content="Lass uns über Bello sprechen",
            assistant_content="Gerne!",
            session_id="sess_1",
        )
        assert provider._eidolon.store_turn_calls == 1
        assert provider._eidolon.last_message == "Lass uns über Bello sprechen"
        assert provider._eidolon.last_ai_response == "Gerne!"

    def test_sync_turn_passes_session_id_through(self, provider):
        provider.sync_turn(
            user_content="Hello",
            assistant_content="Hi!",
            session_id="sess_1",
        )
        assert provider._eidolon.last_session_id == "sess_1"

    def test_sync_turn_updates_active_topic(self, provider):
        provider.sync_turn(
            user_content="Hello",
            assistant_content="Hi!",
            session_id="sess_1",
        )
        # store_turn returns 42, so active_topic_id should be 42
        assert provider._active_topic_id == 42

    def test_sync_turn_skips_when_not_initialized(self):
        from adapters.hermes import EidolonMemoryProvider
        p = EidolonMemoryProvider()
        # Should not crash
        p.sync_turn(user_content="Hello", assistant_content="Hi", session_id="sess_1")

    def test_queue_prefetch_is_noop(self, provider):
        # Should not crash
        provider.queue_prefetch("hello", session_id="sess_1")

    def test_get_tool_schemas(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "eidolon_search"
        assert "query" in schemas[0]["parameters"]["properties"]

    def test_handle_tool_call_search(self, provider):
        result = provider.handle_tool_call("eidolon_search", {"query": "test"})
        assert "related_topics" in result
        assert "memory_items" in result

    def test_handle_tool_call_unknown_raises(self, provider):
        with pytest.raises(NotImplementedError):
            provider.handle_tool_call("unknown_tool", {})

    def test_on_session_switch_resets_topic(self, provider):
        provider._active_topic_id = 42
        provider.on_session_switch("new_sess", reset=True)
        assert provider._active_topic_id is None

    def test_on_session_switch_keeps_topic(self, provider):
        provider._active_topic_id = 42
        provider.on_session_switch("new_sess", reset=False)
        assert provider._active_topic_id == 42

    def test_shutdown_cleans_up(self, provider):
        provider.shutdown()
        assert provider._eidolon is None
        assert provider._store is None
        assert provider._initialized is False

    def test_shutdown_closes_tracer(self, provider):
        mock_tracer = MagicMock()
        provider._tracer = mock_tracer
        provider.shutdown()
        mock_tracer.close.assert_called_once()
        assert provider._tracer is None

    def test_shutdown_without_tracer_does_not_raise(self, provider):
        provider._tracer = None
        provider.shutdown()  # must not raise

    def test_initialize_resumes_open_topic(self, tmp_path):
        """A restart mid-topic must not silently forget the open topic —
        initialize() should resume it from storage instead of defaulting
        to None."""
        from adapters.hermes import EidolonMemoryProvider

        config_path = tmp_path / "eidolon.yaml"
        config_path.write_text("db_path: ':memory:'\n")

        mock_eidolon = MockEidolon()
        mock_eidolon.active_topic_to_resume = 7

        mock_config = MagicMock()
        mock_config.language = "en"
        mock_config.retrieval.semantic = 0.4
        mock_config.retrieval.tag_overlap = 0.3
        mock_config.retrieval.person = 0.2
        mock_config.retrieval.temporal = 0.1

        with patch("adapters.hermes.ConfigLoader") as MockConfigLoader, \
             patch("adapters.hermes.ProviderFactory") as MockProviderFactory, \
             patch("adapters.hermes.Eidolon", return_value=mock_eidolon):
            MockConfigLoader.return_value.load.return_value = mock_config
            factory = MockProviderFactory.return_value
            factory.create_storage.return_value = MockStore()
            factory.create_llm.return_value = MagicMock()
            factory.create_embedder.return_value = MagicMock()

            p = EidolonMemoryProvider()
            p.initialize(session_id="sess_1", hermes_home=str(tmp_path))

        assert p._active_topic_id == 7
        assert mock_eidolon.last_get_active_topic_peer_id == "user"

    def test_initialize_no_open_topic_stays_none(self, tmp_path):
        from adapters.hermes import EidolonMemoryProvider

        config_path = tmp_path / "eidolon.yaml"
        config_path.write_text("db_path: ':memory:'\n")

        mock_eidolon = MockEidolon()  # active_topic_to_resume defaults to None

        mock_config = MagicMock()
        mock_config.language = "en"
        mock_config.retrieval.semantic = 0.4
        mock_config.retrieval.tag_overlap = 0.3
        mock_config.retrieval.person = 0.2
        mock_config.retrieval.temporal = 0.1

        with patch("adapters.hermes.ConfigLoader") as MockConfigLoader, \
             patch("adapters.hermes.ProviderFactory") as MockProviderFactory, \
             patch("adapters.hermes.Eidolon", return_value=mock_eidolon):
            MockConfigLoader.return_value.load.return_value = mock_config
            factory = MockProviderFactory.return_value
            factory.create_storage.return_value = MockStore()
            factory.create_llm.return_value = MagicMock()
            factory.create_embedder.return_value = MagicMock()

            p = EidolonMemoryProvider()
            p.initialize(session_id="sess_1", hermes_home=str(tmp_path))

        assert p._active_topic_id is None

    def test_initialize_wires_tracer_from_factory(self, tmp_path):
        """create_tracer()'s result must be stored for shutdown() and passed
        through to Eidolon(), not silently dropped."""
        from adapters.hermes import EidolonMemoryProvider

        config_path = tmp_path / "eidolon.yaml"
        config_path.write_text("db_path: ':memory:'\n")

        mock_eidolon = MockEidolon()
        mock_config = MagicMock()
        mock_config.language = "en"
        mock_config.retrieval.semantic = 0.4
        mock_config.retrieval.tag_overlap = 0.3
        mock_config.retrieval.person = 0.2
        mock_config.retrieval.temporal = 0.1
        sentinel_tracer = MagicMock()

        with patch("adapters.hermes.ConfigLoader") as MockConfigLoader, \
             patch("adapters.hermes.ProviderFactory") as MockProviderFactory, \
             patch("adapters.hermes.Eidolon", return_value=mock_eidolon) as MockEidolonClass:
            MockConfigLoader.return_value.load.return_value = mock_config
            factory = MockProviderFactory.return_value
            factory.create_storage.return_value = MockStore()
            factory.create_llm.return_value = MagicMock()
            factory.create_embedder.return_value = MagicMock()
            factory.create_tracer.return_value = sentinel_tracer

            p = EidolonMemoryProvider()
            p.initialize(session_id="sess_1", hermes_home=str(tmp_path))

        assert p._tracer is sentinel_tracer
        assert MockEidolonClass.call_args.kwargs["tracer"] is sentinel_tracer


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
