"""Tests for the Eidolon Hermes adapter (updated for new API)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from adapters.hermes.session_state import SessionState
from core.models import ContextBundle, EnvironmentSnapshot, Fact, MemoryItem, RelatedTopic, TopicChange


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
        self.last_peer_id: str | None = None
        self.last_request_id: str | None = None
        self.request_ids_seen: list[str | None] = []

    def get_active_topic(self, peer_id: str) -> int | None:
        self.get_active_topic_calls += 1
        self.last_get_active_topic_peer_id = peer_id
        return self.active_topic_to_resume

    def detect_topic(
        self,
        message: str,
        active_topic_id: int | None = None,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ):
        from core.models import TopicChange
        self.detect_topic_calls += 1
        self.last_message = message
        self.last_active_topic_id = active_topic_id
        self.last_session_id = session_id
        self.last_request_id = request_id
        self.request_ids_seen.append(request_id)
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
        request_id: str | None = None,
    ) -> ContextBundle:
        self.retrieve_context_calls += 1
        self.last_session_id = session_id
        self.last_peer_id = peer_id
        self.last_request_id = request_id
        self.request_ids_seen.append(request_id)
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
        request_id: str | None = None,
    ) -> int | None:
        self.store_turn_calls += 1
        self.last_message = user_message
        self.last_ai_response = ai_response
        self.last_peer_id = peer_id
        self.last_environment = environment
        self.last_old_topic_id = old_topic_id
        self.last_new_topic_label = new_topic_label
        self.last_session_id = session_id
        self.last_request_id = request_id
        self.request_ids_seen.append(request_id)
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

    def test_prefetch_uses_resolved_peer_id(self, provider):
        provider._peer_id = "alice"
        provider.prefetch("hello", session_id="sess_1")
        assert provider._eidolon.last_peer_id == "alice"

    def test_prefetch_caches_topic_change_for_sync_turn(self, provider):
        provider.prefetch("hello", session_id="sess_1")
        session = provider._sessions["sess_1"]
        assert session.pending_topic_change is not None
        assert session.pending_query == "hello"

    def test_prefetch_resolves_empty_session_id_via_fallback(self, provider):
        # Mirrors Hermes' prefetch_all() call site, which currently never
        # forwards session_id (see adapters/hermes/__init__.py comments).
        provider.prefetch("hello", session_id="sess_1")
        provider.prefetch("second message", session_id="")
        assert "" not in provider._sessions
        assert provider._sessions["sess_1"].pending_query == "second message"

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

    def test_sync_turn_uses_resolved_peer_id(self, provider):
        provider._peer_id = "alice"
        provider.sync_turn(
            user_content="Hello",
            assistant_content="Hi!",
            session_id="sess_1",
        )
        assert provider._eidolon.last_peer_id == "alice"

    def test_sync_turn_updates_active_topic(self, provider):
        provider.sync_turn(
            user_content="Hello",
            assistant_content="Hi!",
            session_id="sess_1",
        )
        # store_turn returns 42, so active_topic_id should be 42
        assert provider._sessions["sess_1"].active_topic_id == 42

    def test_sync_turn_without_prior_prefetch_recomputes_topic(self, provider):
        # No cached decision — must fall back to calling detect_topic itself
        # rather than crashing or silently skipping topic detection.
        provider.sync_turn(
            user_content="Hello",
            assistant_content="Hi!",
            session_id="sess_1",
        )
        assert provider._eidolon.detect_topic_calls == 1

    def test_sync_turn_reuses_cached_topic_change_from_prefetch(self, provider):
        provider.prefetch("hello", session_id="sess_1")
        assert provider._eidolon.detect_topic_calls == 1
        provider.sync_turn(
            user_content="hello",
            assistant_content="hi",
            session_id="sess_1",
        )
        # Same message as prefetch's — cache hit, no second LLM call.
        assert provider._eidolon.detect_topic_calls == 1

    def test_sync_turn_recomputes_when_message_differs_from_cache(self, provider):
        provider.prefetch("hello", session_id="sess_1")
        provider.sync_turn(
            user_content="something else entirely",
            assistant_content="hi",
            session_id="sess_1",
        )
        # Cached decision was for a different message — must not be trusted.
        assert provider._eidolon.detect_topic_calls == 2

    def test_sync_turn_clears_pending_cache_after_use(self, provider):
        provider.prefetch("hello", session_id="sess_1")
        provider.sync_turn(
            user_content="hello",
            assistant_content="hi",
            session_id="sess_1",
        )
        session = provider._sessions["sess_1"]
        assert session.pending_topic_change is None
        assert session.pending_query is None

    def test_prefetch_shares_one_request_id_across_detect_and_retrieve(self, provider):
        provider.prefetch("hello", session_id="sess_1")
        eidolon = provider._eidolon
        assert len(eidolon.request_ids_seen) == 2
        assert eidolon.request_ids_seen[0] is not None
        assert eidolon.request_ids_seen[0] == eidolon.request_ids_seen[1]

    def test_sync_turn_reuses_prefetchs_request_id_on_cache_hit(self, provider):
        provider.prefetch("hello", session_id="sess_1")
        prefetch_request_id = provider._eidolon.request_ids_seen[0]

        provider.sync_turn(
            user_content="hello", assistant_content="hi", session_id="sess_1"
        )

        # store_turn's request_id (the 3rd call recorded) must match the
        # id prefetch's two calls already shared — one id for the turn.
        assert provider._eidolon.request_ids_seen == [
            prefetch_request_id, prefetch_request_id, prefetch_request_id
        ]

    def test_sync_turn_gets_fresh_request_id_on_cache_miss(self, provider):
        # No preceding prefetch() — store_turn() must get request_id=None
        # and mint its own, exactly like today's fallback behavior.
        provider.sync_turn(
            user_content="hello", assistant_content="hi", session_id="sess_1"
        )
        eidolon = provider._eidolon
        # detect_topic (fallback) and store_turn both got request_id=None
        # passed in from the adapter — each mints its own independently
        # inside the real Eidolon core; the mock just records what it
        # was handed.
        assert eidolon.request_ids_seen == [None, None]

    def test_prefetch_clears_pending_request_id_after_sync_turn(self, provider):
        provider.prefetch("hello", session_id="sess_1")
        provider.sync_turn(
            user_content="hello", assistant_content="hi", session_id="sess_1"
        )
        assert provider._sessions["sess_1"].pending_request_id is None

    def test_sessions_are_isolated_by_session_id(self, provider):
        provider.prefetch("hello A", session_id="sess_a")
        provider.sync_turn(
            user_content="hello A", assistant_content="hi", session_id="sess_a"
        )
        provider.prefetch("hello B", session_id="sess_b")

        assert provider._sessions["sess_a"].active_topic_id == 42
        assert provider._sessions["sess_b"].active_topic_id is None
        # sess_a's cache was consumed by its own sync_turn; sess_b's is untouched.
        assert provider._sessions["sess_a"].pending_query is None
        assert provider._sessions["sess_b"].pending_query == "hello B"

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

    def test_handle_search_uses_resolved_peer_id(self, provider):
        provider._peer_id = "alice"
        provider.handle_tool_call("eidolon_search", {"query": "test"})
        assert provider._eidolon.last_peer_id == "alice"

    def test_handle_tool_call_unknown_raises(self, provider):
        with pytest.raises(NotImplementedError):
            provider.handle_tool_call("unknown_tool", {})

    def test_on_session_switch_resets_topic(self, provider):
        provider._sessions["sess_1"] = SessionState(active_topic_id=42)
        provider.on_session_switch("sess_1", reset=True)
        assert provider._sessions["sess_1"].active_topic_id is None

    def test_on_session_switch_carries_topic_forward_on_resume(self, provider):
        provider._sessions["sess_1"] = SessionState(active_topic_id=42)
        provider.on_session_switch("sess_2", parent_session_id="sess_1", reset=False)
        assert provider._sessions["sess_2"].active_topic_id == 42
        assert "sess_1" not in provider._sessions

    def test_on_session_switch_rewound_clears_pending_cache_only(self, provider):
        provider._sessions["sess_1"] = SessionState(
            active_topic_id=42,
            pending_topic_change=TopicChange(type="NEW_TOPIC", label="X"),
            pending_query="hi",
        )
        provider.on_session_switch("sess_1", rewound=True)
        session = provider._sessions["sess_1"]
        assert session.pending_topic_change is None
        assert session.pending_query is None
        assert session.active_topic_id == 42

    def test_on_session_switch_updates_current_session_fallback(self, provider):
        provider.on_session_switch("sess_2", reset=True)
        provider.prefetch("hello", session_id="")
        assert provider._sessions["sess_2"].pending_query == "hello"

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

        assert p._sessions["sess_1"].active_topic_id == 7
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

        assert p._sessions["sess_1"].active_topic_id is None

    def test_initialize_falls_back_to_user_when_no_identity_kwargs(self, tmp_path):
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

        assert p._peer_id == "user"

    def test_initialize_resolves_peer_id_from_user_id(self, tmp_path):
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

        with patch("adapters.hermes.ConfigLoader") as MockConfigLoader, \
             patch("adapters.hermes.ProviderFactory") as MockProviderFactory, \
             patch("adapters.hermes.Eidolon", return_value=mock_eidolon):
            MockConfigLoader.return_value.load.return_value = mock_config
            factory = MockProviderFactory.return_value
            factory.create_storage.return_value = MockStore()
            factory.create_llm.return_value = MagicMock()
            factory.create_embedder.return_value = MagicMock()

            p = EidolonMemoryProvider()
            p.initialize(
                session_id="sess_1", hermes_home=str(tmp_path), user_id="platform-123"
            )

        assert p._peer_id == "platform-123"

    def test_initialize_prefers_user_id_alt_over_user_id(self, tmp_path):
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

        with patch("adapters.hermes.ConfigLoader") as MockConfigLoader, \
             patch("adapters.hermes.ProviderFactory") as MockProviderFactory, \
             patch("adapters.hermes.Eidolon", return_value=mock_eidolon):
            MockConfigLoader.return_value.load.return_value = mock_config
            factory = MockProviderFactory.return_value
            factory.create_storage.return_value = MockStore()
            factory.create_llm.return_value = MagicMock()
            factory.create_embedder.return_value = MagicMock()

            p = EidolonMemoryProvider()
            p.initialize(
                session_id="sess_1",
                hermes_home=str(tmp_path),
                user_id="u1",
                user_id_alt="alt1",
            )

        assert p._peer_id == "alt1"

    def test_initialize_seeds_current_session_fallback(self, tmp_path):
        """An empty-session_id prefetch() call right after initialize() must
        still land in the session initialize() was called with — this is
        the workaround for Hermes' prefetch_all() call site not forwarding
        session_id at all."""
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

        p.prefetch("hello", session_id="")
        assert "" not in p._sessions
        assert p._sessions["sess_1"].pending_query == "hello"

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
