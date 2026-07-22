"""Tests for the Eidolon Hermes adapter (updated for new API)."""

from __future__ import annotations

import json
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
        self.last_retrieve_context_active_topic_id: int | None = None
        self.last_old_topic_id: int | None = None
        self.last_new_topic_label: str | None = None
        self.last_turn_id: int | None = None
        self.active_topic_to_resume: int | None = None
        self.active_topics_by_session: dict[str | None, int] = {}
        self.get_active_topic_calls = 0
        self.last_get_active_topic_peer_id: str | None = None
        self.last_get_active_topic_session_id: str | None = None
        self.get_active_topic_session_ids_seen: list[str | None] = []
        self.last_session_id: str | None = None
        self.last_peer_id: str | None = None
        self.last_request_id: str | None = None
        self.request_ids_seen: list[str | None] = []
        self.correct_fact_calls = 0
        self.last_correct_fact_args: tuple | None = None
        self.inspect_cluster_calls = 0
        self.last_inspect_cluster_id: int | None = None
        self.cluster_topics: list[dict] = []

    def get_active_topic(self, peer_id: str, session_id: str | None = None) -> int | None:
        self.get_active_topic_calls += 1
        self.last_get_active_topic_peer_id = peer_id
        self.last_get_active_topic_session_id = session_id
        self.get_active_topic_session_ids_seen.append(session_id)
        if session_id in self.active_topics_by_session:
            return self.active_topics_by_session[session_id]
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
        active_topic_id: int | None = None,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> ContextBundle:
        self.retrieve_context_calls += 1
        self.last_session_id = session_id
        self.last_peer_id = peer_id
        self.last_request_id = request_id
        self.last_environment = environment
        self.last_retrieve_context_active_topic_id = active_topic_id
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
        turn_id: int | None = None,
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
        self.last_turn_id = turn_id
        self.last_old_topic_id = old_topic_id
        self.last_new_topic_label = new_topic_label
        self.last_session_id = session_id
        self.last_request_id = request_id
        self.request_ids_seen.append(request_id)
        return 42  # new topic id

    def correct_fact(
        self,
        peer_id: str,
        category: str,
        key: str,
        value: str | None = None,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.correct_fact_calls += 1
        self.last_correct_fact_args = (peer_id, category, key, value)

    def inspect_cluster(self, cluster_id: int) -> list[dict]:
        self.inspect_cluster_calls += 1
        self.last_inspect_cluster_id = cluster_id
        return self.cluster_topics


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

    def test_prefetch_passes_session_active_topic_id_to_retrieve_context(self, provider):
        provider._sessions["sess_1"] = SessionState(active_topic_id=7)
        provider.prefetch("hello", session_id="sess_1")
        assert provider._eidolon.last_retrieve_context_active_topic_id == 7

    def test_prefetch_populates_real_time_in_environment(self, provider):
        provider.prefetch("hello", session_id="sess_1")
        env = provider._eidolon.last_environment
        assert env.time is not None
        assert env.day_of_week is not None

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
        assert len(schemas) == 3
        names = [s["name"] for s in schemas]
        assert "eidolon_search" in names
        assert "eidolon_correct_fact" in names
        assert "eidolon_inspect_cluster" in names
        search_schema = next(s for s in schemas if s["name"] == "eidolon_search")
        assert "query" in search_schema["parameters"]["properties"]
        correct_schema = next(s for s in schemas if s["name"] == "eidolon_correct_fact")
        assert set(correct_schema["parameters"]["required"]) == {"category", "key"}
        assert "value" in correct_schema["parameters"]["properties"]
        inspect_schema = next(s for s in schemas if s["name"] == "eidolon_inspect_cluster")
        assert inspect_schema["parameters"]["required"] == ["cluster_id"]

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

    def test_handle_correct_fact_upserts(self, provider):
        result = provider.handle_tool_call(
            "eidolon_correct_fact", {"category": "location", "key": "city", "value": "Berlin"}
        )
        assert provider._eidolon.correct_fact_calls == 1
        assert provider._eidolon.last_correct_fact_args == ("user", "location", "city", "Berlin")
        payload = json.loads(result)
        assert payload["status"] == "ok"
        assert payload["action"] == "updated"

    def test_handle_correct_fact_deletes_when_value_omitted(self, provider):
        result = provider.handle_tool_call(
            "eidolon_correct_fact", {"category": "lifestyle", "key": "smoking"}
        )
        assert provider._eidolon.last_correct_fact_args == ("user", "lifestyle", "smoking", None)
        payload = json.loads(result)
        assert payload["action"] == "deleted"

    def test_handle_correct_fact_uses_resolved_peer_id(self, provider):
        provider._peer_id = "alice"
        provider.handle_tool_call("eidolon_correct_fact", {"category": "location", "key": "city", "value": "Berlin"})
        assert provider._eidolon.last_correct_fact_args[0] == "alice"

    def test_handle_correct_fact_missing_category_returns_error(self, provider):
        result = provider.handle_tool_call("eidolon_correct_fact", {"key": "city", "value": "Berlin"})
        assert provider._eidolon.correct_fact_calls == 0
        assert "error" in json.loads(result)

    def test_handle_correct_fact_missing_key_returns_error(self, provider):
        result = provider.handle_tool_call("eidolon_correct_fact", {"category": "location", "value": "Berlin"})
        assert provider._eidolon.correct_fact_calls == 0
        assert "error" in json.loads(result)

    def test_handle_correct_fact_when_not_initialized(self):
        from adapters.hermes import EidolonMemoryProvider
        p = EidolonMemoryProvider()
        result = p.handle_tool_call("eidolon_correct_fact", {"category": "location", "key": "city", "value": "Berlin"})
        assert "error" in json.loads(result)

    def test_handle_inspect_cluster_returns_topics(self, provider):
        provider._eidolon.cluster_topics = [
            {"label": "Considering a bike", "summary": "User is thinking about a new bike", "closed_at": "2025-09-21"},
            {"label": "Bought a bike", "summary": "User bought a bike near Fischmarkt", "closed_at": "2025-10-02"},
        ]
        result = provider.handle_tool_call("eidolon_inspect_cluster", {"cluster_id": 17})
        assert provider._eidolon.inspect_cluster_calls == 1
        assert provider._eidolon.last_inspect_cluster_id == 17
        payload = json.loads(result)
        assert payload["cluster_id"] == 17
        assert len(payload["topics"]) == 2
        assert payload["topics"][0]["label"] == "Considering a bike"

    def test_handle_inspect_cluster_missing_id_returns_error(self, provider):
        result = provider.handle_tool_call("eidolon_inspect_cluster", {})
        assert provider._eidolon.inspect_cluster_calls == 0
        assert "error" in json.loads(result)

    def test_handle_inspect_cluster_non_integer_id_returns_error(self, provider):
        result = provider.handle_tool_call("eidolon_inspect_cluster", {"cluster_id": "not-a-number"})
        assert provider._eidolon.inspect_cluster_calls == 0
        assert "error" in json.loads(result)

    def test_handle_inspect_cluster_when_not_initialized(self):
        from adapters.hermes import EidolonMemoryProvider
        p = EidolonMemoryProvider()
        result = p.handle_tool_call("eidolon_inspect_cluster", {"cluster_id": 17})
        assert "error" in json.loads(result)

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

    def test_on_session_switch_resets_turn_number(self, provider):
        provider._sessions["sess_1"] = SessionState(turn_number=48)
        provider.on_session_switch("sess_1", reset=True)
        assert provider._sessions["sess_1"].turn_number == 0

    def test_on_session_switch_carries_turn_number_forward_on_resume(self, provider):
        provider._sessions["sess_1"] = SessionState(turn_number=5)
        provider.on_session_switch("sess_2", parent_session_id="sess_1", reset=False)
        assert provider._sessions["sess_2"].turn_number == 5

    def test_on_turn_start_caches_turn_number_on_current_session(self, provider):
        # Mirrors how Hermes actually calls this: no session_id at all, so
        # the adapter has to fall back to whichever session prefetch()/
        # sync_turn() most recently resolved as current.
        provider.prefetch("hello", session_id="sess_1")
        provider.on_turn_start(3, "hello")
        assert provider._sessions["sess_1"].turn_number == 3

    def test_on_turn_start_no_op_when_not_initialized(self):
        from adapters.hermes import EidolonMemoryProvider
        p = EidolonMemoryProvider()
        p.on_turn_start(3, "hello")  # must not raise
        assert p._sessions == {}

    def test_on_turn_start_failure_is_caught_and_logged(self, provider, caplog):
        # _get_or_create_session raising must not propagate — Hermes' own
        # error handling for this hook only logs at debug (easy to miss),
        # so the adapter's own try/except + warning is what actually
        # surfaces a bug here.
        provider._get_or_create_session = MagicMock(side_effect=RuntimeError("boom"))
        with caplog.at_level("WARNING"):
            provider.on_turn_start(3, "hello")  # must not raise
        assert "on_turn_start failed" in caplog.text

    def test_sync_turn_passes_cached_turn_number_as_turn_id(self, provider):
        provider._sessions["sess_1"] = SessionState(turn_number=7)
        provider.sync_turn(
            user_content="hi", assistant_content="hello", session_id="sess_1",
        )
        assert provider._eidolon.last_turn_id == 7

    def test_sync_turn_defaults_turn_id_to_zero_for_a_fresh_session(self, provider):
        provider.sync_turn(
            user_content="hi", assistant_content="hello", session_id="sess_1",
        )
        assert provider._eidolon.last_turn_id == 0

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

    def test_concurrent_sessions_resume_independent_active_topics(self, provider):
        """A peer's concurrent sessions (web, Telegram, CLI, ...) must each
        resume their own topic thread — not share one, and not clobber
        each other. _get_or_create_session() resumes per (peer, session),
        not just once at plugin load (see adapters/hermes/__init__.py)."""
        provider._eidolon.active_topics_by_session = {"web": 5, "telegram": 9}

        web_session = provider._get_or_create_session("web")
        telegram_session = provider._get_or_create_session("telegram")

        assert web_session.active_topic_id == 5
        assert telegram_session.active_topic_id == 9
        # Each resume call was scoped to its own session_id, not a shared
        # "current session" fallback.
        assert provider._eidolon.get_active_topic_session_ids_seen == ["web", "telegram"]

    def test_second_session_first_message_does_not_touch_first_sessions_topic(self, provider):
        """Getting a brand-new session must not perturb another session's
        already-tracked state."""
        provider._sessions["web"] = SessionState(active_topic_id=5)
        provider._eidolon.active_topics_by_session = {"telegram": None}

        provider._get_or_create_session("telegram")

        assert provider._sessions["web"].active_topic_id == 5

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

    def test_initialize_records_error_and_reraises_when_storage_fails(self, tmp_path, caplog):
        """A storage/llm/embedder construction failure (e.g. 'database is
        locked') must not vanish silently — it's recorded to the tracer
        (built first for exactly this reason) AND logged via the standard
        logging module (visible even when debug.enabled is off, unlike the
        tracer path), before the same exception still propagates unchanged
        so Hermes' own error handling is unaffected."""
        from adapters.hermes import EidolonMemoryProvider

        config_path = tmp_path / "eidolon.yaml"
        config_path.write_text("db_path: ':memory:'\n")

        mock_config = MagicMock()
        sentinel_tracer = MagicMock()
        boom = RuntimeError("database is locked")

        with patch("adapters.hermes.ConfigLoader") as MockConfigLoader, \
             patch("adapters.hermes.ProviderFactory") as MockProviderFactory:
            MockConfigLoader.return_value.load.return_value = mock_config
            factory = MockProviderFactory.return_value
            factory.create_tracer.return_value = sentinel_tracer
            factory.create_storage.side_effect = boom

            p = EidolonMemoryProvider()
            with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="database is locked"):
                p.initialize(session_id="sess_1", hermes_home=str(tmp_path))

        assert "Eidolon initialize failed" in caplog.text

        sentinel_tracer.record.assert_called_once_with(
            "error", {"step": "initialize", "exception": repr(boom)}, session_id="sess_1",
        )
        # The tracer itself must still be reachable for shutdown() even
        # though initialize() didn't complete.
        assert p._tracer is sentinel_tracer


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

    def test_pre_llm_call_passes_active_topic_id_to_retrieve_context(self, hook):
        hook._active_topic_id = 3
        hook.pre_llm_call(
            user_message="Hello",
            session_id="sess_1",
            peer_id="user",
        )
        assert hook._eidolon.last_retrieve_context_active_topic_id == 3

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
