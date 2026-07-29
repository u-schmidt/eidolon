"""Backend-agnostic EidolonClient contract tests.

Runs the same assertions against both the in-process ``Eidolon`` facade
and an ``EidolonHttpClient`` talking to that same kind of instance over
a real FastAPI app (the ``make_eidolon_client`` fixture in
tests/conftest.py is parametrized over both) — same insurance
tests/core/test_storage_provider_contract.py already gives the two
storage backends, applied one layer up: since ``EidolonHttpClient``
isn't an independent reimplementation of core logic but a thin
transport wrapper around the exact same ``Eidolon``, a behavior
difference here would mean the HTTP (de)serialization in
client/wire.py or server/app.py lost or mangled something, not that the
two "implementations" genuinely disagree.

Backend-specific behavior (server auth, wire-format edge cases) lives
elsewhere — this suite only covers what's observable through the
EidolonClient contract itself.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from core import Eidolon
from core.models import EnvironmentSnapshot, Fact, MemoryItem
from tests.mocks import EidolonMockEmbedder, EidolonMockLLM, MockStorage


def _make_eidolon(**llm_kwargs) -> tuple[Eidolon, MockStorage]:
    storage = MockStorage()
    eidolon = Eidolon(storage=storage, llm=EidolonMockLLM(**llm_kwargs), embedder=EidolonMockEmbedder())
    return eidolon, storage


class TestPrepareContext:
    def test_new_topic_returns_topic_change_and_bundle(self, make_eidolon_client):
        eidolon, storage = _make_eidolon(detection_response="NEW_TOPIC:Bello")
        storage.facts.append(
            Fact(workspace_id="user", category="basic", key="name", value="Alex", source="user")
        )
        client = make_eidolon_client(eidolon)

        result = client.prepare_context(
            message="Let's talk about Bello", workspace_id="user",
            environment=EnvironmentSnapshot(situation="conversation"),
        )

        assert result.topic_change.type == "NEW_TOPIC"
        assert result.topic_change.label == "Bello"
        assert result.bundle.current_topic_label == "Bello"
        assert len(result.bundle.facts) == 1
        assert result.bundle.facts[0].value == "Alex"

    def test_continue_resolves_current_topic_label_from_summary(self, make_eidolon_client):
        eidolon, storage = _make_eidolon(detection_response="CONTINUE")
        tid = storage.create_topic("General", workspace_id="user")
        storage.topics[tid]["summary"] = "Talking about Bello's diet"
        client = make_eidolon_client(eidolon)

        result = client.prepare_context(
            message="mach weiter", workspace_id="user",
            environment=EnvironmentSnapshot(), active_topic_id=tid,
        )

        assert result.topic_change.type == "CONTINUE"
        assert result.bundle.current_topic_label == "Talking about Bello's diet"

    def test_environment_round_trips_including_datetime(self, make_eidolon_client):
        eidolon, _ = _make_eidolon()
        client = make_eidolon_client(eidolon)
        now = datetime.now(UTC).replace(microsecond=0)

        result = client.prepare_context(
            message="hi", workspace_id="user",
            environment=EnvironmentSnapshot(situation="conversation", time=now, day_of_week="Monday"),
        )

        assert result.bundle.environment is not None
        assert result.bundle.environment.situation == "conversation"
        assert result.bundle.environment.day_of_week == "Monday"
        assert result.bundle.environment.time == now


class TestDetectTopic:
    def test_continue(self, make_eidolon_client):
        eidolon, _ = _make_eidolon(detection_response="CONTINUE")
        client = make_eidolon_client(eidolon)

        result = client.detect_topic("jep, mach weiter", active_topic_id=1)
        assert result.type == "CONTINUE"
        assert result.label is None

    def test_new_topic(self, make_eidolon_client):
        eidolon, _ = _make_eidolon(detection_response="NEW_TOPIC:Bello")
        client = make_eidolon_client(eidolon)

        result = client.detect_topic("Lass uns über Bello sprechen")
        assert result.type == "NEW_TOPIC"
        assert result.label == "Bello"


class TestRetrieveContext:
    def test_bundle_contents_round_trip(self, make_eidolon_client):
        eidolon, storage = _make_eidolon()
        storage.facts.append(Fact(workspace_id="user", category="basic", key="name", value="Alex", source="user"))
        client = make_eidolon_client(eidolon)

        bundle = client.retrieve_context(
            message="test", workspace_id="user", environment=EnvironmentSnapshot(),
            topic_label="Test", is_new_topic=True,
        )

        assert len(bundle.facts) == 1
        assert bundle.facts[0].category == "basic"
        assert bundle.facts[0].importance == "medium"
        assert bundle.current_topic_label == "Test"


class TestStoreTurnAndGetActiveTopic:
    def test_store_turn_creates_topic_visible_via_get_active_topic(self, make_eidolon_client):
        eidolon, _ = _make_eidolon()
        client = make_eidolon_client(eidolon)

        assert client.get_active_topic("user") is None

        new_topic_id = client.store_turn(
            user_message="Let's talk about Bello", ai_response="Sure!", workspace_id="user",
            environment=EnvironmentSnapshot(), new_topic_label="Bello",
        )

        assert isinstance(new_topic_id, int)
        assert client.get_active_topic("user") == new_topic_id

    def test_store_turn_closes_old_topic_on_new_topic_label(self, make_eidolon_client):
        eidolon, storage = _make_eidolon()
        client = make_eidolon_client(eidolon)
        old_id = storage.create_topic("Old topic", workspace_id="user")

        client.store_turn(
            user_message="new subject", ai_response="ok", workspace_id="user",
            environment=EnvironmentSnapshot(), old_topic_id=old_id, new_topic_label="New topic",
        )

        assert old_id in storage.closed_topics


class TestConsolidateFacts:
    def test_runs_without_error_and_advances_checkpoint(self, make_eidolon_client):
        eidolon, storage = _make_eidolon(consolidation_response='{"upserts": [], "deletes": []}')
        client = make_eidolon_client(eidolon)

        # No-op with nothing accumulated — must not raise either way.
        client.consolidate_facts("user")
        assert storage.get_consolidation_checkpoint("user") == 0


class TestCloseStaleTopics:
    def test_closes_only_flagged_stale_topics(self, make_eidolon_client):
        eidolon, storage = _make_eidolon(summary_response="")
        client = make_eidolon_client(eidolon)
        stale_id = storage.create_topic("Stale", workspace_id="user")
        fresh_id = storage.create_topic("Fresh", workspace_id="user")
        storage.stale_topic_ids.add(stale_id)

        closed = client.close_stale_topics(datetime.now(UTC) - timedelta(hours=1), workspace_id="user")

        assert closed == [stale_id]
        assert fresh_id not in storage.closed_topics


class TestCorrectFact:
    def test_upsert_then_delete(self, make_eidolon_client):
        eidolon, storage = _make_eidolon()
        client = make_eidolon_client(eidolon)

        client.correct_fact("user", "location", "city", "Berlin")
        assert storage.get_facts("user")[0].value == "Berlin"
        assert storage.get_facts("user")[0].source == "user"

        client.correct_fact("user", "location", "city", None)
        assert storage.get_facts("user") == []


class TestInspectCluster:
    def test_returns_member_topics(self, make_eidolon_client):
        eidolon, storage = _make_eidolon()
        client = make_eidolon_client(eidolon)
        cid = storage.create_cluster("user", "Bike shopping", "Recurring bike talk")
        tid = storage.create_topic("Considering a bike", workspace_id="user")
        storage.assign_topic_to_cluster(tid, cid)

        topics = client.inspect_cluster(cid)

        assert len(topics) == 1
        assert topics[0]["label"] == "Considering a bike"

    def test_empty_for_unknown_cluster(self, make_eidolon_client):
        eidolon, _ = _make_eidolon()
        client = make_eidolon_client(eidolon)
        assert client.inspect_cluster(999) == []


class TestInspectTopic:
    def test_returns_memory_items(self, make_eidolon_client):
        eidolon, storage = _make_eidolon()
        client = make_eidolon_client(eidolon)
        tid = storage.create_topic("Considering a bike", workspace_id="user")
        storage.store_memory_item(MemoryItem(
            content="bike is blue", item_type="fact",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            topic_id=tid,
        ))

        items = client.inspect_topic(tid)

        assert len(items) == 1
        assert items[0]["content"] == "bike is blue"
        assert items[0]["item_type"] == "fact"

    def test_empty_for_unknown_topic(self, make_eidolon_client):
        eidolon, _ = _make_eidolon()
        client = make_eidolon_client(eidolon)
        assert client.inspect_topic(999) == []
