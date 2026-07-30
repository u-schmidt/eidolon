"""Tests for EntityProfileConsolidator.

Uses the real SqliteStore (not a hand-rolled mock), same rationale as
test_fact_consolidator.py: these tests verify the consolidator actually
matches the storage layer's real contracts — scan/consolidation
checkpoint semantics, get_topics_since/get_topics_mentioning_entity
filtering, etc.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.consolidation.entity_profile_consolidator import EntityProfileConsolidator
from core.stores.sqlite_store import SqliteStore
from tests.mocks import MockLLM, SpyMetricsRecorder, SpyTracer


@pytest.fixture
def store():
    tmp = tempfile.mkdtemp()
    s = SqliteStore(str(Path(tmp) / "test.db"), use_vec_index=False)
    s.connect()
    yield s
    s.close()


def _closed_topic(store: SqliteStore, workspace_id: str, named_entities: list[str], summary: str = "a summary") -> int:
    tid = store.create_topic("Test Topic", workspace_id=workspace_id)
    store.close_topic(tid, summary)
    store.update_topic_entities(tid, named_entities)
    return tid


class TestConsolidateNoOp:
    def test_no_llm_call_when_no_closed_topics(self, store: SqliteStore):
        class FailIfCalledLLM:
            def complete(self, *a, **kw):
                raise AssertionError("LLM should not be called when there's nothing to scan")

        EntityProfileConsolidator(store, FailIfCalledLLM()).consolidate("user")  # must not raise

    def test_no_llm_call_when_nothing_new_since_last_scan(self, store: SqliteStore):
        _closed_topic(store, "user", [])  # no entities, nothing to consolidate anyway
        EntityProfileConsolidator(store, MockLLM("{}")).consolidate("user")

        class FailIfCalledLLM:
            def complete(self, *a, **kw):
                raise AssertionError("LLM should not be called again — no new topics since checkpoint")

        EntityProfileConsolidator(store, FailIfCalledLLM()).consolidate("user")  # must not raise

    def test_no_llm_call_for_entities_not_yet_materialized(self, store: SqliteStore):
        _closed_topic(store, "user", ["lisa"])  # single mention, threshold defaults to 2

        class FailIfCalledLLM:
            def complete(self, *a, **kw):
                raise AssertionError("LLM should not be called — lisa hasn't crossed the threshold")

        EntityProfileConsolidator(store, FailIfCalledLLM(), threshold=2).consolidate("user")
        assert store.get_materialized_entity_names("user") == []


class TestMaterialization:
    def test_materializes_at_threshold_within_one_pass(self, store: SqliteStore):
        _closed_topic(store, "user", ["lisa"])
        _closed_topic(store, "user", ["lisa"])  # 2nd mention, both scanned in the same pass
        llm = MockLLM('{"upserts": [], "deletes": []}')

        EntityProfileConsolidator(store, llm, threshold=2).consolidate("user")

        assert store.get_materialized_entity_names("user") == ["lisa"]

    def test_stays_unmaterialized_below_threshold(self, store: SqliteStore):
        _closed_topic(store, "user", ["lisa"])
        _closed_topic(store, "user", ["bob"])  # different entity — lisa still only mentioned once
        llm = MockLLM('{"upserts": [], "deletes": []}')

        EntityProfileConsolidator(store, llm, threshold=2).consolidate("user")

        assert store.get_materialized_entity_names("user") == []

    def test_materialization_is_per_workspace(self, store: SqliteStore):
        _closed_topic(store, "user", ["lisa"])
        _closed_topic(store, "other_workspace", ["lisa"])
        llm = MockLLM('{"upserts": [], "deletes": []}')

        EntityProfileConsolidator(store, llm, threshold=1).consolidate("user")
        EntityProfileConsolidator(store, llm, threshold=1).consolidate("other_workspace")

        assert store.get_materialized_entity_names("user") == ["lisa"]
        assert store.get_materialized_entity_names("other_workspace") == ["lisa"]
        # each workspace's own card is independent — not asserting cross-
        # contamination beyond both containing "lisa", since name
        # collision across workspaces isn't this feature's concern.


class TestConsolidateApplies:
    def test_applies_upserts_to_entity_facts(self, store: SqliteStore):
        _closed_topic(store, "user", ["lisa"], summary="met lisa at work")
        llm = MockLLM('{"upserts": [{"category": "relationship", "key": "role", "value": "coworker"}], "deletes": []}')

        EntityProfileConsolidator(store, llm, threshold=1).consolidate("user")

        facts = store.get_entity_facts("user", ["lisa"])
        assert len(facts) == 1
        assert facts[0].entity_name == "lisa"
        assert facts[0].category == "relationship"
        assert facts[0].key == "role"
        assert facts[0].value == "coworker"
        assert facts[0].source == "ai"

    def test_applies_deletes(self, store: SqliteStore):
        store.set_entity_fact("user", "lisa", "relationship", "role", "coworker", "ai")
        _closed_topic(store, "user", ["lisa"], summary="lisa no longer works there")
        llm = MockLLM('{"upserts": [], "deletes": [{"category": "relationship", "key": "role"}]}')

        EntityProfileConsolidator(store, llm, threshold=1).consolidate("user")

        assert store.get_entity_facts("user", ["lisa"]) == []

    def test_upsert_updates_existing_key_in_place(self, store: SqliteStore):
        store.set_entity_fact("user", "lisa", "relationship", "role", "coworker", "ai")
        _closed_topic(store, "user", ["lisa"], summary="lisa is now a manager")
        llm = MockLLM('{"upserts": [{"category": "relationship", "key": "role", "value": "manager"}], "deletes": []}')

        EntityProfileConsolidator(store, llm, threshold=1).consolidate("user")

        facts = store.get_entity_facts("user", ["lisa"])
        assert len(facts) == 1
        assert facts[0].value == "manager"

    def test_skips_incomplete_upsert_entries(self, store: SqliteStore):
        _closed_topic(store, "user", ["lisa"])
        llm = MockLLM('{"upserts": [{"category": "relationship", "key": ""}], "deletes": []}')

        EntityProfileConsolidator(store, llm, threshold=1).consolidate("user")

        assert store.get_entity_facts("user", ["lisa"]) == []

    def test_importance_falls_back_to_medium_when_invalid(self, store: SqliteStore):
        _closed_topic(store, "user", ["lisa"])
        llm = MockLLM(
            '{"upserts": [{"category": "relationship", "key": "role", "value": "coworker", '
            '"importance": "urgent"}], "deletes": []}'
        )

        EntityProfileConsolidator(store, llm, threshold=1).consolidate("user")

        assert store.get_entity_facts("user", ["lisa"])[0].importance == "medium"


class TestConsolidateCheckpoints:
    def test_entity_consolidation_checkpoint_advances_on_success(self, store: SqliteStore):
        tid = _closed_topic(store, "user", ["lisa"])
        llm = MockLLM('{"upserts": [], "deletes": []}')

        EntityProfileConsolidator(store, llm, threshold=1).consolidate("user")

        assert store.get_entity_consolidation_checkpoint("user", "lisa") == tid

    def test_scan_checkpoint_advances_even_if_entity_consolidation_fails(self, store: SqliteStore):
        tid = _closed_topic(store, "user", ["lisa"])
        llm = MockLLM("not valid json")

        EntityProfileConsolidator(store, llm, threshold=1).consolidate("user")

        assert store.get_entity_scan_checkpoint("user") == tid
        assert store.get_entity_consolidation_checkpoint("user", "lisa") == 0
        assert store.get_entity_facts("user", ["lisa"]) == []

    def test_first_consolidation_pulls_full_mention_history(self, store: SqliteStore):
        """threshold=1 materializes lisa on her first mention; the pass
        that materializes her must still see that same topic as source
        material (checkpoint starts at 0, not at the topic that just
        triggered materialization) — see EntityProfileConsolidator's
        docstring."""
        _closed_topic(store, "user", ["lisa"], summary="first mention of lisa")
        llm = MockLLM('{"upserts": [], "deletes": []}')

        EntityProfileConsolidator(store, llm, threshold=1).consolidate("user")

        assert "first mention of lisa" in llm.last_prompt

    def test_second_pass_only_sees_new_topics_not_already_consolidated(self, store: SqliteStore):
        _closed_topic(store, "user", ["lisa"], summary="first mention of lisa")
        llm1 = MockLLM('{"upserts": [], "deletes": []}')
        EntityProfileConsolidator(store, llm1, threshold=1).consolidate("user")

        _closed_topic(store, "user", ["lisa"], summary="second mention of lisa")
        llm2 = MockLLM('{"upserts": [], "deletes": []}')
        EntityProfileConsolidator(store, llm2, threshold=1).consolidate("user")

        assert "second mention of lisa" in llm2.last_prompt
        assert "first mention of lisa" not in llm2.last_prompt

    def test_failed_entity_is_not_retried_until_touched_by_a_new_topic(self, store: SqliteStore):
        _closed_topic(store, "user", ["lisa"], summary="first mention")
        EntityProfileConsolidator(store, MockLLM("not valid json"), threshold=1).consolidate("user")
        assert store.get_entity_facts("user", ["lisa"]) == []

        # No new topic mentioning lisa — a second pass with a working LLM
        # still doesn't retry her, since nothing touched her this batch.
        class FailIfCalledLLM:
            def complete(self, *a, **kw):
                raise AssertionError("lisa wasn't touched by any new topic — should not be retried")

        EntityProfileConsolidator(store, FailIfCalledLLM(), threshold=1).consolidate("user")

        # A further topic mentioning her makes her eligible again.
        _closed_topic(store, "user", ["lisa"], summary="second mention")
        good_llm = MockLLM('{"upserts": [{"category": "relationship", "key": "role", "value": "coworker"}], "deletes": []}')
        EntityProfileConsolidator(store, good_llm, threshold=1).consolidate("user")

        assert len(store.get_entity_facts("user", ["lisa"])) == 1


class TestConsolidateTracing:
    def test_records_entity_profile_scan_event(self, store: SqliteStore):
        _closed_topic(store, "user", ["lisa"])
        tracer = SpyTracer()

        EntityProfileConsolidator(store, MockLLM('{"upserts": [], "deletes": []}'), threshold=1, tracer=tracer).consolidate(
            "user", request_id="req1", session_id="sess1"
        )

        events = tracer.events_of_type("entity_profile_scan")
        assert len(events) == 1
        assert events[0]["request_id"] == "req1"
        assert events[0]["session_id"] == "sess1"
        assert events[0]["payload"]["workspace_id"] == "user"
        assert events[0]["payload"]["newly_materialized"] == ["lisa"]
        assert events[0]["payload"]["consolidated_entities"] == ["lisa"]

    def test_records_entity_profile_consolidation_event(self, store: SqliteStore):
        _closed_topic(store, "user", ["lisa"])
        tracer = SpyTracer()

        EntityProfileConsolidator(
            store, MockLLM('{"upserts": [{"category": "relationship", "key": "role", "value": "coworker"}], "deletes": []}'),
            threshold=1, tracer=tracer,
        ).consolidate("user")

        events = tracer.events_of_type("entity_profile_consolidation")
        assert len(events) == 1
        assert events[0]["payload"]["entity_name"] == "lisa"
        assert events[0]["payload"]["upserts"] == [{"category": "relationship", "key": "role", "value": "coworker"}]

    def test_records_error_event_on_malformed_entity_response(self, store: SqliteStore, caplog):
        _closed_topic(store, "user", ["lisa"])
        tracer = SpyTracer()

        with caplog.at_level("WARNING"):
            EntityProfileConsolidator(store, MockLLM("not valid json"), threshold=1, tracer=tracer).consolidate("user")

        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "entity_profile_consolidation" for e in error_events)
        assert tracer.events_of_type("entity_profile_consolidation") == []
        # The scan-level pass still records its own summary event —
        # per-entity failure doesn't abort the batch.
        assert len(tracer.events_of_type("entity_profile_scan")) == 1
        assert "entity_profile_consolidation failed" in caplog.text

    def test_records_llm_call_metric_on_success(self, store: SqliteStore):
        _closed_topic(store, "user", ["lisa"])
        metrics = SpyMetricsRecorder()

        EntityProfileConsolidator(
            store, MockLLM('{"upserts": [], "deletes": []}'), threshold=1, metrics=metrics
        ).consolidate("user", request_id="req1")

        assert len(metrics.llm_calls) == 1
        assert metrics.llm_calls[0]["call_type"] == "entity_profile_consolidation"
        assert metrics.llm_calls[0]["succeeded"] is True
        assert metrics.llm_calls[0]["workspace_id"] == "user"

    def test_records_error_metric_on_malformed_response(self, store: SqliteStore, caplog):
        _closed_topic(store, "user", ["lisa"])
        metrics = SpyMetricsRecorder()

        with caplog.at_level("WARNING"):
            EntityProfileConsolidator(
                store, MockLLM("not valid json"), threshold=1, metrics=metrics
            ).consolidate("user")

        assert len(metrics.events) == 1
        assert metrics.events[0]["severity"] == "error"
        assert metrics.events[0]["step"] == "entity_profile_consolidation"
        assert metrics.llm_calls == []


class TestConsolidatePrompt:
    def test_system_prompt_mentions_entity_name_and_target_max(self, store: SqliteStore):
        _closed_topic(store, "user", ["lisa"])
        llm = MockLLM('{"upserts": [], "deletes": []}')

        EntityProfileConsolidator(store, llm, threshold=1, target_max=7).consolidate("user")

        assert "lisa" in llm.last_system
        assert "7" in llm.last_system

    def test_prompt_includes_current_profile_and_source_topics(self, store: SqliteStore):
        store.set_entity_fact("user", "lisa", "relationship", "role", "coworker", "ai")
        _closed_topic(store, "user", ["lisa"], summary="lisa got promoted")
        llm = MockLLM('{"upserts": [], "deletes": []}')

        EntityProfileConsolidator(store, llm, threshold=1).consolidate("user")

        assert "relationship: role = coworker" in llm.last_prompt
        assert "lisa got promoted" in llm.last_prompt
