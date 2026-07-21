"""Tests for FactConsolidator.

Uses the real SqliteStore (not a hand-rolled mock) so these tests verify
FactConsolidator actually matches the storage layer's real contracts —
checkpoint semantics, get_fact_items_since filtering, etc.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.consolidation.fact_consolidator import FactConsolidator
from core.models import MemoryItem
from core.stores.sqlite_store import SqliteStore
from tests.mocks import MockLLM, SpyTracer


@pytest.fixture
def store():
    tmp = tempfile.mkdtemp()
    s = SqliteStore(str(Path(tmp) / "test.db"))
    s.connect()
    yield s
    s.close()


def _fact_item(content: str) -> MemoryItem:
    return MemoryItem(
        content=content, item_type="fact",
        source_peer="user", subject_peer="user", attribution_type="self_statement",
    )


class TestConsolidateNoOp:
    def test_no_llm_call_when_nothing_new(self, store: SqliteStore):
        class FailIfCalledLLM:
            def complete(self, *a, **kw):
                raise AssertionError("LLM should not be called when nothing is new")

        FactConsolidator(store, FailIfCalledLLM()).consolidate("user")  # must not raise

    def test_no_llm_call_after_everything_already_consolidated(self, store: SqliteStore):
        store.store_memory_item(_fact_item("likes cats"))
        llm = MockLLM('{"upserts": [{"category": "preferences", "key": "pets", "value": "likes cats"}], "deletes": []}')
        FactConsolidator(store, llm).consolidate("user")

        class FailIfCalledLLM:
            def complete(self, *a, **kw):
                raise AssertionError("LLM should not be called again — nothing new since checkpoint")

        FactConsolidator(store, FailIfCalledLLM()).consolidate("user")  # must not raise


class TestConsolidateApplies:
    def test_applies_upserts(self, store: SqliteStore):
        store.store_memory_item(_fact_item("likes cats"))
        llm = MockLLM('{"upserts": [{"category": "preferences", "key": "pets", "value": "likes cats"}], "deletes": []}')

        FactConsolidator(store, llm).consolidate("user")

        facts = store.get_facts("user")
        assert len(facts) == 1
        assert facts[0].category == "preferences"
        assert facts[0].key == "pets"
        assert facts[0].value == "likes cats"
        assert facts[0].source == "ai"

    def test_upsert_updates_existing_key_in_place(self, store: SqliteStore):
        store.set_fact("user", "location", "city", "Hamburg", "ai")
        store.store_memory_item(_fact_item("moved to Berlin"))
        llm = MockLLM('{"upserts": [{"category": "location", "key": "city", "value": "Berlin"}], "deletes": []}')

        FactConsolidator(store, llm).consolidate("user")

        facts = store.get_facts("user")
        assert len(facts) == 1  # updated in place, not duplicated
        assert facts[0].value == "Berlin"

    def test_applies_deletes(self, store: SqliteStore):
        store.set_fact("user", "lifestyle", "smoking", "smokes occasionally", "ai")
        store.store_memory_item(_fact_item("quit smoking entirely"))
        llm = MockLLM('{"upserts": [], "deletes": [{"category": "lifestyle", "key": "smoking"}]}')

        FactConsolidator(store, llm).consolidate("user")

        assert store.get_facts("user") == []

    def test_normalizes_category_and_key_casing(self, store: SqliteStore):
        store.store_memory_item(_fact_item("likes cats"))
        llm = MockLLM('{"upserts": [{"category": "  Preferences ", "key": "PETS", "value": "likes cats"}], "deletes": []}')

        FactConsolidator(store, llm).consolidate("user")

        facts = store.get_facts("user")
        assert facts[0].category == "preferences"
        assert facts[0].key == "pets"

    def test_skips_incomplete_upsert_entries(self, store: SqliteStore):
        store.store_memory_item(_fact_item("likes cats"))
        llm = MockLLM('{"upserts": [{"category": "preferences", "key": ""}], "deletes": []}')

        FactConsolidator(store, llm).consolidate("user")

        assert store.get_facts("user") == []


class TestConsolidateCheckpoint:
    def test_checkpoint_advances_on_success(self, store: SqliteStore):
        item_id = store.store_memory_item(_fact_item("likes cats"))
        llm = MockLLM('{"upserts": [{"category": "preferences", "key": "pets", "value": "likes cats"}], "deletes": []}')

        FactConsolidator(store, llm).consolidate("user")

        assert store.get_consolidation_checkpoint("user") == item_id

    def test_checkpoint_does_not_advance_on_malformed_response(self, store: SqliteStore):
        store.store_memory_item(_fact_item("likes cats"))
        llm = MockLLM("not valid json")

        FactConsolidator(store, llm).consolidate("user")

        assert store.get_consolidation_checkpoint("user") == 0
        assert store.get_facts("user") == []

    def test_failed_pass_is_retried_next_time(self, store: SqliteStore):
        store.store_memory_item(_fact_item("likes cats"))

        FactConsolidator(store, MockLLM("not valid json")).consolidate("user")
        assert store.get_facts("user") == []

        good_llm = MockLLM('{"upserts": [{"category": "preferences", "key": "pets", "value": "likes cats"}], "deletes": []}')
        FactConsolidator(store, good_llm).consolidate("user")
        assert len(store.get_facts("user")) == 1


class TestConsolidateTracing:
    def test_records_fact_consolidation_event_on_success(self, store: SqliteStore):
        store.store_memory_item(_fact_item("likes cats"))
        llm = MockLLM('{"upserts": [{"category": "preferences", "key": "pets", "value": "likes cats"}], "deletes": []}')
        tracer = SpyTracer()

        FactConsolidator(store, llm, tracer=tracer).consolidate("user", request_id="req1", session_id="sess1")

        events = tracer.events_of_type("fact_consolidation")
        assert len(events) == 1
        event = events[0]
        assert event["request_id"] == "req1"
        assert event["session_id"] == "sess1"
        assert event["payload"]["peer_id"] == "user"
        assert event["payload"]["new_item_count"] == 1
        assert event["payload"]["total_facts_after"] == 1
        assert event["payload"]["upserts"] == [{"category": "preferences", "key": "pets", "value": "likes cats"}]

    def test_records_error_event_on_malformed_response(self, store: SqliteStore):
        store.store_memory_item(_fact_item("likes cats"))
        tracer = SpyTracer()

        FactConsolidator(store, MockLLM("not valid json"), tracer=tracer).consolidate("user")

        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "fact_consolidation" for e in error_events)
        assert tracer.events_of_type("fact_consolidation") == []


class TestConsolidatePeerIdVsSubjectPeer:
    """peer_id identifies *whose* profile this is; subject_peer is always
    FragmentaryExtractor's fixed "user"/"ai"/"other" label — a different
    value space. Regression coverage for a bug where consolidate() looked
    up new items by peer_id instead of the "user" subject label, so
    consolidation silently no-op'd forever for any peer_id that wasn't
    literally the string "user" (which every other caller/test happened
    to use, masking it completely)."""

    def test_finds_and_consolidates_facts_for_a_peer_id_other_than_user(self, store: SqliteStore):
        store.store_memory_item(_fact_item("likes cats"))  # subject_peer="user" regardless of peer_id
        llm = MockLLM('{"upserts": [{"category": "preferences", "key": "pets", "value": "likes cats"}], "deletes": []}')

        FactConsolidator(store, llm).consolidate("telegram:12345")

        facts = store.get_facts("telegram:12345")
        assert len(facts) == 1
        assert facts[0].value == "likes cats"

    def test_facts_and_checkpoint_are_scoped_by_peer_id_not_leaked(self, store: SqliteStore):
        store.store_memory_item(_fact_item("likes cats"))
        llm = MockLLM('{"upserts": [{"category": "preferences", "key": "pets", "value": "likes cats"}], "deletes": []}')

        FactConsolidator(store, llm).consolidate("telegram:12345")

        assert store.get_facts("user") == []  # not written into an unrelated peer's profile
        assert store.get_consolidation_checkpoint("telegram:12345") != 0
        assert store.get_consolidation_checkpoint("user") == 0


class TestConsolidatePrompt:
    def test_prompt_includes_current_profile_and_new_items(self, store: SqliteStore):
        store.set_fact("user", "location", "city", "Hamburg", "ai")
        store.store_memory_item(_fact_item("has a partner named Sam"))

        llm = MockLLM('{"upserts": [], "deletes": []}')
        FactConsolidator(store, llm).consolidate("user")

        assert "location: city = Hamburg" in llm.last_prompt
        assert "has a partner named Sam" in llm.last_prompt

    def test_system_prompt_mentions_target_max(self, store: SqliteStore):
        store.store_memory_item(_fact_item("likes cats"))
        llm = MockLLM('{"upserts": [], "deletes": []}')

        FactConsolidator(store, llm, target_max=42).consolidate("user")

        assert "42" in llm.last_system
