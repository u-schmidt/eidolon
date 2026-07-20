"""Tests for the SQLite storage backend."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.models import Fact, MemoryItem
from core.stores.sqlite_store import SqliteStore
from tests.mocks import SpyTracer


@pytest.fixture
def store():
    """Create a fresh SqliteStore in a temp directory for each test."""
    tmp = tempfile.mkdtemp()
    db_path = str(Path(tmp) / "test.db")
    s = SqliteStore(db_path)
    s.connect()
    yield s
    s.close()


class TestFacts:
    def test_set_and_get_facts(self, store: SqliteStore):
        store.set_fact("user", "basic", "name", "Alex", "user")
        store.set_fact("user", "basic", "age", "42", "user")

        facts = store.get_facts("user")
        assert len(facts) == 2
        assert all(isinstance(f, Fact) for f in facts)
        assert facts[0].key == "age"  # sorted by category, key
        assert facts[1].key == "name"

    def test_get_facts_empty(self, store: SqliteStore):
        assert store.get_facts("unknown") == []

    def test_set_fact_overwrites(self, store: SqliteStore):
        store.set_fact("user", "basic", "name", "Alex", "user")
        store.set_fact("user", "basic", "name", "Alex Morgan", "ai")

        facts = store.get_facts("user")
        assert len(facts) == 1
        assert facts[0].value == "Alex Morgan"
        assert facts[0].source == "ai"

    def test_delete_fact(self, store: SqliteStore):
        store.set_fact("user", "basic", "name", "Alex", "user")
        store.delete_fact("user", "basic", "name")
        assert store.get_facts("user") == []

    def test_delete_fact_missing_is_a_noop(self, store: SqliteStore):
        store.delete_fact("user", "basic", "name")  # must not raise
        assert store.get_facts("user") == []


class TestFactsConsolidation:
    def test_count_user_turns_only_counts_user_role(self, store: SqliteStore):
        store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        store.store_message(turn_id=1, role="ai", content="hello", peer_id="ai")
        store.store_message(turn_id=2, role="user", content="hi again", peer_id="user")

        assert store.count_user_turns("user") == 2
        assert store.count_user_turns("someone_else") == 0

    def test_get_fact_items_since_filters_by_type_subject_and_id(self, store: SqliteStore):
        fact_item = MemoryItem(
            content="likes cats", item_type="fact",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
        )
        essence_item = MemoryItem(
            content="chatted about weather", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
        )
        other_subject_item = MemoryItem(
            content="ai likes helping", item_type="fact",
            source_peer="ai", subject_peer="ai", attribution_type="ai_about_self",
        )
        fact_id = store.store_memory_item(fact_item)
        store.store_memory_item(essence_item)
        store.store_memory_item(other_subject_item)

        results = store.get_fact_items_since("user", 0)
        assert [r.content for r in results] == ["likes cats"]
        assert results[0].id == fact_id

        assert store.get_fact_items_since("user", fact_id) == []

    def test_consolidation_checkpoint_defaults_to_zero(self, store: SqliteStore):
        assert store.get_consolidation_checkpoint("user") == 0

    def test_consolidation_checkpoint_set_and_get(self, store: SqliteStore):
        store.set_consolidation_checkpoint("user", 5)
        assert store.get_consolidation_checkpoint("user") == 5

    def test_consolidation_checkpoint_upserts(self, store: SqliteStore):
        store.set_consolidation_checkpoint("user", 5)
        store.set_consolidation_checkpoint("user", 9)
        assert store.get_consolidation_checkpoint("user") == 9

    def test_consolidation_checkpoint_is_per_peer(self, store: SqliteStore):
        store.set_consolidation_checkpoint("user", 5)
        assert store.get_consolidation_checkpoint("other_peer") == 0


class TestTopics:
    def test_create_and_get_topic(self, store: SqliteStore):
        tid = store.create_topic("Bello Features", summary="Discussion about Bello", tags=["bello", "fitness"])
        assert isinstance(tid, int)

        topic = store.get_topic(tid)
        assert topic is not None
        assert topic["label"] == "Bello Features"
        assert topic["tags"] == ["bello", "fitness"]
        assert topic["status"] == "active"

    def test_get_topic_not_found(self, store: SqliteStore):
        assert store.get_topic(999) is None

    def test_close_topic(self, store: SqliteStore):
        tid = store.create_topic("Test", summary="")
        store.close_topic(tid, "Final summary")
        topic = store.get_topic(tid)
        assert topic["status"] == "closed"
        assert topic["summary"] == "Final summary"
        assert topic["closed_at"] is not None

    def test_get_active_topic_returns_open_topic(self, store: SqliteStore):
        tid = store.create_topic("Ongoing", summary="", peer_id="user")
        assert store.get_active_topic("user") == tid

    def test_get_active_topic_none_when_closed(self, store: SqliteStore):
        tid = store.create_topic("Done", summary="", peer_id="user")
        store.close_topic(tid, "Wrapped up")
        assert store.get_active_topic("user") is None

    def test_get_active_topic_none_for_unknown_peer(self, store: SqliteStore):
        store.create_topic("Ongoing", summary="", peer_id="user")
        assert store.get_active_topic("someone_else") is None

    def test_get_active_topic_picks_most_recent(self, store: SqliteStore):
        store.create_topic("Older", summary="", peer_id="user")
        newer = store.create_topic("Newer", summary="", peer_id="user")
        assert store.get_active_topic("user") == newer

    def test_search_topics_by_tags(self, store: SqliteStore):
        store.create_topic("Bello", summary="", tags=["bello", "fitness"])
        store.create_topic("Mampfi", summary="", tags=["mampfi", "shopping"])
        store.create_topic("Fitness", summary="", tags=["fitness", "health"])

        # Close them so they're searchable
        store.close_topic(1, "")
        store.close_topic(2, "")
        store.close_topic(3, "")

        results = store.search_topics(tags=["fitness"])
        assert len(results) >= 2  # Bello + Fitness
        labels = [r["label"] for r in results]
        assert "Bello" in labels
        assert "Fitness" in labels

    def test_search_topics_respects_custom_weights(self, store: SqliteStore):
        """A zeroed-out tag_overlap weight must drop tag-only matches from scoring."""
        store.create_topic("Bello", summary="", tags=["bello", "fitness"])
        store.close_topic(1, "")

        # Default weights: tag overlap alone (0.3) produces a positive score.
        default_results = store.search_topics(tags=["fitness"])
        assert len(default_results) == 1

        # With tag_overlap weight zeroed out and no embedding/peer signal,
        # the topic should score 0 and drop out of the results.
        zeroed_results = store.search_topics(
            tags=["fitness"],
            weights={"semantic": 0.4, "tag_overlap": 0.0, "person": 0.2, "temporal": 0.1},
        )
        assert zeroed_results == []

    def test_search_topics_applies_person_weight_for_matching_peer(self, store: SqliteStore):
        """A topic with no tag/semantic match still surfaces on peer_id
        alone once the person weight applies — this is the exact path a
        row-index bug in the dict construction used to silently break
        (topic["peer_id"] was reading the wrong column, so it could never
        equal the caller's peer_id and this boost never fired)."""
        store.create_topic("Bello", summary="", peer_id="user")
        store.close_topic(1, "")

        results = store.search_topics(peer_id="user")
        assert len(results) == 1
        assert results[0]["peer_id"] == "user"

    def test_search_topics_returns_correct_created_at(self, store: SqliteStore):
        store.create_topic("Bello", summary="", tags=["fitness"])
        store.close_topic(1, "")

        results = store.search_topics(tags=["fitness"])
        assert len(results) == 1
        # A real timestamp string, not a stray peer_id/status value.
        assert results[0]["created_at"] is not None
        assert results[0]["created_at"] != "user"
        assert results[0]["status"] == "closed"

    def test_search_topics_records_scoring_summary_event(self, store: SqliteStore):
        store.create_topic("Bello", summary="", tags=["bello", "fitness"])
        store.create_topic("Fitness", summary="", tags=["fitness", "health"])
        store.close_topic(1, "")
        store.close_topic(2, "")

        tracer = SpyTracer()
        results = store.search_topics(tags=["fitness"], limit=1, tracer=tracer, request_id="req1")

        assert len(results) == 1
        events = tracer.events_of_type("retrieval_scoring_summary")
        assert len(events) == 1  # one batched event for the whole call, not one per candidate
        event = events[0]
        assert event["request_id"] == "req1"
        assert event["payload"]["candidate_type"] == "topic"
        assert event["payload"]["total_scored"] == 2  # both candidates scored positively
        assert event["payload"]["included_count"] == 1  # only 1 made the limit=1 cut
        top = event["payload"]["top_candidates"]
        assert len(top) == 2  # both fit within limit + padding
        included_flags = [c["included"] for c in top]
        assert included_flags.count(True) == 1
        assert included_flags.count(False) == 1

    def test_search_topics_score_breakdown_values(self, store: SqliteStore):
        tid = store.create_topic("Bello", summary="", embedding=[1.0, 0.0, 0.0])
        store.close_topic(tid, "")

        tracer = SpyTracer()
        store.search_topics(
            query_embedding=[1.0, 0.0, 0.0],
            weights={"semantic": 0.5, "tag_overlap": 0.3, "person": 0.2, "temporal": 0.1},
            tracer=tracer,
        )

        event = tracer.events_of_type("retrieval_scoring_summary")[0]
        candidate = event["payload"]["top_candidates"][0]
        assert candidate["semantic_raw"] == pytest.approx(1.0)
        assert candidate["semantic_weighted"] == pytest.approx(0.5)
        assert candidate["final_score"] == pytest.approx(0.5)

    def test_search_topics_scoring_summary_caps_top_candidates(self, store: SqliteStore):
        """More positively-scored candidates than limit + padding must still
        only persist the top slice — this is the whole point of the change."""
        for i in range(20):
            store.create_topic(f"Topic {i}", summary="", tags=["fitness"])
            store.close_topic(i + 1, "")

        tracer = SpyTracer()
        store.search_topics(tags=["fitness"], limit=1, tracer=tracer)

        event = tracer.events_of_type("retrieval_scoring_summary")[0]
        assert event["payload"]["total_scored"] == 20
        assert len(event["payload"]["top_candidates"]) == 1 + 10  # limit + _TRACE_TOP_K_PADDING

    def test_search_topics_no_tracer_does_not_raise(self, store: SqliteStore):
        store.create_topic("Bello", summary="", tags=["fitness"])
        store.close_topic(1, "")
        results = store.search_topics(tags=["fitness"])
        assert len(results) == 1


class TestGetRecentTopics:
    def test_returns_closed_topics_newest_first(self, store: SqliteStore):
        older = store.create_topic("Older", summary="", peer_id="user")
        newer = store.create_topic("Newer", summary="", peer_id="user")
        store.close_topic(older, "Older summary")
        store.close_topic(newer, "Newer summary")

        results = store.get_recent_topics("user", limit=10)
        assert [t["id"] for t in results] == [newer, older]

    def test_excludes_active_topics(self, store: SqliteStore):
        closed = store.create_topic("Closed", summary="", peer_id="user")
        store.close_topic(closed, "Wrapped up")
        store.create_topic("Still open", summary="", peer_id="user")

        results = store.get_recent_topics("user", limit=10)
        assert [t["id"] for t in results] == [closed]

    def test_filters_by_peer_id(self, store: SqliteStore):
        mine = store.create_topic("Mine", summary="", peer_id="user")
        store.close_topic(mine, "")
        theirs = store.create_topic("Theirs", summary="", peer_id="someone_else")
        store.close_topic(theirs, "")

        results = store.get_recent_topics("user", limit=10)
        assert [t["id"] for t in results] == [mine]

    def test_respects_limit(self, store: SqliteStore):
        for i in range(5):
            tid = store.create_topic(f"Topic {i}", summary="", peer_id="user")
            store.close_topic(tid, "")

        results = store.get_recent_topics("user", limit=2)
        assert len(results) == 2

    def test_empty_when_no_closed_topics(self, store: SqliteStore):
        assert store.get_recent_topics("user", limit=10) == []


class TestRecentMessagesByTopic:
    def test_returns_most_recent_within_limit_oldest_first(self, store: SqliteStore):
        tid = store.create_topic("Bello", summary="")
        for i in range(5):
            store.store_message(turn_id=i, role="user", content=f"msg {i}", peer_id="user", topic_id=tid)

        recent = store.get_recent_messages_by_topic(tid, limit=2)
        assert [m["content"] for m in recent] == ["msg 3", "msg 4"]

    def test_returns_all_when_fewer_than_limit(self, store: SqliteStore):
        tid = store.create_topic("Bello", summary="")
        store.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=tid)

        recent = store.get_recent_messages_by_topic(tid, limit=10)
        assert len(recent) == 1

    def test_empty_for_topic_with_no_messages(self, store: SqliteStore):
        tid = store.create_topic("Bello", summary="")
        assert store.get_recent_messages_by_topic(tid, limit=4) == []

    def test_only_messages_for_given_topic(self, store: SqliteStore):
        t1 = store.create_topic("First", summary="")
        t2 = store.create_topic("Second", summary="")
        store.store_message(turn_id=1, role="user", content="about first", peer_id="user", topic_id=t1)
        store.store_message(turn_id=2, role="user", content="about second", peer_id="user", topic_id=t2)

        recent = store.get_recent_messages_by_topic(t1, limit=10)
        assert [m["content"] for m in recent] == ["about first"]


class TestMemoryItems:
    def test_store_and_search(self, store: SqliteStore):
        item = MemoryItem(
            content="User talked about Bello features",
            item_type="essence",
            source_peer="user",
            subject_peer="user",
            attribution_type="self_statement",
        )
        mid = store.store_memory_item(item)
        assert isinstance(mid, int)

    def test_search_by_peer(self, store: SqliteStore):
        store.store_memory_item(MemoryItem(
            content="About Bello", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
        ))
        store.store_memory_item(MemoryItem(
            content="About Mampfi", item_type="essence",
            source_peer="ai", subject_peer="user", attribution_type="observation",
        ))

        # Without embedding, peer filter still works via score
        results = store.search_memory_items(peer_id="user")
        assert len(results) >= 1

    def test_search_memory_items_records_scoring_summary_event(self, store: SqliteStore):
        store.store_memory_item(MemoryItem(
            content="About Bello", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
        ))
        store.store_memory_item(MemoryItem(
            content="About Mampfi", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
        ))

        tracer = SpyTracer()
        results = store.search_memory_items(peer_id="user", limit=1, tracer=tracer, request_id="req1")

        assert len(results) == 1
        events = tracer.events_of_type("retrieval_scoring_summary")
        assert len(events) == 1  # one batched event, not one per candidate
        event = events[0]
        assert event["request_id"] == "req1"
        assert event["payload"]["candidate_type"] == "memory_item"
        assert event["payload"]["total_scored"] == 2
        assert event["payload"]["included_count"] == 1
        included_flags = [c["included"] for c in event["payload"]["top_candidates"]]
        assert included_flags.count(True) == 1
        assert included_flags.count(False) == 1

    def test_search_memory_items_no_tracer_does_not_raise(self, store: SqliteStore):
        store.store_memory_item(MemoryItem(
            content="About Bello", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
        ))
        results = store.search_memory_items(peer_id="user")
        assert len(results) == 1
