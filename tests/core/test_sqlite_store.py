"""Tests for the SQLite storage backend."""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from core.models import Fact, MemoryItem
from core.stores.sqlite_store import SqliteStore, _recency_score
from tests.mocks import SpyTracer


@pytest.fixture
def store():
    """Create a fresh SqliteStore in a temp directory for each test.

    use_vec_index=False so these tests deterministically exercise the
    brute-force scan regardless of whether sqlite-vec happens to be
    installed in the environment — the accelerated path has its own
    dedicated tests (TestVecIndexAcceleratedSearch below).
    """
    tmp = tempfile.mkdtemp()
    db_path = str(Path(tmp) / "test.db")
    s = SqliteStore(db_path, use_vec_index=False)
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

    def test_update_topic_tags_replaces_tags(self, store: SqliteStore):
        tid = store.create_topic("Test", tags=["old"])
        store.update_topic_tags(tid, ["fitness", "gym"])
        assert store.get_topic(tid)["tags"] == ["fitness", "gym"]

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

    def test_get_active_topic_scoped_per_session(self, store: SqliteStore):
        """A peer's concurrent sessions (web, Telegram, CLI, ...) must each
        get their own active topic, not share whichever is newest."""
        web_topic = store.create_topic("Web thread", summary="", peer_id="user", session_id="web")
        telegram_topic = store.create_topic("Telegram thread", summary="", peer_id="user", session_id="telegram")

        assert store.get_active_topic("user", session_id="web") == web_topic
        assert store.get_active_topic("user", session_id="telegram") == telegram_topic

    def test_get_active_topic_with_no_session_id_only_matches_sessionless_topics(self, store: SqliteStore):
        """None is not a wildcard across sessions — it only matches topics
        created with no session_id (legacy rows, single-session callers
        like the CLI)."""
        store.create_topic("Web thread", summary="", peer_id="user", session_id="web")
        sessionless = store.create_topic("CLI thread", summary="", peer_id="user")

        assert store.get_active_topic("user") == sessionless
        assert store.get_active_topic("user", session_id="web") != sessionless

    def test_create_topic_stores_session_id(self, store: SqliteStore):
        tid = store.create_topic("Web thread", summary="", peer_id="user", session_id="web")
        assert store.get_active_topic("user", session_id="web") == tid

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

    def test_search_topics_excludes_other_peers_even_with_strong_match(self, store: SqliteStore):
        """peer_id is a hard SQL filter, not just a soft scoring bonus —
        a different peer's topic must never surface regardless of how
        well it scores on tags/semantics. Without this, one peer's
        conversation could leak into another peer's retrieved context."""
        store.create_topic("Bello", summary="", tags=["fitness"], peer_id="someone_else", embedding=[1.0, 0.0, 0.0])
        store.close_topic(1, "")
        store.create_topic("Bello 2", summary="", tags=["fitness"], peer_id="user", embedding=[1.0, 0.0, 0.0])
        store.close_topic(2, "")

        results = store.search_topics(query_embedding=[1.0, 0.0, 0.0], tags=["fitness"], peer_id="user")

        assert [r["id"] for r in results] == [2]

    def test_search_topics_with_no_peer_id_searches_every_peer(self, store: SqliteStore):
        """peer_id=None keeps the pre-existing 'search everyone' behavior
        for callers that intentionally omit it — not a new default."""
        store.create_topic("Bello", summary="", tags=["fitness"], peer_id="someone_else")
        store.close_topic(1, "")
        store.create_topic("Bello 2", summary="", tags=["fitness"], peer_id="user")
        store.close_topic(2, "")

        results = store.search_topics(tags=["fitness"])

        assert {r["id"] for r in results} == {1, 2}

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
        # temporal=0 isolates the semantic term for an exact final_score.
        store.search_topics(
            query_embedding=[1.0, 0.0, 0.0],
            weights={"semantic": 0.5, "tag_overlap": 0.3, "person": 0.2, "temporal": 0.0},
            tracer=tracer,
        )

        event = tracer.events_of_type("retrieval_scoring_summary")[0]
        candidate = event["payload"]["top_candidates"][0]
        assert candidate["semantic_raw"] == pytest.approx(1.0)
        assert candidate["semantic_weighted"] == pytest.approx(0.5)
        assert candidate["temporal_weighted"] == pytest.approx(0.0)
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

    def test_search_topics_excludes_clustered_topics(self, store: SqliteStore):
        """Once a topic joins a cluster, its consolidated summary
        represents it — it must not also compete individually."""
        tid = store.create_topic("Bello", summary="", tags=["fitness"], peer_id="user")
        store.close_topic(tid, "")
        cid = store.create_cluster("user", "Fitness", "summary")
        store.assign_topic_to_cluster(tid, cid)

        assert store.search_topics(tags=["fitness"]) == []


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

    def test_excludes_clustered_topics(self, store: SqliteStore):
        clustered = store.create_topic("Clustered", summary="", peer_id="user")
        store.close_topic(clustered, "")
        cid = store.create_cluster("user", "Thread", "summary")
        store.assign_topic_to_cluster(clustered, cid)
        unclustered = store.create_topic("Unclustered", summary="", peer_id="user")
        store.close_topic(unclustered, "")

        results = store.get_recent_topics("user", limit=10)
        assert [t["id"] for t in results] == [unclustered]


class TestClusters:
    def test_create_and_get_cluster(self, store: SqliteStore):
        cid = store.create_cluster("user", "Bike purchase", "User wants a new bike", tags=["bike"])
        assert isinstance(cid, int)

        cluster = store.get_cluster(cid)
        assert cluster is not None
        assert cluster["peer_id"] == "user"
        assert cluster["label"] == "Bike purchase"
        assert cluster["summary"] == "User wants a new bike"
        assert cluster["tags"] == ["bike"]
        assert cluster["topic_count"] == 0

    def test_get_cluster_not_found(self, store: SqliteStore):
        assert store.get_cluster(999) is None

    def test_update_cluster_rewrites_summary_in_place(self, store: SqliteStore):
        cid = store.create_cluster("user", "Bike", "old summary")
        store.update_cluster(cid, summary="new consolidated summary")

        cluster = store.get_cluster(cid)
        assert cluster["summary"] == "new consolidated summary"
        assert cluster["label"] == "Bike"  # unchanged when not given

    def test_update_cluster_optional_fields_only_set_when_given(self, store: SqliteStore):
        cid = store.create_cluster("user", "Bike", "summary", tags=["bike"])
        store.update_cluster(cid, summary="new summary", label="Bike saga", tags=["bike", "shopping"])

        cluster = store.get_cluster(cid)
        assert cluster["label"] == "Bike saga"
        assert cluster["tags"] == ["bike", "shopping"]

    def test_assign_topic_to_cluster_sets_id_and_increments_count(self, store: SqliteStore):
        tid = store.create_topic("Considering a bike", summary="", peer_id="user")
        store.close_topic(tid, "thinking about it")
        cid = store.create_cluster("user", "Bike", "summary")

        store.assign_topic_to_cluster(tid, cid)

        assert store.get_cluster(cid)["topic_count"] == 1
        assert [m["id"] for m in store.get_topics_by_cluster(cid)] == [tid]

    def test_get_topics_by_cluster_oldest_first(self, store: SqliteStore):
        cid = store.create_cluster("user", "Bike", "summary")
        t1 = store.create_topic("First", summary="", peer_id="user")
        store.close_topic(t1, "first summary")
        t2 = store.create_topic("Second", summary="", peer_id="user")
        store.close_topic(t2, "second summary")
        store.assign_topic_to_cluster(t1, cid)
        store.assign_topic_to_cluster(t2, cid)

        assert [m["id"] for m in store.get_topics_by_cluster(cid)] == [t1, t2]

    def test_get_recent_topics_in_cluster_newest_first_and_limited(self, store: SqliteStore):
        cid = store.create_cluster("user", "Bike", "summary")
        t1 = store.create_topic("First", summary="", peer_id="user")
        store.close_topic(t1, "first")
        t2 = store.create_topic("Second", summary="", peer_id="user")
        store.close_topic(t2, "second")
        store.assign_topic_to_cluster(t1, cid)
        store.assign_topic_to_cluster(t2, cid)

        recent = store.get_recent_topics_in_cluster(cid, limit=1)
        assert [r["id"] for r in recent] == [t2]

    def test_search_clusters_scores_by_semantic_similarity(self, store: SqliteStore):
        store.create_cluster("user", "Bike", "summary", embedding=[1.0, 0.0, 0.0])
        results = store.search_clusters(query_embedding=[1.0, 0.0, 0.0], peer_id="user")
        assert len(results) == 1

    def test_search_clusters_hard_filters_by_peer_id(self, store: SqliteStore):
        """Unlike search_topics's soft person-weight bonus, peer_id here
        is a real SQL filter — clusters must never merge across peers."""
        store.create_cluster("someone_else", "Bike", "summary", embedding=[1.0, 0.0, 0.0])
        results = store.search_clusters(query_embedding=[1.0, 0.0, 0.0], peer_id="user")
        assert results == []

    def test_search_clusters_empty_without_peer_id(self, store: SqliteStore):
        store.create_cluster("user", "Bike", "summary", embedding=[1.0, 0.0, 0.0])
        assert store.search_clusters(query_embedding=[1.0, 0.0, 0.0]) == []

    def test_search_clusters_records_scoring_summary_event(self, store: SqliteStore):
        store.create_cluster("user", "Bike", "summary", embedding=[1.0, 0.0, 0.0])
        tracer = SpyTracer()
        store.search_clusters(query_embedding=[1.0, 0.0, 0.0], peer_id="user", tracer=tracer, request_id="req1")

        events = tracer.events_of_type("retrieval_scoring_summary")
        assert len(events) == 1
        assert events[0]["payload"]["candidate_type"] == "cluster"


class TestSearchClusterCandidates:
    def test_finds_unclustered_topic_with_matching_embedding(self, store: SqliteStore):
        other = store.create_topic("Other bike topic", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        store.close_topic(other, "other summary")
        new = store.create_topic("New bike topic", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        store.close_topic(new, "new summary")

        candidates = store.search_cluster_candidates(new, "user")
        assert len(candidates) == 1
        assert candidates[0]["candidate_kind"] == "topic"
        assert candidates[0]["id"] == other

    def test_finds_existing_cluster_with_matching_embedding(self, store: SqliteStore):
        cid = store.create_cluster("user", "Bike", "summary", embedding=[1.0, 0.0, 0.0])
        new = store.create_topic("New bike topic", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        store.close_topic(new, "new summary")

        candidates = store.search_cluster_candidates(new, "user")
        assert len(candidates) == 1
        assert candidates[0]["candidate_kind"] == "cluster"
        assert candidates[0]["id"] == cid

    def test_excludes_the_topic_itself(self, store: SqliteStore):
        new = store.create_topic("Solo topic", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        store.close_topic(new, "summary")

        assert store.search_cluster_candidates(new, "user") == []

    def test_excludes_already_clustered_topics(self, store: SqliteStore):
        """The already-clustered topic itself shouldn't reappear as a raw
        'topic' candidate — only its cluster should."""
        cid = store.create_cluster("user", "Bike", "summary", embedding=[1.0, 0.0, 0.0])
        already = store.create_topic("Already clustered", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        store.close_topic(already, "summary")
        store.assign_topic_to_cluster(already, cid)

        new = store.create_topic("New topic", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        store.close_topic(new, "summary")

        candidates = store.search_cluster_candidates(new, "user")
        assert [c["candidate_kind"] for c in candidates] == ["cluster"]

    def test_peer_scoped(self, store: SqliteStore):
        other_peer_topic = store.create_topic(
            "Other peer's topic", summary="", peer_id="someone_else", embedding=[1.0, 0.0, 0.0]
        )
        store.close_topic(other_peer_topic, "summary")
        new = store.create_topic("My topic", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        store.close_topic(new, "summary")

        assert store.search_cluster_candidates(new, "user") == []

    def test_empty_when_topic_has_no_embedding(self, store: SqliteStore):
        new = store.create_topic("No embedding", summary="", peer_id="user")
        store.close_topic(new, "summary")
        assert store.search_cluster_candidates(new, "user") == []

    def test_records_scoring_summary_event(self, store: SqliteStore):
        store.create_cluster("user", "Bike", "summary", embedding=[1.0, 0.0, 0.0])
        new = store.create_topic("New topic", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        store.close_topic(new, "summary")

        tracer = SpyTracer()
        store.search_cluster_candidates(new, "user", tracer=tracer, request_id="req1")

        events = tracer.events_of_type("retrieval_scoring_summary")
        assert len(events) == 1
        assert events[0]["payload"]["candidate_type"] == "cluster_candidate"
        assert events[0]["payload"]["topic_id"] == new


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
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        store.store_memory_item(MemoryItem(
            content="About Bello", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=msg_id, embedding=[1.0, 0.0, 0.0],
        ))
        other_msg_id = store.store_message(turn_id=1, role="ai", content="hi", peer_id="someone_else")
        store.store_memory_item(MemoryItem(
            content="About Mampfi", item_type="essence",
            source_peer="ai", subject_peer="user", attribution_type="observation",
            message_id=other_msg_id, embedding=[1.0, 0.0, 0.0],
        ))

        results = store.search_memory_items(peer_id="user", query_embedding=[1.0, 0.0, 0.0])
        assert [item.content for item in results] == ["About Bello"]

    def test_search_memory_items_excludes_other_peers_even_with_matching_source_peer(self, store: SqliteStore):
        """Regression: source_peer/subject_peer are always the fixed
        "user"/"ai"/"other" extraction vocabulary, never the real peer_id
        — an item belonging to a different real peer must never surface
        just because its source_peer happens to say "user" too, even
        with a strong semantic match."""
        mine_msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        store.store_memory_item(MemoryItem(
            content="Mine", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=mine_msg_id, embedding=[1.0, 0.0, 0.0],
        ))
        theirs_msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="telegram:999")
        store.store_memory_item(MemoryItem(
            content="Theirs", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=theirs_msg_id, embedding=[1.0, 0.0, 0.0],
        ))

        results = store.search_memory_items(peer_id="user", query_embedding=[1.0, 0.0, 0.0])
        assert [item.content for item in results] == ["Mine"]

    def test_search_memory_items_finds_items_for_a_real_non_literal_peer_id(self, store: SqliteStore):
        """The messages join makes peer-scoping work for any real peer_id
        — not just the literal string "user", which is all the old
        source_peer/subject_peer comparison ever matched."""
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="telegram:12345")
        store.store_memory_item(MemoryItem(
            content="About a trip", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=msg_id, embedding=[1.0, 0.0, 0.0],
        ))

        results = store.search_memory_items(peer_id="telegram:12345", query_embedding=[1.0, 0.0, 0.0])
        assert [item.content for item in results] == ["About a trip"]

    def test_search_memory_items_records_scoring_summary_event(self, store: SqliteStore):
        msg_id1 = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        store.store_memory_item(MemoryItem(
            content="About Bello", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=msg_id1, embedding=[1.0, 0.0, 0.0],
        ))
        msg_id2 = store.store_message(turn_id=2, role="user", content="hi", peer_id="user")
        store.store_memory_item(MemoryItem(
            content="About Mampfi", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=msg_id2, embedding=[1.0, 0.0, 0.0],
        ))

        tracer = SpyTracer()
        results = store.search_memory_items(
            peer_id="user", query_embedding=[1.0, 0.0, 0.0], limit=1, tracer=tracer, request_id="req1"
        )

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
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        store.store_memory_item(MemoryItem(
            content="About Bello", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=msg_id, embedding=[1.0, 0.0, 0.0],
        ))
        results = store.search_memory_items(peer_id="user", query_embedding=[1.0, 0.0, 0.0])
        assert len(results) == 1


class TestMemoryItemsByTurn:
    def test_returns_only_items_from_the_given_turn(self, store: SqliteStore):
        turn1_user_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        turn1_ai_id = store.store_message(turn_id=1, role="ai", content="hello", peer_id="ai")
        turn2_user_id = store.store_message(turn_id=2, role="user", content="bye", peer_id="user")

        first = MemoryItem(
            content="User greeted", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=turn1_user_id,
        )
        second = MemoryItem(
            content="AI greeted back", item_type="essence",
            source_peer="ai", subject_peer="ai", attribution_type="self_statement",
            message_id=turn1_ai_id,
        )
        other_turn = MemoryItem(
            content="User said bye", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=turn2_user_id,
        )
        store.store_memory_item(first)
        store.store_memory_item(second)
        store.store_memory_item(other_turn)

        results = store.get_memory_items_by_turn(1)

        assert [item.content for item in results] == ["User greeted", "AI greeted back"]

    def test_ordered_oldest_first(self, store: SqliteStore):
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        for content in ("third", "first", "second"):
            store.store_memory_item(MemoryItem(
                content=content, item_type="essence",
                source_peer="user", subject_peer="user", attribution_type="self_statement",
                message_id=msg_id,
            ))

        results = store.get_memory_items_by_turn(1)

        assert [item.content for item in results] == ["third", "first", "second"]
        assert results[0].id < results[1].id < results[2].id

    def test_empty_when_turn_has_no_items(self, store: SqliteStore):
        assert store.get_memory_items_by_turn(999) == []

    def test_embedding_round_trips(self, store: SqliteStore):
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        store.store_memory_item(MemoryItem(
            content="About Bello", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=msg_id, embedding=[0.1, 0.2, 0.3],
        ))

        results = store.get_memory_items_by_turn(1)

        assert results[0].embedding == [0.1, 0.2, 0.3]

    def test_no_embedding_stays_none(self, store: SqliteStore):
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        store.store_memory_item(MemoryItem(
            content="About Bello", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=msg_id,
        ))

        results = store.get_memory_items_by_turn(1)

        assert results[0].embedding is None


class TestSchema:
    def test_expected_indexes_exist(self, store: SqliteStore):
        names = {
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert {
            "idx_messages_topic",
            "idx_messages_peer_role",
            "idx_topics_peer_status",
            "idx_memory_items_subject_type",
        } <= names

    def test_user_version_is_current(self, store: SqliteStore):
        from core.stores.migrations import MIGRATIONS

        version = store._conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == max(MIGRATIONS)

    def test_active_topic_query_uses_index(self, store: SqliteStore):
        """The per-turn get_active_topic lookup must be an index seek, not
        a full table scan — regression guard for the topics index."""
        store.create_topic("Ongoing", peer_id="user", session_id="web")
        plan = store._conn.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT id FROM topics WHERE peer_id = ? AND session_id IS ? "
            "AND status = 'active' ORDER BY created_at DESC, id DESC LIMIT 1",
            ("user", "web"),
        ).fetchall()
        detail = " ".join(str(row[3]) for row in plan)
        assert "idx_topics_peer_status" in detail
        assert "SCAN" not in detail or "USING INDEX" in detail


class TestRecencyScore:
    _NOW = datetime(2026, 1, 1, tzinfo=UTC)

    def test_just_now_scores_one(self):
        assert _recency_score("2026-01-01 00:00:00", self._NOW, 30.0) == pytest.approx(1.0)

    def test_one_half_life_scores_half(self):
        # 30 days before _NOW, half_life 30 -> 0.5
        assert _recency_score("2025-12-02 00:00:00", self._NOW, 30.0) == pytest.approx(0.5)

    def test_two_half_lives_scores_quarter(self):
        # 60 days before _NOW, half_life 30 -> 0.25
        assert _recency_score("2025-11-02 00:00:00", self._NOW, 30.0) == pytest.approx(0.25)

    def test_none_scores_zero(self):
        assert _recency_score(None, self._NOW, 30.0) == 0.0

    def test_unparseable_scores_zero(self):
        assert _recency_score("not-a-date", self._NOW, 30.0) == 0.0

    def test_future_timestamp_clamps_to_one(self):
        assert _recency_score("2027-01-01 00:00:00", self._NOW, 30.0) == 1.0

    def test_naive_now_is_treated_as_utc(self):
        # A naive `now` must not raise on the aware/naive subtraction.
        assert _recency_score("2020-01-01 00:00:00", datetime(2026, 1, 1), 30.0) < 0.01


class TestTemporalScoring:
    _NOW = datetime(2026, 1, 1, tzinfo=UTC)

    def _backdate_topic(self, store: SqliteStore, topic_id: int, when: str) -> None:
        store._conn.execute("UPDATE topics SET closed_at = ? WHERE id = ?", (when, topic_id))
        store._conn.commit()

    def test_reranks_equally_relevant_topics_by_recency(self, store: SqliteStore):
        """Two topics with identical tag relevance must be ordered newest-first
        once the temporal term breaks the tie."""
        old = store.create_topic("Old", tags=["fitness"])
        store.close_topic(old, "old")
        new = store.create_topic("New", tags=["fitness"])
        store.close_topic(new, "new")
        self._backdate_topic(store, old, "2020-01-01 00:00:00")
        self._backdate_topic(store, new, "2025-12-31 00:00:00")

        results = store.search_topics(tags=["fitness"], now=self._NOW)
        assert [r["id"] for r in results] == [new, old]

    def test_does_not_surface_irrelevant_recent_topic(self, store: SqliteStore):
        """A brand-new topic with no tag/semantic/peer overlap must stay out —
        temporal re-ranks relevant candidates, it never creates relevance."""
        t = store.create_topic("Unrelated", tags=["cooking"])
        store.close_topic(t, "unrelated")  # closed_at = now, i.e. very recent

        results = store.search_topics(tags=["fitness"], now=self._NOW)
        assert results == []

    def test_reranks_equally_relevant_memory_items_by_recency(self, store: SqliteStore):
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        old_id = store.store_memory_item(MemoryItem(
            content="old", item_type="essence", source_peer="user",
            subject_peer="user", attribution_type="self_statement",
            message_id=msg_id, embedding=[1.0, 0.0, 0.0],
        ))
        new_id = store.store_memory_item(MemoryItem(
            content="new", item_type="essence", source_peer="user",
            subject_peer="user", attribution_type="self_statement",
            message_id=msg_id, embedding=[1.0, 0.0, 0.0],
        ))
        store._conn.execute("UPDATE memory_items SET created_at = ? WHERE id = ?", ("2020-01-01 00:00:00", old_id))
        store._conn.execute("UPDATE memory_items SET created_at = ? WHERE id = ?", ("2025-12-31 00:00:00", new_id))
        store._conn.commit()

        results = store.search_memory_items(peer_id="user", query_embedding=[1.0, 0.0, 0.0], now=self._NOW)
        assert [item.id for item in results] == [new_id, old_id]


class TestVecIndexAcceleratedSearch:
    """Parallel coverage for the sqlite-vec-accelerated search path —
    proves it returns results equivalent to (not just different from)
    the brute-force scan, and exercises what's unique to it (peer
    partitioning via the vector index itself, backfill on connect()).
    Skipped entirely if sqlite-vec isn't installed.
    """

    @pytest.fixture
    def accel_store(self):
        pytest.importorskip("sqlite_vec")
        tmp = tempfile.mkdtemp()
        s = SqliteStore(str(Path(tmp) / "test.db"))  # use_vec_index defaults True
        s.connect()
        assert s.vec_index.available, "sqlite-vec installed but failed to load in this environment"
        yield s
        s.close()

    def test_search_topics_accelerated_path_is_peer_isolated(self, accel_store: SqliteStore):
        mine = accel_store.create_topic("Mine", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        accel_store.close_topic(mine, "closed")
        theirs = accel_store.create_topic("Theirs", summary="", peer_id="someone_else", embedding=[1.0, 0.0, 0.0])
        accel_store.close_topic(theirs, "closed")

        results = accel_store.search_topics(query_embedding=[1.0, 0.0, 0.0], peer_id="user")

        assert [r["id"] for r in results] == [mine]

    def test_search_topics_accelerated_matches_brute_force_ranking(self, accel_store: SqliteStore):
        near = accel_store.create_topic("Near", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        accel_store.close_topic(near, "closed")
        far = accel_store.create_topic("Far", summary="", peer_id="user", embedding=[0.0, 1.0, 0.0])
        accel_store.close_topic(far, "closed")

        results = accel_store.search_topics(query_embedding=[0.9, 0.1, 0.0], peer_id="user")

        assert [r["id"] for r in results] == [near, far]

    def test_search_memory_items_accelerated_path_finds_real_peer_id(self, accel_store: SqliteStore):
        msg_id = accel_store.store_message(turn_id=1, role="user", content="hi", peer_id="telegram:12345")
        item_id = accel_store.store_memory_item(MemoryItem(
            content="about a trip", item_type="essence", source_peer="user", subject_peer="user",
            attribution_type="self_statement", message_id=msg_id, embedding=[1.0, 0.0, 0.0],
        ))

        results = accel_store.search_memory_items(peer_id="telegram:12345", query_embedding=[1.0, 0.0, 0.0])

        assert [item.id for item in results] == [item_id]

    def test_search_clusters_accelerated_path_is_peer_isolated(self, accel_store: SqliteStore):
        mine = accel_store.create_cluster("user", "Mine", "summary", embedding=[1.0, 0.0, 0.0])
        accel_store.create_cluster("someone_else", "Theirs", "summary", embedding=[1.0, 0.0, 0.0])

        results = accel_store.search_clusters(query_embedding=[1.0, 0.0, 0.0], peer_id="user")

        assert [r["id"] for r in results] == [mine]

    def test_search_cluster_candidates_accelerated_path_excludes_self_and_clustered(self, accel_store: SqliteStore):
        new_topic = accel_store.create_topic("New", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        accel_store.close_topic(new_topic, "closed")
        other_topic = accel_store.create_topic("Other", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        accel_store.close_topic(other_topic, "closed")
        clustered_topic = accel_store.create_topic("Clustered", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        accel_store.close_topic(clustered_topic, "closed")
        cid = accel_store.create_cluster("user", "Existing cluster", "summary", embedding=[1.0, 0.0, 0.0])
        accel_store.assign_topic_to_cluster(clustered_topic, cid)

        candidates = accel_store.search_cluster_candidates(new_topic, "user")

        kinds = {(c["candidate_kind"], c["id"]) for c in candidates}
        assert kinds == {("topic", other_topic), ("cluster", cid)}

    def test_write_paths_populate_the_index_immediately(self, accel_store: SqliteStore):
        """A topic created through create_topic (not a manual upsert) must
        be findable through the accelerated path right away — proves the
        write-path wiring, not just VecIndex in isolation."""
        tid = accel_store.create_topic("Bike", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        accel_store.close_topic(tid, "closed")

        assert accel_store.vec_index.knn("vec_topics", "user", [1.0, 0.0, 0.0], k=5) == [
            (tid, pytest.approx(1.0))
        ]

    def test_backfill_on_connect_indexes_pre_existing_embeddings(self):
        """Data written while the index was unavailable must become
        searchable once sqlite-vec loads later — not silently invisible
        forever. See SqliteStore._backfill_vec_index."""
        pytest.importorskip("sqlite_vec")
        tmp = tempfile.mkdtemp()
        db_path = str(Path(tmp) / "test.db")

        without_index = SqliteStore(db_path, use_vec_index=False)
        without_index.connect()
        tid = without_index.create_topic("Bike", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        without_index.close_topic(tid, "closed")
        without_index.close()

        with_index = SqliteStore(db_path)  # use_vec_index defaults True
        with_index.connect()
        try:
            assert with_index.vec_index.available
            hits = with_index.vec_index.knn("vec_topics", "user", [1.0, 0.0, 0.0], k=5)
            assert hits == [(tid, pytest.approx(1.0))]
        finally:
            with_index.close()

    def test_backfill_does_not_leave_an_uncommitted_transaction(self):
        """Real deployment failure: _backfill_vec_index's DELETE+INSERT
        statements open an implicit transaction; if connect() returns
        without committing it, a second, independent connection to the
        same file hits 'database is locked' immediately (SQLite's
        default busy_timeout is 0, no retry) — this is exactly what
        happened when EidolonMemoryProvider.initialize() raced a second
        connection against a real deployed DB. A second connection must
        be able to write right after connect() returns."""
        pytest.importorskip("sqlite_vec")
        tmp = tempfile.mkdtemp()
        db_path = str(Path(tmp) / "test.db")

        seed = SqliteStore(db_path, use_vec_index=False)
        seed.connect()
        seed.create_topic("Bike", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        seed.close()

        store = SqliteStore(db_path)  # use_vec_index defaults True, triggers a real backfill
        store.connect()
        try:
            second = sqlite3.connect(db_path, timeout=0)
            try:
                second.execute(
                    "INSERT INTO messages (turn_id, role, content, peer_id) VALUES (?, ?, ?, ?)",
                    (1, "user", "hello", "user"),
                )
                second.commit()
            finally:
                second.close()
        finally:
            store.close()
