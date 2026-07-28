"""Backend-agnostic StorageProvider contract tests.

Runs the same assertions against both SqliteStore and PostgresStore (the
``store``/``store_backend`` fixtures in tests/conftest.py are parametrized
over both) — StorageProvider being a Protocol (core/interfaces.py) means
callers can't tell backends apart, so the test suite enforces that
mechanically rather than trusting two independently-maintained files to
stay in sync.

Backend-specific behavior lives elsewhere: tests/core/test_sqlite_store.py
(PRAGMA/migration mechanics, the optional sqlite-vec accelerated path)
and tests/core/test_postgres_store_specifics.py (schema_migrations
mechanics).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.interfaces import StorageProvider
from core.models import Fact, MemoryItem, MemoryItemBudget
from tests.mocks import SpyTracer


class TestFacts:
    def test_set_and_get_facts(self, store: StorageProvider):
        store.set_fact("user", "basic", "name", "Alex", "user")
        store.set_fact("user", "basic", "age", "42", "user")

        facts = store.get_facts("user")
        assert len(facts) == 2
        assert all(isinstance(f, Fact) for f in facts)
        assert facts[0].key == "age"  # sorted by category, key
        assert facts[1].key == "name"

    def test_get_facts_empty(self, store: StorageProvider):
        assert store.get_facts("unknown") == []

    def test_set_fact_overwrites(self, store: StorageProvider):
        store.set_fact("user", "basic", "name", "Alex", "user")
        store.set_fact("user", "basic", "name", "Alex Morgan", "ai")

        facts = store.get_facts("user")
        assert len(facts) == 1
        assert facts[0].value == "Alex Morgan"
        assert facts[0].source == "ai"

    def test_delete_fact(self, store: StorageProvider):
        store.set_fact("user", "basic", "name", "Alex", "user")
        store.delete_fact("user", "basic", "name")
        assert store.get_facts("user") == []

    def test_delete_fact_missing_is_a_noop(self, store: StorageProvider):
        store.delete_fact("user", "basic", "name")  # must not raise
        assert store.get_facts("user") == []

    def test_set_fact_defaults_importance_to_medium(self, store: StorageProvider):
        store.set_fact("user", "basic", "name", "Alex", "user")
        assert store.get_facts("user")[0].importance == "medium"

    def test_set_fact_stores_and_overwrites_importance(self, store: StorageProvider):
        store.set_fact("user", "work", "employer", "Acme", "ai", importance="high")
        assert store.get_facts("user")[0].importance == "high"

        store.set_fact("user", "work", "employer", "Acme Corp", "ai", importance="low")
        facts = store.get_facts("user")
        assert len(facts) == 1
        assert facts[0].importance == "low"

    def test_get_facts_orders_by_importance_then_category_key(self, store: StorageProvider):
        store.set_fact("user", "z", "low_fact", "v", "ai", importance="low")
        store.set_fact("user", "a", "high_fact", "v", "ai", importance="high")
        store.set_fact("user", "m", "medium_fact", "v", "ai", importance="medium")

        facts = store.get_facts("user")
        assert [f.importance for f in facts] == ["high", "medium", "low"]

    def test_get_all_facts_returns_facts_from_every_peer(self, store: StorageProvider):
        store.set_fact("u1", "basic", "name", "Alex", "user")
        store.set_fact("u2", "basic", "name", "Sam", "user")

        facts = store.get_all_facts()
        assert {f.peer_id for f in facts} == {"u1", "u2"}

    def test_get_all_facts_empty(self, store: StorageProvider):
        assert store.get_all_facts() == []


class TestFactsConsolidation:
    def test_count_user_turns_only_counts_user_role(self, store: StorageProvider):
        store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        store.store_message(turn_id=1, role="ai", content="hello", peer_id="ai")
        store.store_message(turn_id=2, role="user", content="hi again", peer_id="user")

        assert store.count_user_turns("user") == 2
        assert store.count_user_turns("someone_else") == 0

    def test_get_fact_items_since_filters_by_subject_and_id_not_type(self, store: StorageProvider):
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
        essence_id = store.store_memory_item(essence_item)
        store.store_memory_item(other_subject_item)

        # Every item_type comes back — only subject_peer/id gate. A concrete
        # fact is often mislabeled "essence" at extraction time (see
        # get_fact_items_since's docstring); filtering by type here would
        # permanently exclude it from ever reaching fact consolidation.
        results = store.get_fact_items_since("user", 0)
        assert [r.content for r in results] == ["likes cats", "chatted about weather"]
        assert [r.id for r in results] == [fact_id, essence_id]

        assert [r.content for r in store.get_fact_items_since("user", fact_id)] == ["chatted about weather"]
        assert store.get_fact_items_since("user", essence_id) == []

    def test_consolidation_checkpoint_defaults_to_zero(self, store: StorageProvider):
        assert store.get_consolidation_checkpoint("user") == 0

    def test_consolidation_checkpoint_set_and_get(self, store: StorageProvider):
        store.set_consolidation_checkpoint("user", 5)
        assert store.get_consolidation_checkpoint("user") == 5

    def test_consolidation_checkpoint_upserts(self, store: StorageProvider):
        store.set_consolidation_checkpoint("user", 5)
        store.set_consolidation_checkpoint("user", 9)
        assert store.get_consolidation_checkpoint("user") == 9

    def test_consolidation_checkpoint_is_per_peer(self, store: StorageProvider):
        store.set_consolidation_checkpoint("user", 5)
        assert store.get_consolidation_checkpoint("other_peer") == 0


class TestTopics:
    def test_create_and_get_topic(self, store: StorageProvider):
        tid = store.create_topic("Bello Features", summary="Discussion about Bello", tags=["bello", "fitness"])
        assert isinstance(tid, int)

        topic = store.get_topic(tid)
        assert topic is not None
        assert topic["label"] == "Bello Features"
        assert topic["tags"] == ["bello", "fitness"]
        assert topic["status"] == "active"

    def test_get_topic_not_found(self, store: StorageProvider):
        assert store.get_topic(999) is None

    def test_close_topic(self, store: StorageProvider):
        tid = store.create_topic("Test", summary="")
        store.close_topic(tid, "Final summary")
        topic = store.get_topic(tid)
        assert topic["status"] == "closed"
        assert topic["summary"] == "Final summary"
        assert topic["closed_at"] is not None

    def test_update_topic_tags_replaces_tags(self, store: StorageProvider):
        tid = store.create_topic("Test", tags=["old"])
        store.update_topic_tags(tid, ["fitness", "gym"])
        assert store.get_topic(tid)["tags"] == ["fitness", "gym"]

    def test_update_topic_label_replaces_label(self, store: StorageProvider):
        tid = store.create_topic("")
        store.update_topic_label(tid, "Buying a bike")
        assert store.get_topic(tid)["label"] == "Buying a bike"

    def test_count_messages_by_topic_counts_only_that_topics_messages(self, store: StorageProvider):
        tid_a = store.create_topic("Topic A", peer_id="user")
        tid_b = store.create_topic("Topic B", peer_id="user")
        store.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=tid_a)
        store.store_message(turn_id=1, role="ai", content="hello", peer_id="ai", topic_id=tid_a)
        store.store_message(turn_id=2, role="user", content="other", peer_id="user", topic_id=tid_b)

        assert store.count_messages_by_topic(tid_a) == 2
        assert store.count_messages_by_topic(tid_b) == 1

    def test_count_messages_by_topic_zero_when_none(self, store: StorageProvider):
        tid = store.create_topic("Fresh", peer_id="user")
        assert store.count_messages_by_topic(tid) == 0

    def test_store_message_increments_persisted_message_count(self, store: StorageProvider):
        """topics.message_count is a real, kept-in-sync column (see
        store_message()) — distinct from count_messages_by_topic, which
        recomputes live. Drives search_topics/get_recent_topics'
        min_message_count filter."""
        tid = store.create_topic("Bello", peer_id="user")
        assert store.get_topic(tid)["message_count"] == 0

        store.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=tid)
        assert store.get_topic(tid)["message_count"] == 1

        store.store_message(turn_id=1, role="ai", content="hello", peer_id="ai", topic_id=tid)
        assert store.get_topic(tid)["message_count"] == 2

    def test_store_message_without_topic_id_does_not_crash(self, store: StorageProvider):
        # No topic to increment — must be a clean no-op, not an error.
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=None)
        assert isinstance(msg_id, int)

    def test_get_active_topic_returns_open_topic(self, store: StorageProvider):
        tid = store.create_topic("Ongoing", summary="", peer_id="user")
        assert store.get_active_topic("user") == tid

    def test_get_active_topic_none_when_closed(self, store: StorageProvider):
        tid = store.create_topic("Done", summary="", peer_id="user")
        store.close_topic(tid, "Wrapped up")
        assert store.get_active_topic("user") is None

    def test_get_active_topic_none_for_unknown_peer(self, store: StorageProvider):
        store.create_topic("Ongoing", summary="", peer_id="user")
        assert store.get_active_topic("someone_else") is None

    def test_get_active_topic_picks_most_recent(self, store: StorageProvider):
        store.create_topic("Older", summary="", peer_id="user")
        newer = store.create_topic("Newer", summary="", peer_id="user")
        assert store.get_active_topic("user") == newer

    def test_get_active_topic_scoped_per_session(self, store: StorageProvider):
        """A peer's concurrent sessions (web, Telegram, CLI, ...) must each
        get their own active topic, not share whichever is newest."""
        web_topic = store.create_topic("Web thread", summary="", peer_id="user", session_id="web")
        telegram_topic = store.create_topic("Telegram thread", summary="", peer_id="user", session_id="telegram")

        assert store.get_active_topic("user", session_id="web") == web_topic
        assert store.get_active_topic("user", session_id="telegram") == telegram_topic

    def test_get_active_topic_with_no_session_id_only_matches_sessionless_topics(self, store: StorageProvider):
        """None is not a wildcard across sessions — it only matches topics
        created with no session_id (legacy rows, single-session callers
        like the CLI)."""
        store.create_topic("Web thread", summary="", peer_id="user", session_id="web")
        sessionless = store.create_topic("CLI thread", summary="", peer_id="user")

        assert store.get_active_topic("user") == sessionless
        assert store.get_active_topic("user", session_id="web") != sessionless

    def test_create_topic_stores_session_id(self, store: StorageProvider):
        tid = store.create_topic("Web thread", summary="", peer_id="user", session_id="web")
        assert store.get_active_topic("user", session_id="web") == tid

    def test_search_topics_by_tags(self, store: StorageProvider):
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

    def test_search_topics_respects_custom_weights(self, store: StorageProvider):
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

    def test_search_topics_applies_person_weight_for_matching_peer(self, store: StorageProvider):
        """A topic with no tag/semantic match still surfaces on peer_id
        alone once the person weight applies."""
        store.create_topic("Bello", summary="", peer_id="user")
        store.close_topic(1, "")

        results = store.search_topics(peer_id="user")
        assert len(results) == 1
        assert results[0]["peer_id"] == "user"

    def test_search_topics_excludes_other_peers_even_with_strong_match(self, store: StorageProvider):
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

    def test_search_topics_with_no_peer_id_searches_every_peer(self, store: StorageProvider):
        """peer_id=None keeps the pre-existing 'search everyone' behavior
        for callers that intentionally omit it — not a new default."""
        store.create_topic("Bello", summary="", tags=["fitness"], peer_id="someone_else")
        store.close_topic(1, "")
        store.create_topic("Bello 2", summary="", tags=["fitness"], peer_id="user")
        store.close_topic(2, "")

        results = store.search_topics(tags=["fitness"])

        assert {r["id"] for r in results} == {1, 2}

    def test_search_topics_returns_correct_created_at(self, store: StorageProvider):
        store.create_topic("Bello", summary="", tags=["fitness"])
        store.close_topic(1, "")

        results = store.search_topics(tags=["fitness"])
        assert len(results) == 1
        assert results[0]["created_at"] is not None
        assert results[0]["status"] == "closed"

    def test_search_topics_records_scoring_summary_event(self, store: StorageProvider):
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

    def test_search_topics_score_breakdown_values(self, store: StorageProvider):
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

    def test_search_topics_scoring_summary_caps_top_candidates(self, store: StorageProvider):
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

    def test_search_topics_no_tracer_does_not_raise(self, store: StorageProvider):
        store.create_topic("Bello", summary="", tags=["fitness"])
        store.close_topic(1, "")
        results = store.search_topics(tags=["fitness"])
        assert len(results) == 1

    def test_search_topics_excludes_clustered_topics(self, store: StorageProvider):
        """Once a topic joins a cluster, its consolidated summary
        represents it — it must not also compete individually."""
        tid = store.create_topic("Bello", summary="", tags=["fitness"], peer_id="user")
        store.close_topic(tid, "")
        cid = store.create_cluster("user", "Fitness", "summary")
        store.assign_topic_to_cluster(tid, cid)

        assert store.search_topics(tags=["fitness"]) == []

    def test_search_topics_min_message_count_excludes_thin_topics(self, store: StorageProvider):
        thin = store.create_topic("Thin", summary="", tags=["fitness"], peer_id="user")
        store.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=thin)
        store.close_topic(thin, "")

        thick = store.create_topic("Thick", summary="", tags=["fitness"], peer_id="user")
        for _ in range(3):
            store.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=thick)
        store.close_topic(thick, "")

        results = store.search_topics(tags=["fitness"], min_message_count=2)
        assert [r["id"] for r in results] == [thick]

    def test_search_topics_min_message_count_boundary_is_strictly_greater_than(
        self, store: StorageProvider,
    ):
        """A topic with exactly `min_message_count` messages must be
        excluded — the filter is `>`, not `>=`."""
        tid = store.create_topic("Exactly at threshold", summary="", tags=["fitness"], peer_id="user")
        for _ in range(3):
            store.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=tid)
        store.close_topic(tid, "")

        assert store.search_topics(tags=["fitness"], min_message_count=3) == []
        assert len(store.search_topics(tags=["fitness"], min_message_count=2)) == 1

    def test_search_topics_default_min_message_count_applies_no_filtering(self, store: StorageProvider):
        tid = store.create_topic("Untouched", summary="", tags=["fitness"], peer_id="user")
        store.close_topic(tid, "")  # zero messages

        assert [r["id"] for r in store.search_topics(tags=["fitness"])] == [tid]


class TestGetStaleActiveTopics:
    def test_stale_when_last_message_predates_cutoff(self, store_backend):
        store = store_backend.store
        tid = store.create_topic("Old chat", summary="", peer_id="user")
        mid = store.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=tid)
        store_backend.backdate("messages", mid, "created_at", datetime(2020, 1, 1, tzinfo=UTC))

        stale = store.get_stale_active_topics(datetime(2025, 1, 1, tzinfo=UTC))

        assert [t["id"] for t in stale] == [tid]
        assert stale[0]["status"] == "active"

    def test_not_stale_when_last_message_after_cutoff(self, store: StorageProvider):
        tid = store.create_topic("Fresh chat", summary="", peer_id="user")
        store.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=tid)

        # cutoff far in the past — the just-stored message is newer than it
        assert store.get_stale_active_topics(datetime(2020, 1, 1, tzinfo=UTC)) == []

    def test_falls_back_to_topic_created_at_when_no_messages(self, store_backend):
        store = store_backend.store
        tid = store.create_topic("Never used", summary="", peer_id="user")
        store_backend.backdate("topics", tid, "created_at", datetime(2020, 1, 1, tzinfo=UTC))

        stale = store.get_stale_active_topics(datetime(2025, 1, 1, tzinfo=UTC))

        assert [t["id"] for t in stale] == [tid]

    def test_excludes_closed_topics(self, store_backend):
        store = store_backend.store
        tid = store.create_topic("Done", summary="", peer_id="user")
        store.close_topic(tid, "wrapped up")
        store_backend.backdate("topics", tid, "created_at", datetime(2020, 1, 1, tzinfo=UTC))

        assert store.get_stale_active_topics(datetime(2025, 1, 1, tzinfo=UTC)) == []

    def test_filters_by_peer_id(self, store_backend):
        store = store_backend.store
        mine = store.create_topic("Mine", summary="", peer_id="user")
        theirs = store.create_topic("Theirs", summary="", peer_id="someone_else")
        store_backend.backdate("topics", mine, "created_at", datetime(2020, 1, 1, tzinfo=UTC))
        store_backend.backdate("topics", theirs, "created_at", datetime(2020, 1, 1, tzinfo=UTC))

        stale = store.get_stale_active_topics(datetime(2025, 1, 1, tzinfo=UTC), peer_id="user")

        assert [t["id"] for t in stale] == [mine]

    def test_ordered_oldest_first(self, store_backend):
        store = store_backend.store
        newer = store.create_topic("Newer", summary="", peer_id="user")
        older = store.create_topic("Older", summary="", peer_id="user")
        store_backend.backdate("topics", newer, "created_at", datetime(2020, 6, 1, tzinfo=UTC))
        store_backend.backdate("topics", older, "created_at", datetime(2020, 1, 1, tzinfo=UTC))

        stale = store.get_stale_active_topics(datetime(2025, 1, 1, tzinfo=UTC))

        assert [t["id"] for t in stale] == [older, newer]


class TestGetRecentTopics:
    def test_returns_closed_topics_newest_first(self, store: StorageProvider):
        older = store.create_topic("Older", summary="", peer_id="user")
        newer = store.create_topic("Newer", summary="", peer_id="user")
        store.close_topic(older, "Older summary")
        store.close_topic(newer, "Newer summary")

        results = store.get_recent_topics("user", limit=10)
        assert [t["id"] for t in results] == [newer, older]

    def test_excludes_active_topics(self, store: StorageProvider):
        closed = store.create_topic("Closed", summary="", peer_id="user")
        store.close_topic(closed, "Wrapped up")
        store.create_topic("Still open", summary="", peer_id="user")

        results = store.get_recent_topics("user", limit=10)
        assert [t["id"] for t in results] == [closed]

    def test_filters_by_peer_id(self, store: StorageProvider):
        mine = store.create_topic("Mine", summary="", peer_id="user")
        store.close_topic(mine, "")
        theirs = store.create_topic("Theirs", summary="", peer_id="someone_else")
        store.close_topic(theirs, "")

        results = store.get_recent_topics("user", limit=10)
        assert [t["id"] for t in results] == [mine]

    def test_respects_limit(self, store: StorageProvider):
        for i in range(5):
            tid = store.create_topic(f"Topic {i}", summary="", peer_id="user")
            store.close_topic(tid, "")

        results = store.get_recent_topics("user", limit=2)
        assert len(results) == 2

    def test_empty_when_no_closed_topics(self, store: StorageProvider):
        assert store.get_recent_topics("user", limit=10) == []

    def test_excludes_clustered_topics(self, store: StorageProvider):
        clustered = store.create_topic("Clustered", summary="", peer_id="user")
        store.close_topic(clustered, "")
        cid = store.create_cluster("user", "Thread", "summary")
        store.assign_topic_to_cluster(clustered, cid)
        unclustered = store.create_topic("Unclustered", summary="", peer_id="user")
        store.close_topic(unclustered, "")

        results = store.get_recent_topics("user", limit=10)
        assert [t["id"] for t in results] == [unclustered]

    def test_min_message_count_excludes_thin_topics(self, store: StorageProvider):
        thin = store.create_topic("Thin", summary="", peer_id="user")
        store.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=thin)
        store.close_topic(thin, "")

        thick = store.create_topic("Thick", summary="", peer_id="user")
        for _ in range(3):
            store.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=thick)
        store.close_topic(thick, "")

        results = store.get_recent_topics("user", limit=10, min_message_count=2)
        assert [t["id"] for t in results] == [thick]

    def test_min_message_count_boundary_is_strictly_greater_than(self, store: StorageProvider):
        tid = store.create_topic("Exactly at threshold", summary="", peer_id="user")
        for _ in range(3):
            store.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=tid)
        store.close_topic(tid, "")

        assert store.get_recent_topics("user", limit=10, min_message_count=3) == []
        assert len(store.get_recent_topics("user", limit=10, min_message_count=2)) == 1

    def test_default_min_message_count_applies_no_filtering(self, store: StorageProvider):
        tid = store.create_topic("Untouched", summary="", peer_id="user")
        store.close_topic(tid, "")  # zero messages

        assert [t["id"] for t in store.get_recent_topics("user", limit=10)] == [tid]


class TestGetClosedTopicsSince:
    def test_returns_topics_closed_at_or_after_since(self, store_backend):
        store = store_backend.store
        tid = store.create_topic("Recent", summary="", peer_id="user")
        store.close_topic(tid, "wrapped up")
        store_backend.backdate("topics", tid, "closed_at", datetime(2025, 6, 1, tzinfo=UTC))

        results = store.get_closed_topics_since(datetime(2025, 1, 1, tzinfo=UTC))

        assert [t["id"] for t in results] == [tid]
        assert results[0]["peer_id"] == "user"
        assert results[0]["message_count"] == 0

    def test_excludes_topics_closed_before_since(self, store_backend):
        store = store_backend.store
        tid = store.create_topic("Old", summary="", peer_id="user")
        store.close_topic(tid, "wrapped up")
        store_backend.backdate("topics", tid, "closed_at", datetime(2020, 1, 1, tzinfo=UTC))

        assert store.get_closed_topics_since(datetime(2025, 1, 1, tzinfo=UTC)) == []

    def test_excludes_active_topics(self, store: StorageProvider):
        store.create_topic("Still open", summary="", peer_id="user")

        assert store.get_closed_topics_since(datetime(2020, 1, 1, tzinfo=UTC)) == []

    def test_ordered_oldest_closed_first(self, store_backend):
        store = store_backend.store
        newer = store.create_topic("Newer", summary="", peer_id="user")
        store.close_topic(newer, "")
        store_backend.backdate("topics", newer, "closed_at", datetime(2025, 6, 1, tzinfo=UTC))
        older = store.create_topic("Older", summary="", peer_id="user")
        store.close_topic(older, "")
        store_backend.backdate("topics", older, "closed_at", datetime(2025, 1, 1, tzinfo=UTC))

        results = store.get_closed_topics_since(datetime(2020, 1, 1, tzinfo=UTC))

        assert [t["id"] for t in results] == [older, newer]

    def test_empty_when_no_closed_topics(self, store: StorageProvider):
        assert store.get_closed_topics_since(datetime(2020, 1, 1, tzinfo=UTC)) == []


class TestListTopics:
    def test_returns_every_topic_regardless_of_status(self, store: StorageProvider):
        open_id = store.create_topic("Open", summary="", peer_id="user")
        closed_id = store.create_topic("Closed", summary="", peer_id="user")
        store.close_topic(closed_id, "wrapped up")

        results = store.list_topics()

        assert {t["id"] for t in results} == {open_id, closed_id}

    def test_filters_by_status(self, store: StorageProvider):
        open_id = store.create_topic("Open", summary="", peer_id="user")
        closed_id = store.create_topic("Closed", summary="", peer_id="user")
        store.close_topic(closed_id, "wrapped up")

        assert [t["id"] for t in store.list_topics(status="closed")] == [closed_id]
        assert [t["id"] for t in store.list_topics(status="active")] == [open_id]

    def test_filters_by_peer_id(self, store: StorageProvider):
        u1 = store.create_topic("Mine", summary="", peer_id="u1")
        store.create_topic("Theirs", summary="", peer_id="u2")

        assert [t["id"] for t in store.list_topics(peer_id="u1")] == [u1]

    def test_empty_when_no_topics(self, store: StorageProvider):
        assert store.list_topics() == []


class TestClusters:
    def test_create_and_get_cluster(self, store: StorageProvider):
        cid = store.create_cluster("user", "Bike purchase", "User wants a new bike", tags=["bike"])
        assert isinstance(cid, int)

        cluster = store.get_cluster(cid)
        assert cluster is not None
        assert cluster["peer_id"] == "user"
        assert cluster["label"] == "Bike purchase"
        assert cluster["summary"] == "User wants a new bike"
        assert cluster["tags"] == ["bike"]
        assert cluster["topic_count"] == 0

    def test_get_cluster_not_found(self, store: StorageProvider):
        assert store.get_cluster(999) is None

    def test_update_cluster_rewrites_summary_in_place(self, store: StorageProvider):
        cid = store.create_cluster("user", "Bike", "old summary")
        store.update_cluster(cid, summary="new consolidated summary")

        cluster = store.get_cluster(cid)
        assert cluster["summary"] == "new consolidated summary"
        assert cluster["label"] == "Bike"  # unchanged when not given

    def test_update_cluster_optional_fields_only_set_when_given(self, store: StorageProvider):
        cid = store.create_cluster("user", "Bike", "summary", tags=["bike"])
        store.update_cluster(cid, summary="new summary", label="Bike saga", tags=["bike", "shopping"])

        cluster = store.get_cluster(cid)
        assert cluster["label"] == "Bike saga"
        assert cluster["tags"] == ["bike", "shopping"]

    def test_assign_topic_to_cluster_sets_id_and_increments_count(self, store: StorageProvider):
        tid = store.create_topic("Considering a bike", summary="", peer_id="user")
        store.close_topic(tid, "thinking about it")
        cid = store.create_cluster("user", "Bike", "summary")

        store.assign_topic_to_cluster(tid, cid)

        assert store.get_cluster(cid)["topic_count"] == 1
        assert [m["id"] for m in store.get_topics_by_cluster(cid)] == [tid]

    def test_get_topics_by_cluster_oldest_first(self, store: StorageProvider):
        cid = store.create_cluster("user", "Bike", "summary")
        t1 = store.create_topic("First", summary="", peer_id="user")
        store.close_topic(t1, "first summary")
        t2 = store.create_topic("Second", summary="", peer_id="user")
        store.close_topic(t2, "second summary")
        store.assign_topic_to_cluster(t1, cid)
        store.assign_topic_to_cluster(t2, cid)

        assert [m["id"] for m in store.get_topics_by_cluster(cid)] == [t1, t2]

    def test_get_recent_topics_in_cluster_newest_first_and_limited(self, store: StorageProvider):
        cid = store.create_cluster("user", "Bike", "summary")
        t1 = store.create_topic("First", summary="", peer_id="user")
        store.close_topic(t1, "first")
        t2 = store.create_topic("Second", summary="", peer_id="user")
        store.close_topic(t2, "second")
        store.assign_topic_to_cluster(t1, cid)
        store.assign_topic_to_cluster(t2, cid)

        recent = store.get_recent_topics_in_cluster(cid, limit=1)
        assert [r["id"] for r in recent] == [t2]

    def test_search_clusters_scores_by_semantic_similarity(self, store: StorageProvider):
        store.create_cluster("user", "Bike", "summary", embedding=[1.0, 0.0, 0.0])
        results = store.search_clusters(query_embedding=[1.0, 0.0, 0.0], peer_id="user")
        assert len(results) == 1

    def test_search_clusters_hard_filters_by_peer_id(self, store: StorageProvider):
        """Unlike search_topics's soft person-weight bonus, peer_id here
        is a real SQL filter — clusters must never merge across peers."""
        store.create_cluster("someone_else", "Bike", "summary", embedding=[1.0, 0.0, 0.0])
        results = store.search_clusters(query_embedding=[1.0, 0.0, 0.0], peer_id="user")
        assert results == []

    def test_search_clusters_empty_without_peer_id(self, store: StorageProvider):
        store.create_cluster("user", "Bike", "summary", embedding=[1.0, 0.0, 0.0])
        assert store.search_clusters(query_embedding=[1.0, 0.0, 0.0]) == []

    def test_search_clusters_records_scoring_summary_event(self, store: StorageProvider):
        store.create_cluster("user", "Bike", "summary", embedding=[1.0, 0.0, 0.0])
        tracer = SpyTracer()
        store.search_clusters(query_embedding=[1.0, 0.0, 0.0], peer_id="user", tracer=tracer, request_id="req1")

        events = tracer.events_of_type("retrieval_scoring_summary")
        assert len(events) == 1
        assert events[0]["payload"]["candidate_type"] == "cluster"


class TestListClusters:
    def test_returns_clusters_from_every_peer(self, store: StorageProvider):
        c1 = store.create_cluster("u1", "Bike", "summary")
        c2 = store.create_cluster("u2", "Trip", "summary")

        assert {c["id"] for c in store.list_clusters()} == {c1, c2}

    def test_filters_by_peer_id(self, store: StorageProvider):
        mine = store.create_cluster("u1", "Bike", "summary")
        store.create_cluster("u2", "Trip", "summary")

        assert [c["id"] for c in store.list_clusters(peer_id="u1")] == [mine]

    def test_empty_when_no_clusters(self, store: StorageProvider):
        assert store.list_clusters() == []


class TestSearchClusterCandidates:
    def test_finds_unclustered_topic_with_matching_embedding(self, store: StorageProvider):
        other = store.create_topic("Other bike topic", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        store.close_topic(other, "other summary")
        new = store.create_topic("New bike topic", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        store.close_topic(new, "new summary")

        candidates = store.search_cluster_candidates(new, "user")
        assert len(candidates) == 1
        assert candidates[0]["candidate_kind"] == "topic"
        assert candidates[0]["id"] == other

    def test_finds_existing_cluster_with_matching_embedding(self, store: StorageProvider):
        cid = store.create_cluster("user", "Bike", "summary", embedding=[1.0, 0.0, 0.0])
        new = store.create_topic("New bike topic", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        store.close_topic(new, "new summary")

        candidates = store.search_cluster_candidates(new, "user")
        assert len(candidates) == 1
        assert candidates[0]["candidate_kind"] == "cluster"
        assert candidates[0]["id"] == cid

    def test_excludes_the_topic_itself(self, store: StorageProvider):
        new = store.create_topic("Solo topic", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        store.close_topic(new, "summary")

        assert store.search_cluster_candidates(new, "user") == []

    def test_excludes_already_clustered_topics(self, store: StorageProvider):
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

    def test_peer_scoped(self, store: StorageProvider):
        other_peer_topic = store.create_topic(
            "Other peer's topic", summary="", peer_id="someone_else", embedding=[1.0, 0.0, 0.0]
        )
        store.close_topic(other_peer_topic, "summary")
        new = store.create_topic("My topic", summary="", peer_id="user", embedding=[1.0, 0.0, 0.0])
        store.close_topic(new, "summary")

        assert store.search_cluster_candidates(new, "user") == []

    def test_empty_when_topic_has_no_embedding(self, store: StorageProvider):
        new = store.create_topic("No embedding", summary="", peer_id="user")
        store.close_topic(new, "summary")
        assert store.search_cluster_candidates(new, "user") == []

    def test_records_scoring_summary_event(self, store: StorageProvider):
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
    def test_returns_most_recent_within_limit_oldest_first(self, store: StorageProvider):
        tid = store.create_topic("Bello", summary="")
        for i in range(5):
            store.store_message(turn_id=i, role="user", content=f"msg {i}", peer_id="user", topic_id=tid)

        recent = store.get_recent_messages_by_topic(tid, limit=2)
        assert [m["content"] for m in recent] == ["msg 3", "msg 4"]

    def test_returns_all_when_fewer_than_limit(self, store: StorageProvider):
        tid = store.create_topic("Bello", summary="")
        store.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=tid)

        recent = store.get_recent_messages_by_topic(tid, limit=10)
        assert len(recent) == 1

    def test_empty_for_topic_with_no_messages(self, store: StorageProvider):
        tid = store.create_topic("Bello", summary="")
        assert store.get_recent_messages_by_topic(tid, limit=4) == []

    def test_only_messages_for_given_topic(self, store: StorageProvider):
        t1 = store.create_topic("First", summary="")
        t2 = store.create_topic("Second", summary="")
        store.store_message(turn_id=1, role="user", content="about first", peer_id="user", topic_id=t1)
        store.store_message(turn_id=2, role="user", content="about second", peer_id="user", topic_id=t2)

        recent = store.get_recent_messages_by_topic(t1, limit=10)
        assert [m["content"] for m in recent] == ["about first"]


class TestGetMessagesByRequestId:
    def test_returns_both_messages_from_one_turn(self, store: StorageProvider):
        tid = store.create_topic("Bello", summary="")
        store.store_message(
            turn_id=1, role="user", content="hi", peer_id="user", topic_id=tid, request_id="req1",
        )
        store.store_message(
            turn_id=1, role="ai", content="hello", peer_id="ai", topic_id=tid, request_id="req1",
        )

        messages = store.get_messages_by_request_id("req1")
        assert [m["role"] for m in messages] == ["user", "ai"]
        assert [m["content"] for m in messages] == ["hi", "hello"]
        assert all(m["topic_id"] == tid for m in messages)

    def test_empty_for_unknown_request_id(self, store: StorageProvider):
        assert store.get_messages_by_request_id("nope") == []

    def test_does_not_include_messages_from_other_requests(self, store: StorageProvider):
        store.store_message(turn_id=1, role="user", content="hi", peer_id="user", request_id="req1")
        store.store_message(turn_id=2, role="user", content="bye", peer_id="user", request_id="req2")

        messages = store.get_messages_by_request_id("req1")
        assert [m["content"] for m in messages] == ["hi"]

    def test_messages_without_request_id_are_not_returned(self, store: StorageProvider):
        store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        assert store.get_messages_by_request_id("") == []


class TestGetRecentTurns:
    def test_lists_user_messages_with_request_id_most_recent_first(self, store: StorageProvider):
        tid = store.create_topic("Bello", summary="")
        store.store_message(turn_id=1, role="user", content="first", peer_id="user", topic_id=tid, request_id="req1")
        store.store_message(turn_id=1, role="ai", content="first reply", peer_id="ai", topic_id=tid, request_id="req1")
        store.store_message(turn_id=2, role="user", content="second", peer_id="user", topic_id=tid, request_id="req2")

        turns = store.get_recent_turns(10)
        assert [t["request_id"] for t in turns] == ["req2", "req1"]
        assert turns[0]["content"] == "second"

    def test_excludes_messages_without_request_id(self, store: StorageProvider):
        store.store_message(turn_id=1, role="user", content="untracked", peer_id="user")
        assert store.get_recent_turns(10) == []

    def test_excludes_ai_role_messages(self, store: StorageProvider):
        store.store_message(turn_id=1, role="ai", content="reply", peer_id="ai", request_id="req1")
        assert store.get_recent_turns(10) == []

    def test_respects_limit(self, store: StorageProvider):
        for i in range(3):
            store.store_message(turn_id=i, role="user", content=f"msg {i}", peer_id="user", request_id=f"req{i}")

        assert len(store.get_recent_turns(2)) == 2

    def test_filters_by_peer_id(self, store: StorageProvider):
        store.store_message(turn_id=1, role="user", content="mine", peer_id="user", request_id="req1")
        store.store_message(turn_id=2, role="user", content="theirs", peer_id="telegram:999", request_id="req2")

        turns = store.get_recent_turns(10, peer_id="user")
        assert [t["request_id"] for t in turns] == ["req1"]


class TestMemoryItems:
    def test_store_and_search(self, store: StorageProvider):
        item = MemoryItem(
            content="User talked about Bello features",
            item_type="essence",
            source_peer="user",
            subject_peer="user",
            attribution_type="self_statement",
        )
        mid = store.store_memory_item(item)
        assert isinstance(mid, int)

    def test_get_memory_item(self, store: StorageProvider):
        item = MemoryItem(
            content="User talked about Bello features",
            item_type="essence",
            source_peer="user",
            subject_peer="user",
            attribution_type="self_statement",
        )
        mid = store.store_memory_item(item)

        fetched = store.get_memory_item(mid)
        assert fetched is not None
        assert fetched["content"] == "User talked about Bello features"
        assert fetched["item_type"] == "essence"
        assert fetched["source_peer"] == "user"
        assert fetched["subject_peer"] == "user"
        assert fetched["attribution_type"] == "self_statement"

    def test_get_memory_item_not_found(self, store: StorageProvider):
        assert store.get_memory_item(999) is None

    def test_search_by_peer(self, store: StorageProvider):
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

    def test_search_memory_items_excludes_other_peers_even_with_matching_source_peer(self, store: StorageProvider):
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

    def test_search_memory_items_finds_items_for_a_real_non_literal_peer_id(self, store: StorageProvider):
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

    def test_search_memory_items_records_scoring_summary_event(self, store: StorageProvider):
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
            peer_id="user", query_embedding=[1.0, 0.0, 0.0],
            budget=MemoryItemBudget(fact=0, surprise=0, feeling=0, free=1),
            tracer=tracer, request_id="req1",
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

    def test_search_memory_items_no_tracer_does_not_raise(self, store: StorageProvider):
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        store.store_memory_item(MemoryItem(
            content="About Bello", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=msg_id, embedding=[1.0, 0.0, 0.0],
        ))
        results = store.search_memory_items(peer_id="user", query_embedding=[1.0, 0.0, 0.0])
        assert len(results) == 1


class TestSearchMemoryItemsTieredBudget:
    def test_reserved_type_wins_over_higher_scored_free_candidate(self, store: StorageProvider):
        """A fact-typed candidate with a much lower score than an
        essence-typed one still gets its reserved slot — tiers aren't a
        single blind top-K across all types."""
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        fact_id = store.store_memory_item(MemoryItem(
            content="fact", item_type="fact", source_peer="user", subject_peer="user",
            attribution_type="self_statement", message_id=msg_id, embedding=[0.1, 1.0, 0.0],
        ))
        essence_id = store.store_memory_item(MemoryItem(
            content="essence", item_type="essence", source_peer="user", subject_peer="user",
            attribution_type="self_statement", message_id=msg_id, embedding=[1.0, 0.0, 0.0],
        ))

        results = store.search_memory_items(
            peer_id="user", query_embedding=[1.0, 0.0, 0.0],
            budget=MemoryItemBudget(fact=1, surprise=0, feeling=0, free=1),
        )

        assert {item.id for item in results} == {fact_id, essence_id}

    def test_exhausted_reservation_rolls_into_free_pool_not_dropped(self, store: StorageProvider):
        """No surprise-type items exist for this peer — that reserved slot
        must not just vanish; the total returned count stays the same,
        with an extra essence item filling the rolled-over free slot."""
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        first_id = store.store_memory_item(MemoryItem(
            content="essence one", item_type="essence", source_peer="user", subject_peer="user",
            attribution_type="self_statement", message_id=msg_id, embedding=[1.0, 0.0, 0.0],
        ))
        second_id = store.store_memory_item(MemoryItem(
            content="essence two", item_type="essence", source_peer="user", subject_peer="user",
            attribution_type="self_statement", message_id=msg_id, embedding=[0.9, 0.1, 0.0],
        ))

        results = store.search_memory_items(
            peer_id="user", query_embedding=[1.0, 0.0, 0.0],
            budget=MemoryItemBudget(fact=0, surprise=1, feeling=0, free=1),
        )

        assert {item.id for item in results} == {first_id, second_id}

    def test_rare_type_item_far_from_query_still_included_when_reserved(self, store: StorageProvider):
        """A surprise-typed item semantically distant from the query must
        still surface when the budget reserves a slot for it, even with
        many closer fact/essence candidates competing for attention."""
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        for i in range(10):
            store.store_memory_item(MemoryItem(
                content=f"near fact {i}", item_type="fact", source_peer="user", subject_peer="user",
                attribution_type="self_statement", message_id=msg_id, embedding=[1.0, 0.0, 0.0],
            ))
        surprise_id = store.store_memory_item(MemoryItem(
            content="far surprise", item_type="surprise", source_peer="user", subject_peer="user",
            attribution_type="self_statement", message_id=msg_id, embedding=[0.05, 1.0, 0.0],
        ))

        results = store.search_memory_items(
            peer_id="user", query_embedding=[1.0, 0.0, 0.0],
            budget=MemoryItemBudget(fact=0, surprise=1, feeling=0, free=0),
        )

        assert [item.id for item in results] == [surprise_id]


class TestMemoryItemsByTurn:
    def test_returns_only_items_from_the_given_turn(self, store: StorageProvider):
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

    def test_ordered_oldest_first(self, store: StorageProvider):
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

    def test_empty_when_turn_has_no_items(self, store: StorageProvider):
        assert store.get_memory_items_by_turn(999) == []

    def test_embedding_round_trips(self, store: StorageProvider):
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        store.store_memory_item(MemoryItem(
            content="About Bello", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=msg_id, embedding=[0.1, 0.2, 0.3],
        ))

        results = store.get_memory_items_by_turn(1)

        assert results[0].embedding == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)

    def test_no_embedding_stays_none(self, store: StorageProvider):
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        store.store_memory_item(MemoryItem(
            content="About Bello", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=msg_id,
        ))

        results = store.get_memory_items_by_turn(1)

        assert results[0].embedding is None


class TestMemoryItemsByTopic:
    def test_returns_only_items_from_the_given_topic(self, store: StorageProvider):
        t1 = store.create_topic("First", summary="", peer_id="user")
        t2 = store.create_topic("Second", summary="", peer_id="user")

        first = MemoryItem(
            content="fact about topic 1", item_type="fact",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            topic_id=t1,
        )
        second = MemoryItem(
            content="essence of topic 1", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            topic_id=t1,
        )
        other_topic = MemoryItem(
            content="fact about topic 2", item_type="fact",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            topic_id=t2,
        )
        store.store_memory_item(first)
        store.store_memory_item(second)
        store.store_memory_item(other_topic)

        results = store.get_memory_items_by_topic(t1)

        assert [item.content for item in results] == ["fact about topic 1", "essence of topic 1"]

    def test_ordered_oldest_first(self, store: StorageProvider):
        topic_id = store.create_topic("Topic", summary="", peer_id="user")
        for content in ("third", "first", "second"):
            store.store_memory_item(MemoryItem(
                content=content, item_type="essence",
                source_peer="user", subject_peer="user", attribution_type="self_statement",
                topic_id=topic_id,
            ))

        results = store.get_memory_items_by_topic(topic_id)

        assert [item.content for item in results] == ["third", "first", "second"]
        assert results[0].id < results[1].id < results[2].id

    def test_empty_when_topic_has_no_items(self, store: StorageProvider):
        topic_id = store.create_topic("Empty", summary="", peer_id="user")
        assert store.get_memory_items_by_topic(topic_id) == []


class TestListMemoryItems:
    def test_returns_every_item_across_topics(self, store: StorageProvider):
        t1 = store.create_topic("First", summary="", peer_id="user")
        t2 = store.create_topic("Second", summary="", peer_id="user")
        id1 = store.store_memory_item(MemoryItem(
            content="a", item_type="fact", source_peer="user", subject_peer="user",
            attribution_type="self_statement", topic_id=t1,
        ))
        id2 = store.store_memory_item(MemoryItem(
            content="b", item_type="essence", source_peer="user", subject_peer="user",
            attribution_type="self_statement", topic_id=t2,
        ))

        assert {item.id for item in store.list_memory_items()} == {id1, id2}

    def test_filters_by_item_type(self, store: StorageProvider):
        topic_id = store.create_topic("Topic", summary="", peer_id="user")
        fact_id = store.store_memory_item(MemoryItem(
            content="a fact", item_type="fact", source_peer="user", subject_peer="user",
            attribution_type="self_statement", topic_id=topic_id,
        ))
        store.store_memory_item(MemoryItem(
            content="an essence", item_type="essence", source_peer="user", subject_peer="user",
            attribution_type="self_statement", topic_id=topic_id,
        ))

        results = store.list_memory_items(item_type="fact")
        assert [item.id for item in results] == [fact_id]

    def test_filters_by_topic_id(self, store: StorageProvider):
        t1 = store.create_topic("First", summary="", peer_id="user")
        t2 = store.create_topic("Second", summary="", peer_id="user")
        id1 = store.store_memory_item(MemoryItem(
            content="a", item_type="fact", source_peer="user", subject_peer="user",
            attribution_type="self_statement", topic_id=t1,
        ))
        store.store_memory_item(MemoryItem(
            content="b", item_type="fact", source_peer="user", subject_peer="user",
            attribution_type="self_statement", topic_id=t2,
        ))

        results = store.list_memory_items(topic_id=t1)
        assert [item.id for item in results] == [id1]

    def test_empty_when_no_items(self, store: StorageProvider):
        assert store.list_memory_items() == []


class TestScoreBreakdownCharacterization:
    """Locks the exact per-entity retrieval_scoring_summary breakdown shape
    against full-dict equality — the guarantee the RetrievalScorer extraction
    must preserve. `now` is pinned to the past and temporal *weight* is 0,
    so the blend is fully deterministic: every row's created_at is later
    than `_NOW`, so its raw recency clamps to 1.0, but with temporal weight
    0 that contributes 0.0 to the final score."""

    _NOW = datetime(2026, 1, 1, tzinfo=UTC)
    _WEIGHTS = {"semantic": 0.5, "tag_overlap": 0.3, "person": 0.2, "temporal": 0.0}

    def test_topic_breakdown_shape(self, store: StorageProvider):
        tid = store.create_topic("Bello", summary="", peer_id="user", tags=["fitness"],
                                 embedding=[1.0, 0.0, 0.0])
        store.close_topic(tid, "")

        tracer = SpyTracer()
        store.search_topics(
            query_embedding=[1.0, 0.0, 0.0], tags=["fitness"], peer_id="user",
            weights=self._WEIGHTS, tracer=tracer, now=self._NOW,
        )

        candidate = tracer.events_of_type("retrieval_scoring_summary")[0]["payload"]["top_candidates"][0]
        assert candidate == {
            "candidate_type": "topic",
            "candidate_id": tid,
            "semantic_raw": pytest.approx(1.0),
            "semantic_weighted": pytest.approx(0.5),
            "tag_overlap_raw": pytest.approx(1.0),
            "tag_overlap_weighted": pytest.approx(0.3),
            "person_weighted": pytest.approx(0.2),
            "temporal_raw": pytest.approx(1.0),
            "temporal_weighted": pytest.approx(0.0),
            "final_score": pytest.approx(1.0),
            "included": True,
        }

    def test_memory_item_breakdown_shape_omits_tag_and_person(self, store: StorageProvider):
        msg_id = store.store_message(turn_id=1, role="user", content="hi", peer_id="user")
        item_id = store.store_memory_item(MemoryItem(
            content="About Bello", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            message_id=msg_id, embedding=[1.0, 0.0, 0.0],
        ))

        tracer = SpyTracer()
        store.search_memory_items(
            query_embedding=[1.0, 0.0, 0.0], peer_id="user",
            weights=self._WEIGHTS, tracer=tracer, now=self._NOW,
        )

        candidate = tracer.events_of_type("retrieval_scoring_summary")[0]["payload"]["top_candidates"][0]
        assert candidate == {
            "candidate_type": "memory_item",
            "candidate_id": item_id,
            "semantic_raw": pytest.approx(1.0),
            "semantic_weighted": pytest.approx(0.5),
            "temporal_raw": pytest.approx(1.0),
            "temporal_weighted": pytest.approx(0.0),
            "final_score": pytest.approx(0.5),
            "included": True,
        }

    def test_cluster_candidate_breakdown_shape_includes_candidate_kind(self, store: StorageProvider):
        cid = store.create_cluster("user", "Bike", "summary", tags=["fitness"], embedding=[1.0, 0.0, 0.0])
        new = store.create_topic("New topic", summary="", peer_id="user", tags=["fitness"],
                                 embedding=[1.0, 0.0, 0.0])
        store.close_topic(new, "summary")

        tracer = SpyTracer()
        store.search_cluster_candidates(new, "user", weights=self._WEIGHTS, tracer=tracer, now=self._NOW)

        candidate = tracer.events_of_type("retrieval_scoring_summary")[0]["payload"]["top_candidates"][0]
        assert candidate == {
            "candidate_type": "cluster_candidate",
            "candidate_kind": "cluster",
            "candidate_id": cid,
            "semantic_raw": pytest.approx(1.0),
            "semantic_weighted": pytest.approx(0.5),
            "tag_overlap_raw": pytest.approx(1.0),
            "tag_overlap_weighted": pytest.approx(0.3),
            "temporal_raw": pytest.approx(1.0),
            "temporal_weighted": pytest.approx(0.0),
            "final_score": pytest.approx(0.8),
            "included": True,
        }


class TestTemporalScoring:
    _NOW = datetime(2026, 1, 1, tzinfo=UTC)

    def test_reranks_equally_relevant_topics_by_recency(self, store_backend):
        """Two topics with identical tag relevance must be ordered newest-first
        once the temporal term breaks the tie."""
        store = store_backend.store
        old = store.create_topic("Old", tags=["fitness"])
        store.close_topic(old, "old")
        new = store.create_topic("New", tags=["fitness"])
        store.close_topic(new, "new")
        store_backend.backdate("topics", old, "closed_at", datetime(2020, 1, 1, tzinfo=UTC))
        store_backend.backdate("topics", new, "closed_at", datetime(2025, 12, 31, tzinfo=UTC))

        results = store.search_topics(tags=["fitness"], now=self._NOW)
        assert [r["id"] for r in results] == [new, old]

    def test_does_not_surface_irrelevant_recent_topic(self, store: StorageProvider):
        """A brand-new topic with no tag/semantic/peer overlap must stay out —
        temporal re-ranks relevant candidates, it never creates relevance."""
        t = store.create_topic("Unrelated", tags=["cooking"])
        store.close_topic(t, "unrelated")  # closed_at = now, i.e. very recent

        results = store.search_topics(tags=["fitness"], now=self._NOW)
        assert results == []

    def test_reranks_equally_relevant_memory_items_by_recency(self, store_backend):
        store = store_backend.store
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
        store_backend.backdate("memory_items", old_id, "created_at", datetime(2020, 1, 1, tzinfo=UTC))
        store_backend.backdate("memory_items", new_id, "created_at", datetime(2025, 12, 31, tzinfo=UTC))

        results = store.search_memory_items(peer_id="user", query_embedding=[1.0, 0.0, 0.0], now=self._NOW)
        assert [item.id for item in results] == [new_id, old_id]
