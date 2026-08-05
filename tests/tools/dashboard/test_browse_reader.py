"""Unit tests for BrowseReader — the pagination layer behind the
dashboard's browse pages (topics/facts/memory-items/clusters). Uses
MockStorage; real created_at/closed_at ordering is covered by
tests/core/test_storage_provider_contract.py's TestListTopics/
TestListClusters/TestListMemoryItems — these tests only exercise
filtering and pagination arithmetic."""

from __future__ import annotations

from core.models import Fact, MemoryItem
from tests.mocks import MockStorage
from tools.dashboard.browse_reader import BrowseReader


class TestOverviewCounts:
    def test_aggregates_counts_across_all_entities(self):
        storage = MockStorage()
        t1 = storage.create_topic("Open", workspace_id="u1")
        t2 = storage.create_topic("Closed", workspace_id="u1")
        storage.close_topic(t2, "done")
        storage.facts.append(Fact(workspace_id="u1", category="basic", key="name", value="Alex", source="user"))
        storage.store_memory_item(MemoryItem(
            content="a", item_type="fact", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=t1,
        ))
        storage.create_cluster("u1", "Bike", "summary")
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        storage.increment_entity_mentions("u1", ["lisa"])
        reader = BrowseReader(storage)

        counts = reader.overview_counts()

        assert counts.total_turns == 1
        assert counts.total_topics == 2
        assert counts.active_topics == 1
        assert counts.total_facts == 1
        assert counts.total_memory_items == 1
        assert counts.total_clusters == 1
        assert counts.total_entity_profiles == 1


class TestTopics:
    def test_lists_all_topics_by_default(self):
        storage = MockStorage()
        storage.create_topic("A", workspace_id="u1")
        storage.create_topic("B", workspace_id="u2")
        reader = BrowseReader(storage)

        page = reader.topics(page=1)

        assert page.total == 2
        assert {t.workspace_id for t in page.items} == {"u1", "u2"}

    def test_filters_by_status_and_workspace_id(self):
        storage = MockStorage()
        tid = storage.create_topic("A", workspace_id="u1")
        storage.close_topic(tid, "done")
        storage.create_topic("B", workspace_id="u1")
        storage.create_topic("C", workspace_id="u2")
        reader = BrowseReader(storage)

        page = reader.topics(page=1, status="closed", workspace_id="u1")

        assert [t.id for t in page.items] == [tid]


class TestFacts:
    def test_lists_all_facts_by_default(self):
        storage = MockStorage()
        storage.facts.append(Fact(workspace_id="u1", category="basic", key="name", value="Alex", source="user"))
        storage.facts.append(Fact(workspace_id="u2", category="basic", key="name", value="Sam", source="user"))
        reader = BrowseReader(storage)

        page = reader.facts(page=1)

        assert page.total == 2

    def test_filters_by_category_and_workspace_id(self):
        storage = MockStorage()
        storage.facts.append(Fact(workspace_id="u1", category="basic", key="name", value="Alex", source="user"))
        storage.facts.append(Fact(workspace_id="u1", category="work", key="role", value="engineer", source="ai"))
        reader = BrowseReader(storage)

        page = reader.facts(page=1, category="work")

        assert [f.key for f in page.items] == ["role"]

    def test_default_sort_preserves_get_all_facts_order(self):
        storage = MockStorage()
        storage.facts.append(Fact(workspace_id="u1", category="z", key="last", value="v", source="user"))
        storage.facts.append(Fact(workspace_id="u1", category="a", key="first", value="v", source="user"))
        reader = BrowseReader(storage)

        page = reader.facts(page=1)

        assert [f.key for f in page.items] == ["last", "first"]

    def test_sort_by_created_is_newest_first(self):
        storage = MockStorage()
        storage.facts.append(Fact(
            workspace_id="u1", category="a", key="old", value="v", source="user",
            created_at="2026-01-01T00:00:00",
        ))
        storage.facts.append(Fact(
            workspace_id="u1", category="a", key="new", value="v", source="user",
            created_at="2026-06-01T00:00:00",
        ))
        storage.facts.append(Fact(workspace_id="u1", category="a", key="undated", value="v", source="user"))
        reader = BrowseReader(storage)

        page = reader.facts(page=1, sort_by="created")

        assert [f.key for f in page.items] == ["new", "old", "undated"]

    def test_sort_by_importance_is_high_first(self):
        storage = MockStorage()
        storage.facts.append(Fact(workspace_id="u1", category="a", key="low", value="v", source="user", importance="low"))
        storage.facts.append(Fact(workspace_id="u1", category="a", key="high", value="v", source="user", importance="high"))
        storage.facts.append(Fact(workspace_id="u1", category="a", key="medium", value="v", source="user", importance="medium"))
        reader = BrowseReader(storage)

        page = reader.facts(page=1, sort_by="importance")

        assert [f.key for f in page.items] == ["high", "medium", "low"]

    def test_sort_by_key_is_alphabetical(self):
        storage = MockStorage()
        storage.facts.append(Fact(workspace_id="u1", category="z", key="banana", value="v", source="user"))
        storage.facts.append(Fact(workspace_id="u1", category="a", key="apple", value="v", source="user"))
        reader = BrowseReader(storage)

        page = reader.facts(page=1, sort_by="key")

        assert [f.key for f in page.items] == ["apple", "banana"]


class TestMemoryItems:
    def test_lists_all_items_by_default(self):
        storage = MockStorage()
        topic_id = storage.create_topic("Topic", workspace_id="u1")
        storage.store_memory_item(MemoryItem(
            content="a", item_type="fact", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=topic_id,
        ))
        reader = BrowseReader(storage)

        page = reader.memory_items(page=1)

        assert page.total == 1

    def test_filters_by_item_type_and_topic_id(self):
        storage = MockStorage()
        t1 = storage.create_topic("First", workspace_id="u1")
        t2 = storage.create_topic("Second", workspace_id="u1")
        storage.store_memory_item(MemoryItem(
            content="a fact", item_type="fact", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=t1,
        ))
        storage.store_memory_item(MemoryItem(
            content="an essence", item_type="essence", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=t1,
        ))
        storage.store_memory_item(MemoryItem(
            content="other topic", item_type="fact", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=t2,
        ))
        reader = BrowseReader(storage)

        page = reader.memory_items(page=1, item_type="fact", topic_id=t1)

        assert [i.content for i in page.items] == ["a fact"]

    def test_passes_q_through_to_storage(self):
        storage = MockStorage()
        topic_id = storage.create_topic("Topic", workspace_id="u1")
        storage.store_memory_item(MemoryItem(
            content="loves hiking", item_type="fact", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=topic_id,
        ))
        storage.store_memory_item(MemoryItem(
            content="works as a teacher", item_type="fact", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=topic_id,
        ))
        reader = BrowseReader(storage)

        page = reader.memory_items(page=1, q="hiking")

        assert [i.content for i in page.items] == ["loves hiking"]


class TestClusters:
    def test_lists_all_clusters_by_default(self):
        storage = MockStorage()
        storage.create_cluster("u1", "Bike", "long summary text")
        reader = BrowseReader(storage)

        page = reader.clusters(page=1)

        assert page.total == 1
        assert page.items[0].summary_preview == "long summary text"

    def test_filters_by_workspace_id(self):
        storage = MockStorage()
        mine = storage.create_cluster("u1", "Bike", "summary")
        storage.create_cluster("u2", "Trip", "summary")
        reader = BrowseReader(storage)

        page = reader.clusters(page=1, workspace_id="u1")

        assert [c.id for c in page.items] == [mine]


class TestClusterDetail:
    def test_returns_none_for_unknown_cluster(self):
        storage = MockStorage()
        reader = BrowseReader(storage)

        assert reader.cluster_detail(999) is None

    def test_returns_cluster_fields_and_member_topics_newest_first(self):
        storage = MockStorage()
        cluster_id = storage.create_cluster(
            "u1", "Bike", "long summary", tags=["fitness"], milestones=["started riding"],
        )
        t1 = storage.create_topic("First ride", workspace_id="u1")
        t2 = storage.create_topic("Second ride", workspace_id="u1")
        storage.assign_topic_to_cluster(t1, cluster_id)
        storage.assign_topic_to_cluster(t2, cluster_id)
        reader = BrowseReader(storage)

        detail = reader.cluster_detail(cluster_id)

        assert detail is not None
        assert detail.id == cluster_id
        assert detail.workspace_id == "u1"
        assert detail.label == "Bike"
        assert detail.summary == "long summary"
        assert detail.tags == ["fitness"]
        assert detail.milestones == ["started riding"]
        assert detail.topic_count == 2
        assert [t.label for t in detail.topics] == ["Second ride", "First ride"]


class TestEntityProfiles:
    def test_lists_all_entity_profiles_including_unmaterialized(self):
        storage = MockStorage()
        storage.increment_entity_mentions("u1", ["lisa"])
        reader = BrowseReader(storage)

        page = reader.entity_profiles(page=1)

        assert page.total == 1
        assert page.items[0].entity_name == "lisa"
        assert page.items[0].materialized_at is None
        assert page.items[0].fact_count == 0

    def test_filters_by_workspace_id(self):
        storage = MockStorage()
        storage.increment_entity_mentions("u1", ["lisa"])
        storage.increment_entity_mentions("u2", ["bob"])
        reader = BrowseReader(storage)

        page = reader.entity_profiles(page=1, workspace_id="u1")

        assert [p.entity_name for p in page.items] == ["lisa"]

    def test_includes_fact_count_once_materialized(self):
        storage = MockStorage()
        storage.increment_entity_mentions("u1", ["lisa"])
        storage.materialize_entities("u1", threshold=1)
        storage.set_entity_fact("u1", "lisa", "relationship", "role", "coworker", "ai")
        reader = BrowseReader(storage)

        row = reader.entity_profiles(page=1).items[0]
        assert row.materialized_at is not None
        assert row.fact_count == 1


class TestEntityProfileDetail:
    def test_returns_none_for_unknown_entity(self):
        storage = MockStorage()
        reader = BrowseReader(storage)

        assert reader.entity_profile_detail("u1", "lisa") is None

    def test_returns_facts_and_source_topics(self):
        storage = MockStorage()
        tid = storage.create_topic("Coffee chat", workspace_id="u1")
        storage.close_topic(tid, "met lisa for coffee")
        storage.update_topic_entities(tid, ["lisa"])
        storage.increment_entity_mentions("u1", ["lisa"])
        storage.materialize_entities("u1", threshold=1)
        storage.set_entity_fact("u1", "lisa", "relationship", "role", "coworker", "ai")
        reader = BrowseReader(storage)

        detail = reader.entity_profile_detail("u1", "lisa")

        assert detail is not None
        assert detail.workspace_id == "u1"
        assert detail.entity_name == "lisa"
        assert detail.mention_count == 1
        assert detail.materialized_at is not None
        assert len(detail.facts) == 1
        assert detail.facts[0].value == "coworker"
        assert [t.id for t in detail.topics] == [tid]
        assert detail.topics[0].label == "Coffee chat"

    def test_returns_row_with_no_facts_when_not_yet_materialized(self):
        storage = MockStorage()
        storage.increment_entity_mentions("u1", ["lisa"])
        reader = BrowseReader(storage)

        detail = reader.entity_profile_detail("u1", "lisa")

        assert detail is not None
        assert detail.materialized_at is None
        assert detail.facts == []


class TestPagination:
    def test_second_page_returns_remaining_items_and_flags(self):
        storage = MockStorage()
        for i in range(75):
            storage.facts.append(Fact(workspace_id="u1", category="basic", key=f"k{i}", value="v", source="user"))
        reader = BrowseReader(storage)

        page1 = reader.facts(page=1)
        page2 = reader.facts(page=2)

        assert len(page1.items) == 50
        assert len(page2.items) == 25
        assert page1.has_next is True
        assert page1.has_prev is False
        assert page2.has_next is False
        assert page2.has_prev is True
        assert page1.total_pages == 2

    def test_empty_page_has_no_next_or_prev(self):
        storage = MockStorage()
        reader = BrowseReader(storage)

        page = reader.facts(page=1)

        assert page.total == 0
        assert page.total_pages == 1
        assert page.has_next is False
        assert page.has_prev is False

    def test_page_below_one_clamped_to_one(self):
        storage = MockStorage()
        storage.facts.append(Fact(workspace_id="u1", category="basic", key="k", value="v", source="user"))
        reader = BrowseReader(storage)

        page = reader.facts(page=0)

        assert page.page == 1
