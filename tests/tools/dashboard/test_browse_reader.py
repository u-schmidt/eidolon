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
        t1 = storage.create_topic("Open", peer_id="u1")
        t2 = storage.create_topic("Closed", peer_id="u1")
        storage.close_topic(t2, "done")
        storage.facts.append(Fact(peer_id="u1", category="basic", key="name", value="Alex", source="user"))
        storage.store_memory_item(MemoryItem(
            content="a", item_type="fact", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=t1,
        ))
        storage.create_cluster("u1", "Bike", "summary")
        storage.store_message(turn_id=1, role="user", content="hi", peer_id="u1", request_id="req1")
        reader = BrowseReader(storage)

        counts = reader.overview_counts()

        assert counts.total_turns == 1
        assert counts.total_topics == 2
        assert counts.active_topics == 1
        assert counts.total_facts == 1
        assert counts.total_memory_items == 1
        assert counts.total_clusters == 1


class TestTopics:
    def test_lists_all_topics_by_default(self):
        storage = MockStorage()
        storage.create_topic("A", peer_id="u1")
        storage.create_topic("B", peer_id="u2")
        reader = BrowseReader(storage)

        page = reader.topics(page=1)

        assert page.total == 2
        assert {t.peer_id for t in page.items} == {"u1", "u2"}

    def test_filters_by_status_and_peer_id(self):
        storage = MockStorage()
        tid = storage.create_topic("A", peer_id="u1")
        storage.close_topic(tid, "done")
        storage.create_topic("B", peer_id="u1")
        storage.create_topic("C", peer_id="u2")
        reader = BrowseReader(storage)

        page = reader.topics(page=1, status="closed", peer_id="u1")

        assert [t.id for t in page.items] == [tid]


class TestFacts:
    def test_lists_all_facts_by_default(self):
        storage = MockStorage()
        storage.facts.append(Fact(peer_id="u1", category="basic", key="name", value="Alex", source="user"))
        storage.facts.append(Fact(peer_id="u2", category="basic", key="name", value="Sam", source="user"))
        reader = BrowseReader(storage)

        page = reader.facts(page=1)

        assert page.total == 2

    def test_filters_by_category_and_peer_id(self):
        storage = MockStorage()
        storage.facts.append(Fact(peer_id="u1", category="basic", key="name", value="Alex", source="user"))
        storage.facts.append(Fact(peer_id="u1", category="work", key="role", value="engineer", source="ai"))
        reader = BrowseReader(storage)

        page = reader.facts(page=1, category="work")

        assert [f.key for f in page.items] == ["role"]


class TestMemoryItems:
    def test_lists_all_items_by_default(self):
        storage = MockStorage()
        topic_id = storage.create_topic("Topic", peer_id="u1")
        storage.store_memory_item(MemoryItem(
            content="a", item_type="fact", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=topic_id,
        ))
        reader = BrowseReader(storage)

        page = reader.memory_items(page=1)

        assert page.total == 1

    def test_filters_by_item_type_and_topic_id(self):
        storage = MockStorage()
        t1 = storage.create_topic("First", peer_id="u1")
        t2 = storage.create_topic("Second", peer_id="u1")
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


class TestClusters:
    def test_lists_all_clusters_by_default(self):
        storage = MockStorage()
        storage.create_cluster("u1", "Bike", "long summary text")
        reader = BrowseReader(storage)

        page = reader.clusters(page=1)

        assert page.total == 1
        assert page.items[0].summary_preview == "long summary text"

    def test_filters_by_peer_id(self):
        storage = MockStorage()
        mine = storage.create_cluster("u1", "Bike", "summary")
        storage.create_cluster("u2", "Trip", "summary")
        reader = BrowseReader(storage)

        page = reader.clusters(page=1, peer_id="u1")

        assert [c.id for c in page.items] == [mine]


class TestPagination:
    def test_second_page_returns_remaining_items_and_flags(self):
        storage = MockStorage()
        for i in range(75):
            storage.facts.append(Fact(peer_id="u1", category="basic", key=f"k{i}", value="v", source="user"))
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
        storage.facts.append(Fact(peer_id="u1", category="basic", key="k", value="v", source="user"))
        reader = BrowseReader(storage)

        page = reader.facts(page=0)

        assert page.page == 1
