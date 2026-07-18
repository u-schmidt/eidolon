"""Tests for the SQLite storage backend."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.models import Fact, MemoryItem
from core.stores.sqlite_store import SqliteStore


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
