"""Tests for ClusterConsolidator.

Uses the real SqliteStore (not a hand-rolled mock), same rationale as
test_fact_consolidator.py — verifies the consolidator actually matches
the storage layer's real search_cluster_candidates/assign_topic_to_cluster
contracts, not a stand-in's assumptions about them.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.consolidation.cluster_consolidator import ClusterConsolidator
from core.stores.sqlite_store import SqliteStore
from tests.mocks import MockEmbedder, MockLLM, SpyMetricsRecorder, SpyTracer

_MATCH_EMBEDDING = [1.0, 0.0, 0.0]


@pytest.fixture
def store():
    tmp = tempfile.mkdtemp()
    s = SqliteStore(str(Path(tmp) / "test.db"), use_vec_index=False)
    s.connect()
    yield s
    s.close()


def _closed_topic(store: SqliteStore, label: str, embedding: list[float] | None = _MATCH_EMBEDDING) -> int:
    tid = store.create_topic(label, summary="", workspace_id="user", embedding=embedding)
    store.close_topic(tid, f"{label} summary")
    return tid


class TestConsolidateTopicNoOp:
    def test_no_llm_call_when_shortlist_is_empty(self, store: SqliteStore):
        topic_id = _closed_topic(store, "Solo topic")  # nothing else to match against

        class FailIfCalledLLM:
            def complete(self, *a, **kw):
                raise AssertionError("LLM should not be called when the shortlist is empty")

        ClusterConsolidator(store, FailIfCalledLLM(), MockEmbedder()).consolidate_topic(topic_id, "user")

    def test_no_op_when_topic_not_found(self, store: SqliteStore):
        class FailIfCalledLLM:
            def complete(self, *a, **kw):
                raise AssertionError("LLM should not be called for a nonexistent topic")

        ClusterConsolidator(store, FailIfCalledLLM(), MockEmbedder()).consolidate_topic(999, "user")


class TestConsolidateTopicMatchesCluster:
    def test_updates_cluster_summary_in_place_and_assigns_topic(self, store: SqliteStore):
        cid = store.create_cluster("user", "Bike", "old summary", embedding=_MATCH_EMBEDDING)
        new_topic = _closed_topic(store, "New bike development")

        llm = MockLLM('{"match_index": 1, "label": "Bike saga", "summary": "consolidated bike story"}')
        ClusterConsolidator(store, llm, MockEmbedder()).consolidate_topic(new_topic, "user")

        cluster = store.get_cluster(cid)
        assert cluster["summary"] == "consolidated bike story"
        assert cluster["label"] == "Bike saga"
        assert cluster["topic_count"] == 1
        assert [t["id"] for t in store.get_topics_by_cluster(cid)] == [new_topic]

    def test_merges_tags_from_matched_cluster_and_new_topic(self, store: SqliteStore):
        cid = store.create_cluster("user", "Bike", "summary", tags=["bike"], embedding=_MATCH_EMBEDDING)
        new_topic = store.create_topic(
            "New development", summary="", workspace_id="user", tags=["shopping"], embedding=_MATCH_EMBEDDING
        )
        store.close_topic(new_topic, "summary")

        llm = MockLLM('{"match_index": 1, "label": "Bike", "summary": "updated"}')
        ClusterConsolidator(store, llm, MockEmbedder()).consolidate_topic(new_topic, "user")

        assert store.get_cluster(cid)["tags"] == ["bike", "shopping"]


class TestConsolidateTopicMatchesRawTopic:
    def test_promotes_both_topics_into_a_new_cluster(self, store: SqliteStore):
        older = _closed_topic(store, "Considering a bike")
        new_topic = _closed_topic(store, "Bought the bike")

        llm = MockLLM('{"match_index": 1, "label": "Bike saga", "summary": "from considering to buying"}')
        ClusterConsolidator(store, llm, MockEmbedder()).consolidate_topic(new_topic, "user")

        # Exactly one cluster now exists, holding both topics.
        clusters = store.search_clusters(query_embedding=_MATCH_EMBEDDING, workspace_id="user")
        assert len(clusters) == 1
        cluster = clusters[0]
        assert cluster["label"] == "Bike saga"
        assert cluster["summary"] == "from considering to buying"
        assert cluster["topic_count"] == 2
        member_ids = {t["id"] for t in store.get_topics_by_cluster(cluster["id"])}
        assert member_ids == {older, new_topic}


class TestConsolidateTopicNoMatch:
    def test_stays_unclustered_when_llm_declares_no_match(self, store: SqliteStore):
        other = _closed_topic(store, "Unrelated topic")
        new_topic = _closed_topic(store, "New topic")

        llm = MockLLM('{"match_index": null, "label": "", "summary": ""}')
        ClusterConsolidator(store, llm, MockEmbedder()).consolidate_topic(new_topic, "user")

        assert store.search_clusters(query_embedding=_MATCH_EMBEDDING, workspace_id="user") == []
        # Both topics remain individually searchable, not swallowed into a
        # cluster (search_topics excludes anything with cluster_id set) —
        # queried with a real embedding now, since person_match no longer
        # rescues a zero-signal query (see its call site's comment).
        assert {t["id"] for t in store.search_topics(query_embedding=_MATCH_EMBEDDING, workspace_id="user")} == {
            other, new_topic,
        }


class TestConsolidateTopicFailure:
    def test_malformed_response_leaves_topic_unclustered(self, store: SqliteStore):
        _closed_topic(store, "Other topic")
        new_topic = _closed_topic(store, "New topic")

        ClusterConsolidator(store, MockLLM("not valid json"), MockEmbedder()).consolidate_topic(new_topic, "user")

        assert store.search_clusters(query_embedding=_MATCH_EMBEDDING, workspace_id="user") == []

    def test_malformed_response_records_error_event(self, store: SqliteStore, caplog):
        _closed_topic(store, "Other topic")
        new_topic = _closed_topic(store, "New topic")
        tracer = SpyTracer()
        metrics = SpyMetricsRecorder()

        with caplog.at_level("WARNING"):
            ClusterConsolidator(
                store, MockLLM("not valid json"), MockEmbedder(), tracer=tracer, metrics=metrics
            ).consolidate_topic(new_topic, "user")

        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "cluster_consolidation" for e in error_events)
        assert tracer.events_of_type("cluster_consolidation") == []
        assert "cluster_consolidation failed" in caplog.text
        assert any(e["step"] == "cluster_consolidation" for e in metrics.events)
        assert metrics.llm_calls == []

    def test_embedding_failure_records_error_event(self, store: SqliteStore, caplog):
        cid = store.create_cluster("user", "Bike", "old summary", embedding=_MATCH_EMBEDDING)
        new_topic = _closed_topic(store, "New bike development")
        llm = MockLLM('{"match_index": 1, "label": "Bike saga", "summary": "consolidated bike story"}')
        tracer = SpyTracer()
        metrics = SpyMetricsRecorder()

        with caplog.at_level("WARNING"):
            ClusterConsolidator(
                store, llm, MockEmbedder(should_fail=True), tracer=tracer, metrics=metrics
            ).consolidate_topic(new_topic, "user")

        # The cluster merge itself still succeeds — only the re-embedding failed.
        assert store.get_cluster(cid)["summary"] == "consolidated bike story"
        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "cluster_summary_embedding" for e in error_events)
        assert "cluster_summary_embedding failed" in caplog.text
        assert any(e["step"] == "cluster_summary_embedding" for e in metrics.events)
        assert metrics.embedding_calls == []


class TestConsolidateTopicTracing:
    def test_records_cluster_consolidation_event_on_match(self, store: SqliteStore):
        cid = store.create_cluster("user", "Bike", "old summary", embedding=_MATCH_EMBEDDING)
        new_topic = _closed_topic(store, "New development")
        llm = MockLLM('{"match_index": 1, "label": "Bike saga", "summary": "consolidated"}')
        tracer = SpyTracer()

        ClusterConsolidator(store, llm, MockEmbedder(), tracer=tracer).consolidate_topic(
            new_topic, "user", request_id="req1", session_id="sess1"
        )

        events = tracer.events_of_type("cluster_consolidation")
        assert len(events) == 1
        event = events[0]
        assert event["request_id"] == "req1"
        assert event["session_id"] == "sess1"
        assert event["payload"]["topic_id"] == new_topic
        assert event["payload"]["workspace_id"] == "user"
        assert event["payload"]["match_index"] == 1
        assert event["payload"]["resulting_cluster_id"] == cid
        assert event["payload"]["candidates"][0]["kind"] == "cluster"

    def test_records_llm_call_and_embedding_call_metrics_on_match(self, store: SqliteStore):
        store.create_cluster("user", "Bike", "old summary", embedding=_MATCH_EMBEDDING)
        new_topic = _closed_topic(store, "New development")
        llm = MockLLM('{"match_index": 1, "label": "Bike saga", "summary": "consolidated"}')
        metrics = SpyMetricsRecorder()

        ClusterConsolidator(store, llm, MockEmbedder(), metrics=metrics).consolidate_topic(
            new_topic, "user", request_id="req1"
        )

        assert len(metrics.llm_calls) == 1
        assert metrics.llm_calls[0]["call_type"] == "cluster_consolidation"
        assert metrics.llm_calls[0]["succeeded"] is True
        assert metrics.llm_calls[0]["workspace_id"] == "user"
        assert metrics.llm_calls[0]["request_id"] == "req1"

        assert len(metrics.embedding_calls) == 1
        assert metrics.embedding_calls[0]["purpose"] == "cluster_summary"
        assert metrics.embedding_calls[0]["succeeded"] is True


class TestConsolidateTopicShortlistSize:
    def test_shortlist_size_bounds_candidates_shown_to_the_llm(self, store: SqliteStore):
        for i in range(5):
            _closed_topic(store, f"Candidate topic {i}")
        new_topic = _closed_topic(store, "New topic")

        llm = MockLLM('{"match_index": null, "label": "", "summary": ""}')
        ClusterConsolidator(store, llm, MockEmbedder(), shortlist_size=2).consolidate_topic(new_topic, "user")

        # Only 2 candidates should have made it into the prompt.
        assert llm.last_prompt.count("past topic, not yet grouped") + llm.last_prompt.count(
            "existing recurring thread"
        ) == 2
