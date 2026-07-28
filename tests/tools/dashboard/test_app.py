"""FastAPI TestClient smoke tests for the turn-detail dashboard router —
confirms both routes render for a rich-mode and a skeleton-mode turn.

Wraps ``create_dashboard_router`` in a bare test app (mounting the real
static dir under the same name production uses, so ``url_for()``
resolves in templates) rather than going through ``server/app.py`` —
keeps these tests focused on rendering, independent of auth/RPC
concerns. Auth and the merged-app wiring are covered by
tests/server/test_app.py instead.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from core.models import Fact, MemoryItem
from tests.mocks import MockStorage
from tools.dashboard.app import DASHBOARD_STATIC_DIR, create_dashboard_router
from tools.dashboard.browse_reader import BrowseReader
from tools.dashboard.cluster_dynamics_reader import ClusterDynamicsReader
from tools.dashboard.performance_reader import PerformanceReader
from tools.dashboard.prompt_quality_reader import PromptQualityReader
from tools.dashboard.retrieval_quality_reader import RetrievalQualityReader
from tools.dashboard.topic_lifecycle_reader import TopicLifecycleReader
from tools.dashboard.turn_reader import TurnReader


class _FakeMetricsReader:
    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def events_for_request(self, request_id: str) -> list[dict]:
        return []

    def llm_calls_for_request(self, request_id: str) -> list[dict]:
        return []

    def embedding_calls_for_request(self, request_id: str) -> list[dict]:
        return []

    def events_since(self, since):
        return []

    def llm_calls_since(self, since):
        return []

    def embedding_calls_since(self, since):
        return []


class _FakeTraceReader:
    def __init__(self, events: dict[str, list[dict]] | None = None) -> None:
        self._events = events or {}

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def events_for_request(self, request_id: str) -> list[dict]:
        return self._events.get(request_id, [])

    def events_of_type_since(self, event_type: str, since) -> list[dict]:
        return []


def _wrap(
    turn_reader: TurnReader,
    storage: MockStorage,
    metrics_reader: object | None = None,
    trace_reader: object | None = None,
    fact_categories: list[str] | None = None,
) -> FastAPI:
    app = FastAPI()
    performance_reader = PerformanceReader(metrics_reader or _FakeMetricsReader())
    retrieval_quality_reader = RetrievalQualityReader(trace_reader or _FakeTraceReader(), storage)
    prompt_quality_reader = PromptQualityReader(trace_reader or _FakeTraceReader(), storage, fact_categories or [])
    topic_lifecycle_reader = TopicLifecycleReader(trace_reader or _FakeTraceReader(), storage)
    cluster_dynamics_reader = ClusterDynamicsReader(trace_reader or _FakeTraceReader(), storage)
    browse_reader = BrowseReader(storage)
    app.include_router(
        create_dashboard_router(
            turn_reader, performance_reader, retrieval_quality_reader,
            prompt_quality_reader, topic_lifecycle_reader, cluster_dynamics_reader,
            browse_reader,
        )
    )
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_STATIC_DIR)), name="dashboard-static")
    return app


@pytest.fixture
def storage() -> MockStorage:
    return MockStorage()


@pytest.fixture
def client(storage: MockStorage) -> TestClient:
    turn_reader = TurnReader(storage, _FakeMetricsReader(), _FakeTraceReader())
    return TestClient(_wrap(turn_reader, storage))


class TestLanding:
    def test_shows_recent_activity(self, client: TestClient, storage: MockStorage):
        storage.store_message(turn_id=1, role="user", content="hello there", peer_id="u1", request_id="req1")

        response = client.get("/")
        assert response.status_code == 200
        assert "hello there" in response.text
        assert "req1" in response.text

    def test_empty_state_when_no_turns(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "No turns found" in response.text

    def test_nav_links_present(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert '/turns"' in response.text

    def test_no_debug_data_shows_muted_placeholder(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "enable debug tracing" in response.text

    def test_fallback_errors_shown_when_debug_data_present(self, storage: MockStorage):
        trace = _TraceReaderWithData(events_of_type={
            "topic_detection": [
                {
                    "payload": {"response": "sure, let's keep going", "result_type": "CONTINUE", "message": "what now"},
                    "request_id": "req-b", "created_at": "t2",
                },
            ],
        })
        turn_reader = TurnReader(storage, _FakeMetricsReader(), trace)
        client = TestClient(_wrap(turn_reader, storage, trace_reader=trace))

        response = client.get("/")
        assert response.status_code == 200
        assert "enable debug tracing" not in response.text
        assert 'Fallback errors (24h)' in response.text


class TestTurnsList:
    def test_lists_recent_turns(self, client: TestClient, storage: MockStorage):
        storage.store_message(turn_id=1, role="user", content="hello there", peer_id="u1", request_id="req1")

        response = client.get("/turns")
        assert response.status_code == 200
        assert "hello there" in response.text
        assert "req1" in response.text

    def test_empty_state_when_no_turns(self, client: TestClient):
        response = client.get("/turns")
        assert response.status_code == 200
        assert "No turns found" in response.text

    def test_filters_by_peer_id(self, client: TestClient, storage: MockStorage):
        storage.store_message(turn_id=1, role="user", content="mine", peer_id="u1", request_id="req1")
        storage.store_message(turn_id=2, role="user", content="theirs", peer_id="u2", request_id="req2")

        response = client.get("/turns", params={"peer_id": "u1"})
        assert response.status_code == 200
        assert "mine" in response.text
        assert "theirs" not in response.text


class TestTurnDetail:
    def test_skeleton_mode_renders(self, client: TestClient, storage: MockStorage):
        storage.store_message(turn_id=1, role="user", content="hi", peer_id="u1", request_id="req1")
        storage.store_message(turn_id=1, role="ai", content="hello", peer_id="ai", request_id="req1")

        response = client.get("/turns/req1")
        assert response.status_code == 200
        assert "No rich trace data" in response.text
        assert "hi" in response.text

    def test_rich_mode_renders(self, storage: MockStorage):
        storage.store_message(turn_id=1, role="user", content="hi", peer_id="u1", request_id="req1")
        trace = _FakeTraceReader({
            "req1": [
                {"event_type": "topic_detection", "payload": {"message": "hi"}, "duration_ms": 5.0,
                 "session_id": None, "created_at": "t1"},
            ],
        })
        turn_reader = TurnReader(storage, _FakeMetricsReader(), trace)
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get("/turns/req1")
        assert response.status_code == 200
        assert "topic_detection" in response.text

    def test_unknown_request_id_returns_404(self, client: TestClient):
        response = client.get("/turns/nope")
        assert response.status_code == 404

    def test_retrieval_scoring_summary_shows_candidate_label(self, storage: MockStorage):
        storage.store_message(turn_id=1, role="user", content="hi", peer_id="u1", request_id="req1")
        topic_id = storage.create_topic("Bello Features", summary="", tags=[])
        trace = _FakeTraceReader({
            "req1": [
                {
                    "event_type": "retrieval_scoring_summary",
                    "payload": {
                        "candidate_type": "topic",
                        "total_scored": 1,
                        "included_count": 1,
                        "top_candidates": [
                            {"candidate_type": "topic", "candidate_id": topic_id, "included": True,
                             "final_score": 0.5, "semantic_weighted": 0.4, "semantic_raw": 0.9,
                             "temporal_weighted": 0.1, "temporal_raw": 0.9},
                        ],
                    },
                    "duration_ms": 5.0, "session_id": None, "created_at": "t1",
                },
            ],
        })
        turn_reader = TurnReader(storage, _FakeMetricsReader(), trace)
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get("/turns/req1")
        assert response.status_code == 200
        assert "Bello Features" in response.text

    def test_timing_headline_and_waterfall_render(self, storage: MockStorage):
        storage.store_message(turn_id=1, role="user", content="hi", peer_id="u1", request_id="req1")
        trace = _FakeTraceReader({
            "req1": [
                {"event_type": "prefetch_complete", "payload": {}, "duration_ms": 120.0,
                 "session_id": None, "created_at": "2026-01-01T00:00:01.000000"},
                {"event_type": "turn_complete", "payload": {}, "duration_ms": 80.0,
                 "session_id": None, "created_at": "2026-01-01T00:00:03.000000"},
            ],
        })
        turn_reader = TurnReader(storage, _FakeMetricsReader(), trace)
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get("/turns/req1")
        assert response.status_code == 200
        assert "120ms" in response.text
        assert "80ms" in response.text
        assert "200ms" in response.text  # total
        assert "<svg" in response.text
        assert "Not available for this turn" not in response.text

    def test_timing_degrades_gracefully_without_wrapping_events(self, storage: MockStorage):
        storage.store_message(turn_id=1, role="user", content="hi", peer_id="u1", request_id="req1")
        trace = _FakeTraceReader({
            "req1": [
                {"event_type": "topic_detection", "payload": {}, "duration_ms": 5.0,
                 "session_id": None, "created_at": "t1"},
            ],
        })
        turn_reader = TurnReader(storage, _FakeMetricsReader(), trace)
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get("/turns/req1")
        assert response.status_code == 200
        assert "Not available for this turn" in response.text

    def test_no_timing_section_for_skeleton_only_turn(self, client: TestClient, storage: MockStorage):
        storage.store_message(turn_id=1, role="user", content="hi", peer_id="u1", request_id="req1")

        response = client.get("/turns/req1")
        assert response.status_code == 200
        assert "<h2>Timing</h2>" not in response.text


class _MetricsReaderWithData(_FakeMetricsReader):
    def __init__(self, llm_calls: list[dict] | None = None, events: list[dict] | None = None) -> None:
        self._llm_calls = llm_calls or []
        self._events = events or []

    def llm_calls_since(self, since):
        return self._llm_calls

    def events_since(self, since):
        return self._events


class TestPerformance:
    def test_renders_with_no_data(self, client: TestClient):
        response = client.get("/performance")
        assert response.status_code == 200
        assert "No LLM calls in this window" in response.text

    def test_renders_call_type_stats(self, storage: MockStorage):
        metrics = _MetricsReaderWithData(llm_calls=[
            {"call_type": "topic_detection", "duration_ms": 12.5, "succeeded": True,
             "peer_id": "u1", "session_id": None, "created_at": "2026-07-26T18:00:00+00:00"},
        ])
        turn_reader = TurnReader(storage, metrics, _FakeTraceReader())
        client = TestClient(_wrap(turn_reader, storage, metrics_reader=metrics))

        response = client.get("/performance")
        assert response.status_code == 200
        assert "topic_detection" in response.text

    def test_renders_top_error_sources(self, storage: MockStorage):
        metrics = _MetricsReaderWithData(events=[
            {"severity": "error", "step": "extraction", "message": "boom",
             "peer_id": "u1", "session_id": "s1", "created_at": "2026-07-26T18:00:00+00:00"},
        ])
        turn_reader = TurnReader(storage, metrics, _FakeTraceReader())
        client = TestClient(_wrap(turn_reader, storage, metrics_reader=metrics))

        response = client.get("/performance")
        assert response.status_code == 200
        assert "u1" in response.text
        assert "s1" in response.text

    def test_window_select_reflects_query_param(self, client: TestClient):
        response = client.get("/performance", params={"window": "1h"})
        assert response.status_code == 200
        assert 'value="1h" selected' in response.text

    def test_unknown_window_returns_400(self, client: TestClient):
        response = client.get("/performance", params={"window": "3y"})
        assert response.status_code == 400


class _TraceReaderWithData(_FakeTraceReader):
    def __init__(self, events_of_type: dict[str, list[dict]] | None = None) -> None:
        super().__init__()
        self._events_of_type = events_of_type or {}

    def events_of_type_since(self, event_type: str, since) -> list[dict]:
        return self._events_of_type.get(event_type, [])


class TestRetrievalQuality:
    def test_renders_with_no_data(self, client: TestClient):
        response = client.get("/retrieval-quality")
        assert response.status_code == 200
        assert "No <code>retrieval_scoring_summary</code> events" in response.text

    def test_renders_top_retrieved_with_label(self, storage: MockStorage):
        topic_id = storage.create_topic("Bello Features", summary="", tags=[])
        trace = _TraceReaderWithData(events_of_type={
            "retrieval_scoring_summary": [
                {"payload": {
                    "candidate_type": "topic",
                    "top_candidates": [
                        {"candidate_type": "topic", "candidate_id": topic_id, "included": True,
                         "final_score": 0.5, "semantic_weighted": 0.4, "semantic_raw": 0.9,
                         "temporal_weighted": 0.1, "temporal_raw": 0.9},
                    ],
                }},
            ],
        })
        turn_reader = TurnReader(storage, _FakeMetricsReader(), trace)
        client = TestClient(_wrap(turn_reader, storage, trace_reader=trace))

        response = client.get("/retrieval-quality")
        assert response.status_code == 200
        assert "Bello Features" in response.text

    def test_window_select_reflects_query_param(self, client: TestClient):
        response = client.get("/retrieval-quality", params={"window": "1h"})
        assert response.status_code == 200
        assert 'value="1h" selected' in response.text

    def test_unknown_window_returns_400(self, client: TestClient):
        response = client.get("/retrieval-quality", params={"window": "3y"})
        assert response.status_code == 400

    def test_nav_link_present(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert '/retrieval-quality"' in response.text

    def test_near_miss_kind_filter_scopes_results(self, storage: MockStorage):
        topic_id = storage.create_topic("Close topic", summary="", tags=[])
        item_id = storage.store_memory_item(MemoryItem(
            content="close item", item_type="fact", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=topic_id,
        ))
        trace = _TraceReaderWithData(events_of_type={
            "retrieval_scoring_summary": [
                {"payload": {"candidate_type": "topic", "top_candidates": [
                    {"candidate_type": "topic", "candidate_id": 99, "included": True,
                     "final_score": 0.5, "semantic_weighted": 0.4, "semantic_raw": 0.9,
                     "temporal_weighted": 0.1, "temporal_raw": 0.9},
                    {"candidate_type": "topic", "candidate_id": topic_id, "included": False,
                     "final_score": 0.46, "semantic_weighted": 0.4, "semantic_raw": 0.9,
                     "temporal_weighted": 0.1, "temporal_raw": 0.9},
                    {"candidate_type": "memory_item", "candidate_id": item_id, "included": False,
                     "final_score": 0.46, "semantic_weighted": 0.4, "semantic_raw": 0.9,
                     "temporal_weighted": 0.1, "temporal_raw": 0.9},
                ]}},
            ],
        })
        turn_reader = TurnReader(storage, _FakeMetricsReader(), trace)
        client = TestClient(_wrap(turn_reader, storage, trace_reader=trace))

        response = client.get("/retrieval-quality", params={"near_miss_kind": "topic"})
        assert response.status_code == 200
        near_miss_section = response.text.split("<h2>Near misses</h2>")[1].split("<h2>")[0]
        assert "Close topic" in near_miss_section
        assert "close item" not in near_miss_section

    def test_near_miss_kind_select_reflects_query_param(self, storage: MockStorage):
        trace = _TraceReaderWithData(events_of_type={
            "retrieval_scoring_summary": [
                {"payload": {"candidate_type": "topic", "top_candidates": [
                    {"candidate_type": "topic", "candidate_id": 1, "included": True,
                     "final_score": 0.5, "semantic_weighted": 0.4, "semantic_raw": 0.9,
                     "temporal_weighted": 0.1, "temporal_raw": 0.9},
                ]}},
            ],
        })
        turn_reader = TurnReader(storage, _FakeMetricsReader(), trace)
        client = TestClient(_wrap(turn_reader, storage, trace_reader=trace))

        response = client.get("/retrieval-quality", params={"near_miss_kind": "cluster"})
        assert response.status_code == 200
        assert 'value="cluster" selected' in response.text


class TestPromptQuality:
    def test_renders_with_no_data(self, client: TestClient):
        response = client.get("/prompt-quality")
        assert response.status_code == 200
        assert "No <code>topic_detection</code> events" in response.text

    def test_renders_fallback_flip_flop_churn_and_outlier(self, storage: MockStorage):
        trace = _TraceReaderWithData(events_of_type={
            "topic_detection": [
                {"payload": {"response": "CONTINUE", "result_type": "CONTINUE"}, "request_id": "req-a", "created_at": "t1"},
                {
                    "payload": {"response": "sure, let's keep going", "result_type": "CONTINUE", "message": "what now"},
                    "request_id": "req-b", "created_at": "t2",
                },
            ],
            "topic_relabel": [
                {"payload": {"topic_id": 1, "label": "Weekend trip"}},
                {"payload": {"topic_id": 1, "label": "Grocery list"}},
            ],
            "fact_consolidation": [
                {"payload": {"upserts": [{"category": "work", "key": "role", "value": "engineer"}]}},
                {"payload": {"upserts": [{"category": "work", "key": "role", "value": "senior engineer"}]}},
            ],
        })
        storage.facts.append(Fact(peer_id="u1", category="typo_cat", key="k", value="v", source="ai"))
        turn_reader = TurnReader(storage, _FakeMetricsReader(), trace)
        client = TestClient(_wrap(turn_reader, storage, trace_reader=trace, fact_categories=["basic"]))

        response = client.get("/prompt-quality")
        assert response.status_code == 200
        assert "Grocery list" in response.text  # relabel flip-flop
        assert "typo_cat" in response.text  # category outlier
        assert "role" in response.text  # fact churn
        assert "what now" in response.text  # fallback example message preview
        assert "/turns/req-b" in response.text  # fallback example turn link

    def test_window_select_reflects_query_param(self, client: TestClient):
        response = client.get("/prompt-quality", params={"window": "1h"})
        assert response.status_code == 200
        assert 'value="1h" selected' in response.text

    def test_unknown_window_returns_400(self, client: TestClient):
        response = client.get("/prompt-quality", params={"window": "3y"})
        assert response.status_code == 400

    def test_nav_link_present(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert '/prompt-quality"' in response.text


class TestTopicLifecycle:
    def test_renders_with_no_data(self, client: TestClient):
        response = client.get("/topic-lifecycle")
        assert response.status_code == 200
        assert "No topics closed in this window" in response.text

    def test_renders_lifespan_and_stale_sweep(self, storage: MockStorage):
        tid = storage.create_topic("Short-lived topic", peer_id="u1")
        storage.close_topic(tid, "done")
        storage.topics[tid]["created_at"] = "2026-07-28T10:00:00"
        storage.topics[tid]["closed_at"] = "2026-07-28T10:30:00"
        storage.topics[tid]["message_count"] = 2

        trace = _TraceReaderWithData(events_of_type={
            "stale_topic_cleanup": [{"payload": {"closed_topic_ids": [tid]}}],
        })
        turn_reader = TurnReader(storage, _FakeMetricsReader(), trace)
        client = TestClient(_wrap(turn_reader, storage, trace_reader=trace))

        response = client.get("/topic-lifecycle")
        assert response.status_code == 200
        assert "Short-lived topic" in response.text  # shortest-lived table
        assert "0.50h" in response.text  # lifespan_hours

    def test_window_select_reflects_query_param(self, client: TestClient):
        response = client.get("/topic-lifecycle", params={"window": "1h"})
        assert response.status_code == 200
        assert 'value="1h" selected' in response.text

    def test_unknown_window_returns_400(self, client: TestClient):
        response = client.get("/topic-lifecycle", params={"window": "3y"})
        assert response.status_code == 400

    def test_nav_link_present(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert '/topic-lifecycle"' in response.text


class TestClusterDynamics:
    def test_renders_with_no_data(self, client: TestClient):
        response = client.get("/cluster-dynamics")
        assert response.status_code == 200
        assert "No <code>cluster_consolidation</code> events" in response.text

    def test_renders_decisions_and_most_active_cluster(self, storage: MockStorage):
        cid = storage.create_cluster("u1", "Recurring topic", "summary")
        storage.clusters[cid]["topic_count"] = 4
        trace = _TraceReaderWithData(events_of_type={
            "cluster_consolidation": [
                {"payload": {
                    "match_index": 1,
                    "candidates": [{"kind": "cluster", "id": cid, "label": "Recurring topic"}],
                    "resulting_cluster_id": cid,
                }},
                {"payload": {
                    "match_index": 1,
                    "candidates": [{"kind": "topic", "id": 99, "label": "Old chat"}],
                    "resulting_cluster_id": 500,
                }},
                {"payload": {"match_index": None, "candidates": [], "resulting_cluster_id": None}},
            ],
        })
        turn_reader = TurnReader(storage, _FakeMetricsReader(), trace)
        client = TestClient(_wrap(turn_reader, storage, trace_reader=trace))

        response = client.get("/cluster-dynamics")
        assert response.status_code == 200
        assert "Recurring topic" in response.text  # most-active cluster label
        assert "3 clustering decisions" in response.text

    def test_window_select_reflects_query_param(self, client: TestClient):
        response = client.get("/cluster-dynamics", params={"window": "1h"})
        assert response.status_code == 200
        assert 'value="1h" selected' in response.text

    def test_unknown_window_returns_400(self, client: TestClient):
        response = client.get("/cluster-dynamics", params={"window": "3y"})
        assert response.status_code == 400

    def test_nav_link_present(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert '/cluster-dynamics"' in response.text


class TestBrowseTopics:
    def test_renders_empty_state(self, client: TestClient):
        response = client.get("/topics")
        assert response.status_code == 200
        assert "No topics found" in response.text

    def test_renders_and_filters(self, storage: MockStorage):
        tid = storage.create_topic("Bike trip", peer_id="u1")
        storage.close_topic(tid, "done")
        storage.create_topic("Open one", peer_id="u2")
        turn_reader = TurnReader(storage, _FakeMetricsReader(), _FakeTraceReader())
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get("/topics", params={"status": "closed", "peer_id": "u1"})
        assert response.status_code == 200
        assert "Bike trip" in response.text
        assert "Open one" not in response.text

    def test_empty_filter_param_is_not_a_literal_match(self, storage: MockStorage):
        # A submitted-but-blank <select>/pagination link carries status=""
        # over the wire — must behave like "no filter", not "status equals
        # the empty string" (which would hide every topic).
        storage.create_topic("Only topic", peer_id="u1")
        turn_reader = TurnReader(storage, _FakeMetricsReader(), _FakeTraceReader())
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get("/topics", params={"status": "", "peer_id": ""})
        assert response.status_code == 200
        assert "Only topic" in response.text


class TestBrowseFacts:
    def test_renders_empty_state(self, client: TestClient):
        response = client.get("/facts")
        assert response.status_code == 200
        assert "No facts found" in response.text

    def test_renders_and_filters(self, storage: MockStorage):
        storage.facts.append(Fact(peer_id="u1", category="basic", key="name", value="Alex", source="user"))
        storage.facts.append(Fact(peer_id="u1", category="work", key="role", value="engineer", source="ai"))
        turn_reader = TurnReader(storage, _FakeMetricsReader(), _FakeTraceReader())
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get("/facts", params={"category": "work"})
        assert response.status_code == 200
        assert "engineer" in response.text
        assert "Alex" not in response.text

    def test_pagination_link_carries_filter_and_page(self, storage: MockStorage):
        for i in range(60):
            storage.facts.append(Fact(peer_id="u1", category="basic", key=f"k{i}", value="v", source="user"))
        turn_reader = TurnReader(storage, _FakeMetricsReader(), _FakeTraceReader())
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get("/facts", params={"category": "basic"})
        assert response.status_code == 200
        assert "Page 1 of 2 (60 total)" in response.text
        assert 'href="http://testserver/facts?category=basic&amp;page=2"' in response.text

    def test_sort_by_importance_orders_high_first(self, storage: MockStorage):
        storage.facts.append(Fact(peer_id="u1", category="a", key="low", value="v", source="user", importance="low"))
        storage.facts.append(Fact(peer_id="u1", category="a", key="high", value="v", source="user", importance="high"))
        turn_reader = TurnReader(storage, _FakeMetricsReader(), _FakeTraceReader())
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get("/facts", params={"sort": "importance"})
        assert response.status_code == 200
        assert response.text.index(">high<") < response.text.index(">low<")


class TestBrowseMemoryItems:
    def test_renders_empty_state(self, client: TestClient):
        response = client.get("/memory-items")
        assert response.status_code == 200
        assert "No memory items found" in response.text

    def test_renders_and_filters(self, storage: MockStorage):
        t1 = storage.create_topic("First", peer_id="u1")
        t2 = storage.create_topic("Second", peer_id="u1")
        storage.store_memory_item(MemoryItem(
            content="a fact", item_type="fact", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=t1,
        ))
        storage.store_memory_item(MemoryItem(
            content="other topic", item_type="fact", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=t2,
        ))
        turn_reader = TurnReader(storage, _FakeMetricsReader(), _FakeTraceReader())
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get("/memory-items", params={"topic_id": t1})
        assert response.status_code == 200
        assert "a fact" in response.text
        assert "other topic" not in response.text

    def test_non_numeric_topic_id_ignored_rather_than_500(self, client: TestClient):
        response = client.get("/memory-items", params={"topic_id": "not-a-number"})
        assert response.status_code == 200

    def test_search_by_q_filters_content(self, storage: MockStorage):
        topic_id = storage.create_topic("Topic", peer_id="u1")
        storage.store_memory_item(MemoryItem(
            content="loves hiking", item_type="fact", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=topic_id,
        ))
        storage.store_memory_item(MemoryItem(
            content="works as a teacher", item_type="fact", source_peer="u1", subject_peer="u1",
            attribution_type="self_statement", topic_id=topic_id,
        ))
        turn_reader = TurnReader(storage, _FakeMetricsReader(), _FakeTraceReader())
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get("/memory-items", params={"q": "hiking"})
        assert response.status_code == 200
        assert "loves hiking" in response.text
        assert "works as a teacher" not in response.text


class TestBrowseClusters:
    def test_renders_empty_state(self, client: TestClient):
        response = client.get("/clusters")
        assert response.status_code == 200
        assert "No clusters found" in response.text

    def test_renders_and_filters(self, storage: MockStorage):
        storage.create_cluster("u1", "Bike", "a summary")
        storage.create_cluster("u2", "Trip", "another summary")
        turn_reader = TurnReader(storage, _FakeMetricsReader(), _FakeTraceReader())
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get("/clusters", params={"peer_id": "u1"})
        assert response.status_code == 200
        assert "Bike" in response.text
        assert "Trip" not in response.text

    def test_row_links_to_cluster_detail(self, storage: MockStorage):
        cid = storage.create_cluster("u1", "Bike", "a summary")
        turn_reader = TurnReader(storage, _FakeMetricsReader(), _FakeTraceReader())
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get("/clusters")
        assert response.status_code == 200
        assert f'href="http://testserver/clusters/{cid}"' in response.text


class TestClusterDetail:
    def test_404_for_unknown_cluster(self, client: TestClient):
        response = client.get("/clusters/999")
        assert response.status_code == 404

    def test_renders_cluster_and_member_topics(self, storage: MockStorage):
        cid = storage.create_cluster("u1", "Bike", "a long cluster summary", tags=["fitness"])
        t1 = storage.create_topic("First ride", peer_id="u1")
        t2 = storage.create_topic("Second ride", peer_id="u1")
        storage.assign_topic_to_cluster(t1, cid)
        storage.assign_topic_to_cluster(t2, cid)
        turn_reader = TurnReader(storage, _FakeMetricsReader(), _FakeTraceReader())
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get(f"/clusters/{cid}")
        assert response.status_code == 200
        assert "Bike" in response.text
        assert "a long cluster summary" in response.text
        assert "fitness" in response.text
        assert "First ride" in response.text
        assert "Second ride" in response.text

    def test_back_link_to_clusters_list(self, storage: MockStorage):
        cid = storage.create_cluster("u1", "Bike", "summary")
        turn_reader = TurnReader(storage, _FakeMetricsReader(), _FakeTraceReader())
        client = TestClient(_wrap(turn_reader, storage))

        response = client.get(f"/clusters/{cid}")
        assert response.status_code == 200
        assert 'href="http://testserver/clusters"' in response.text


class TestNav:
    def test_nav_shows_browse_and_insights_groups(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "Browse</span>" in response.text
        assert "Insights</span>" in response.text
        assert '/topics"' in response.text
        assert '/facts"' in response.text
        assert '/memory-items"' in response.text
        assert '/clusters"' in response.text

    def test_current_page_link_marked_active(self, client: TestClient):
        response = client.get("/topics")
        assert response.status_code == 200
        assert 'class="nav-link active"' in response.text
