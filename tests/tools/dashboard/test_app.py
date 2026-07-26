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

from tests.mocks import MockStorage
from tools.dashboard.app import DASHBOARD_STATIC_DIR, create_dashboard_router
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


class _FakeTraceReader:
    def __init__(self, events: dict[str, list[dict]] | None = None) -> None:
        self._events = events or {}

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def events_for_request(self, request_id: str) -> list[dict]:
        return self._events.get(request_id, [])


def _wrap(turn_reader: TurnReader) -> FastAPI:
    app = FastAPI()
    app.include_router(create_dashboard_router(turn_reader))
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_STATIC_DIR)), name="dashboard-static")
    return app


@pytest.fixture
def storage() -> MockStorage:
    return MockStorage()


@pytest.fixture
def client(storage: MockStorage) -> TestClient:
    turn_reader = TurnReader(storage, _FakeMetricsReader(), _FakeTraceReader())
    return TestClient(_wrap(turn_reader))


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
        client = TestClient(_wrap(turn_reader))

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
        client = TestClient(_wrap(turn_reader))

        response = client.get("/turns/req1")
        assert response.status_code == 200
        assert "Bello Features" in response.text
