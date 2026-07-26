"""FastAPI TestClient smoke tests for the turn-detail dashboard app —
confirms both routes render for a rich-mode and a skeleton-mode turn,
and that the optional HTTP Basic auth gate behaves correctly."""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from tests.mocks import MockStorage
from tools.dashboard.app import create_dashboard_app
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


@pytest.fixture
def storage() -> MockStorage:
    return MockStorage()


@pytest.fixture
def client(storage: MockStorage) -> TestClient:
    turn_reader = TurnReader(storage, _FakeMetricsReader(), _FakeTraceReader())
    app = create_dashboard_app(turn_reader)
    return TestClient(app)


class TestIndex:
    def test_lists_recent_turns(self, client: TestClient, storage: MockStorage):
        storage.store_message(turn_id=1, role="user", content="hello there", peer_id="u1", request_id="req1")

        response = client.get("/")
        assert response.status_code == 200
        assert "hello there" in response.text
        assert "req1" in response.text

    def test_empty_state_when_no_turns(self, client: TestClient):
        response = client.get("/")
        assert response.status_code == 200
        assert "No turns found" in response.text


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
        client = TestClient(create_dashboard_app(turn_reader))

        response = client.get("/turns/req1")
        assert response.status_code == 200
        assert "topic_detection" in response.text

    def test_unknown_request_id_returns_404(self, client: TestClient):
        response = client.get("/turns/nope")
        assert response.status_code == 404


class TestAuth:
    def test_no_auth_required_when_password_unset(self, client: TestClient, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("EIDOLON_DASHBOARD_PASSWORD", raising=False)
        assert client.get("/").status_code == 200

    def test_rejects_missing_credentials_when_password_set(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("EIDOLON_DASHBOARD_PASSWORD", "secret")
        assert client.get("/").status_code == 401

    def test_accepts_correct_password(self, client: TestClient, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("EIDOLON_DASHBOARD_PASSWORD", "secret")
        creds = base64.b64encode(b"eidolon:secret").decode()
        response = client.get("/", headers={"Authorization": f"Basic {creds}"})
        assert response.status_code == 200

    def test_rejects_wrong_password(self, client: TestClient, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("EIDOLON_DASHBOARD_PASSWORD", "secret")
        creds = base64.b64encode(b"eidolon:wrong").decode()
        response = client.get("/", headers={"Authorization": f"Basic {creds}"})
        assert response.status_code == 401
