"""Tests for server/app.py's merged RPC + dashboard FastAPI app.

The RPC contract itself (request/response shapes) is already covered
end-to-end by tests/core/test_eidolon_client_contract.py via
tests/conftest.py's "http" eidolon_client branch. This file covers what
that doesn't: the two-router auth split — RPC routes keep requiring the
bearer token, the dashboard (when mounted) requires its own, separate,
optional password, and neither leaks into the other.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from core.models import ContextBundle, PrefetchResult, TopicChange
from server.app import create_app
from tests.mocks import MockStorage
from tools.dashboard.performance_reader import PerformanceReader
from tools.dashboard.turn_reader import TurnReader

_TOKEN = "test-token"


class _FakeEidolonClient:
    """Only implements what these tests actually call — structural
    Protocol conformance, not a full contract double (see
    test_eidolon_client_contract.py for that)."""

    def get_active_topic(self, peer_id: str, session_id: str | None = None) -> int | None:
        return 42

    def prepare_context(self, *args, **kwargs) -> PrefetchResult:
        raise NotImplementedError

    def detect_topic(self, *args, **kwargs) -> TopicChange:
        raise NotImplementedError

    def retrieve_context(self, *args, **kwargs) -> ContextBundle:
        raise NotImplementedError

    def store_turn(self, *args, **kwargs) -> int | None:
        raise NotImplementedError

    def consolidate_facts(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def close_stale_topics(self, *args, **kwargs) -> list[int]:
        raise NotImplementedError

    def correct_fact(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def inspect_cluster(self, *args, **kwargs) -> list[dict]:
        raise NotImplementedError


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
    def events_for_request(self, request_id: str) -> list[dict]:
        return []


@pytest.fixture(autouse=True)
def _token_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EIDOLON_DEPLOYMENT_TOKEN", _TOKEN)
    monkeypatch.delenv("EIDOLON_DASHBOARD_PASSWORD", raising=False)


@pytest.fixture
def turn_reader() -> TurnReader:
    return TurnReader(MockStorage(), _FakeMetricsReader(), _FakeTraceReader())


@pytest.fixture
def performance_reader() -> PerformanceReader:
    return PerformanceReader(_FakeMetricsReader())


def _rpc_call(client: TestClient, token: str | None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post("/active-topic", json={"peer_id": "u1"}, headers=headers)


class TestRpcAuthUnaffectedByDashboard:
    def test_rpc_requires_token_with_no_dashboard_mounted(self):
        client = TestClient(create_app(_FakeEidolonClient()))
        assert _rpc_call(client, None).status_code == 401
        assert _rpc_call(client, _TOKEN).status_code == 200

    def test_rpc_requires_token_with_dashboard_mounted(self, turn_reader: TurnReader, performance_reader: PerformanceReader):
        client = TestClient(create_app(_FakeEidolonClient(), turn_reader, performance_reader))
        assert _rpc_call(client, None).status_code == 401
        assert _rpc_call(client, _TOKEN).status_code == 200

    def test_rpc_still_requires_token_when_dashboard_password_also_set(
        self, turn_reader: TurnReader, performance_reader: PerformanceReader, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("EIDOLON_DASHBOARD_PASSWORD", "dashsecret")
        client = TestClient(create_app(_FakeEidolonClient(), turn_reader, performance_reader))
        assert _rpc_call(client, None).status_code == 401
        # A dashboard credential must not satisfy the RPC's bearer check.
        response = client.post(
            "/active-topic", json={"peer_id": "u1"},
            auth=("eidolon", "dashsecret"),
        )
        assert response.status_code == 401


class TestDashboardMounting:
    def test_absent_when_no_turn_reader(self):
        client = TestClient(create_app(_FakeEidolonClient()))
        assert client.get("/dashboard/").status_code == 404

    def test_present_when_turn_reader_given(self, turn_reader: TurnReader, performance_reader: PerformanceReader):
        client = TestClient(create_app(_FakeEidolonClient(), turn_reader, performance_reader))
        assert client.get("/dashboard/").status_code == 200

    def test_turn_detail_route_present(self, turn_reader: TurnReader, performance_reader: PerformanceReader):
        client = TestClient(create_app(_FakeEidolonClient(), turn_reader, performance_reader))
        assert client.get("/dashboard/turns/nope").status_code == 404  # unknown turn, not unknown route

    def test_static_assets_served(self, turn_reader: TurnReader, performance_reader: PerformanceReader):
        client = TestClient(create_app(_FakeEidolonClient(), turn_reader, performance_reader))
        response = client.get("/dashboard/static/htmx.min.js")
        assert response.status_code == 200

    def test_performance_route_present(self, turn_reader: TurnReader, performance_reader: PerformanceReader):
        client = TestClient(create_app(_FakeEidolonClient(), turn_reader, performance_reader))
        assert client.get("/dashboard/performance").status_code == 200


class TestDashboardAuthUnaffectedByRpc:
    def test_no_auth_required_when_password_unset(self, turn_reader: TurnReader, performance_reader: PerformanceReader):
        client = TestClient(create_app(_FakeEidolonClient(), turn_reader, performance_reader))
        assert client.get("/dashboard/").status_code == 200

    def test_rejects_missing_credentials_when_password_set(
        self, turn_reader: TurnReader, performance_reader: PerformanceReader, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("EIDOLON_DASHBOARD_PASSWORD", "dashsecret")
        client = TestClient(create_app(_FakeEidolonClient(), turn_reader, performance_reader))
        assert client.get("/dashboard/").status_code == 401

    def test_accepts_correct_password(self, turn_reader: TurnReader, performance_reader: PerformanceReader, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("EIDOLON_DASHBOARD_PASSWORD", "dashsecret")
        client = TestClient(create_app(_FakeEidolonClient(), turn_reader, performance_reader))
        response = client.get("/dashboard/", auth=("eidolon", "dashsecret"))
        assert response.status_code == 200

    def test_rpc_bearer_token_does_not_satisfy_dashboard_auth(
        self, turn_reader: TurnReader, performance_reader: PerformanceReader, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("EIDOLON_DASHBOARD_PASSWORD", "dashsecret")
        client = TestClient(create_app(_FakeEidolonClient(), turn_reader, performance_reader))
        response = client.get("/dashboard/", headers={"Authorization": f"Bearer {_TOKEN}"})
        assert response.status_code == 401
