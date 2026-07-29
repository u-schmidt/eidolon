"""Real end-to-end smoke test: a real uvicorn subprocess + a real
requests-based EidolonHttpClient talking to it over actual sockets.

The parametrized contract suite (test_eidolon_client_contract.py) covers
behavioral parity fast, in-process, via FastAPI's TestClient — deliberately
not exercising real serialization-over-real-sockets or real connection
behavior. This test is the one place that gap is covered, mirroring how
the Postgres migration paired a fast contract suite with a real rollout
smoke test rather than trusting either alone.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from client.http_client import EidolonHttpClient
from core.models import EnvironmentSnapshot


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_until_up(client: EidolonHttpClient, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client.get_active_topic("user")
            return
        except Exception as exc:  # noqa: BLE001 - server may not be listening yet
            last_exc = exc
            time.sleep(0.2)
    raise RuntimeError(f"server never became reachable: {last_exc}")


@pytest.fixture
def running_server(tmp_path: Path):
    config_path = tmp_path / "eidolon.yaml"
    db_path = tmp_path / "smoke.db"
    config_path.write_text(f"storage:\n  provider: sqlite\n  path: {db_path}\n")

    port = _free_port()
    project_root = Path(__file__).resolve().parent.parent.parent
    proc = subprocess.Popen(
        [
            sys.executable, str(project_root / "tools" / "serve.py"),
            "--config", str(config_path), "--port", str(port),
        ],
        env={**os.environ, "EIDOLON_DEPLOYMENT_TOKEN": "smoke-token"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


class TestEidolonHttpClientRealServer:
    def test_full_round_trip_over_real_sockets(self, running_server):
        port = running_server
        client = EidolonHttpClient(base_url=f"http://127.0.0.1:{port}", token="smoke-token", timeout=5.0)
        _wait_until_up(client)

        assert client.get_active_topic("user") is None

        result = client.prepare_context(
            message="Let's talk about Bello", workspace_id="user",
            environment=EnvironmentSnapshot(situation="conversation"),
        )
        assert result.topic_change.type == "NEW_TOPIC"
        assert result.bundle.current_topic_label == result.topic_change.label

        new_topic_id = client.store_turn(
            user_message="Let's talk about Bello", ai_response="Sure!", workspace_id="user",
            environment=EnvironmentSnapshot(situation="conversation"),
            new_topic_label=result.topic_change.label,
        )
        assert isinstance(new_topic_id, int)
        assert client.get_active_topic("user") == new_topic_id

    def test_wrong_token_is_rejected(self, running_server):
        port = running_server
        good = EidolonHttpClient(base_url=f"http://127.0.0.1:{port}", token="smoke-token", timeout=5.0)
        _wait_until_up(good)

        bad = EidolonHttpClient(base_url=f"http://127.0.0.1:{port}", token="wrong-token", timeout=5.0)
        with pytest.raises(Exception, match="401"):
            bad.get_active_topic("user")
