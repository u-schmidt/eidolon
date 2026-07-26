"""HTTP implementation of EidolonClient — talks to server/app.py.

Uses ``requests`` — already an unconditional top-level dependency of
this project (see pyproject.toml) — rather than adding ``httpx`` or
similar. Every call site (the Hermes adapter's prefetch()/sync_turn()/
tool-call handlers) is already synchronous Python, so there's no async
context to preserve, and this keeps the "server" deployment mode from
needing any new package installed into the Hermes container's venv at
all — the concrete payoff of the whole feature.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import requests

from client import wire
from core.models import ContextBundle, EnvironmentSnapshot, PrefetchResult, TopicChange


class EidolonHttpClient:
    """HTTP client for a remote Eidolon core server (see tools/serve.py).

    Implements the same contract as ``core.Eidolon``
    (core/eidolon_client.py's ``EidolonClient`` Protocol) structurally —
    no inheritance needed.
    """

    def __init__(self, base_url: str, token: str, timeout: float = 30.0, session: Any = None) -> None:
        """``session`` defaults to a real ``requests.Session()`` — the
        production path. Tests may inject a FastAPI ``TestClient``
        instead (it implements the same ``.post(url, json=..., headers=
        ...)``-shaped interface deliberately, for exactly this kind of
        substitution), to exercise this class against a real in-process
        app with no real sockets — see tests/conftest.py.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = session if session is not None else requests.Session()
        self._session.headers["Authorization"] = f"Bearer {token}"

    def _call(self, path: str, body: dict[str, Any]) -> Any:
        # A real requests.Session needs an explicit per-request timeout —
        # an injected test session (e.g. FastAPI's TestClient) has no
        # real network to time out on, and passing timeout= to it is a
        # deprecated no-op there (see tests/conftest.py's make_eidolon_client).
        kwargs: dict[str, Any] = {"json": wire.to_jsonable(body)}
        if isinstance(self._session, requests.Session):
            kwargs["timeout"] = self._timeout
        response = self._session.post(f"{self._base_url}{path}", **kwargs)
        response.raise_for_status()
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def get_active_topic(self, peer_id: str, session_id: str | None = None) -> int | None:
        return self._call("/active-topic", {"peer_id": peer_id, "session_id": session_id})

    def prepare_context(
        self,
        message: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        active_topic_id: int | None = None,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> PrefetchResult:
        data = self._call("/prepare-context", {
            "message": message, "peer_id": peer_id, "environment": environment,
            "active_topic_id": active_topic_id, "session_id": session_id, "request_id": request_id,
        })
        return wire.prefetch_result_from_dict(data)

    def detect_topic(
        self,
        message: str,
        active_topic_id: int | None = None,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> TopicChange:
        data = self._call("/detect-topic", {
            "message": message, "active_topic_id": active_topic_id,
            "session_id": session_id, "request_id": request_id,
        })
        return wire.topic_change_from_dict(data)

    def retrieve_context(
        self,
        message: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        topic_label: str | None = None,
        is_new_topic: bool = False,
        active_topic_id: int | None = None,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> ContextBundle:
        data = self._call("/retrieve-context", {
            "message": message, "peer_id": peer_id, "environment": environment,
            "topic_label": topic_label, "is_new_topic": is_new_topic,
            "active_topic_id": active_topic_id, "session_id": session_id, "request_id": request_id,
        })
        return wire.context_bundle_from_dict(data)

    def store_turn(
        self,
        user_message: str,
        ai_response: str,
        peer_id: str,
        environment: EnvironmentSnapshot,
        turn_id: int | None = None,
        old_topic_id: int | None = None,
        new_topic_label: str | None = None,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> int | None:
        return self._call("/store-turn", {
            "user_message": user_message, "ai_response": ai_response, "peer_id": peer_id,
            "environment": environment, "turn_id": turn_id, "old_topic_id": old_topic_id,
            "new_topic_label": new_topic_label, "session_id": session_id, "request_id": request_id,
        })

    def consolidate_facts(
        self,
        peer_id: str,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._call("/consolidate-facts", {
            "peer_id": peer_id, "request_id": request_id, "session_id": session_id,
        })

    def close_stale_topics(
        self,
        cutoff: datetime,
        peer_id: str | None = None,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> list[int]:
        return self._call("/close-stale-topics", {
            "cutoff": cutoff, "peer_id": peer_id, "request_id": request_id, "session_id": session_id,
        })

    def correct_fact(
        self,
        peer_id: str,
        category: str,
        key: str,
        value: str | None = None,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self._call("/correct-fact", {
            "peer_id": peer_id, "category": category, "key": key, "value": value,
            "request_id": request_id, "session_id": session_id,
        })

    def inspect_cluster(self, cluster_id: int) -> list[dict]:
        return self._call("/inspect-cluster", {"cluster_id": cluster_id})

    def inspect_topic(self, topic_id: int) -> list[dict]:
        return self._call("/inspect-topic", {"topic_id": topic_id})
