"""FastAPI app exposing Eidolon's core facade over HTTP.

One route per ``EidolonClient`` method (core/eidolon_client.py) — a
thin translation layer, not a second implementation of core's logic.
Route handlers are free functions rather than class methods: this
follows the same established exception AGENTS.md already carves out for
tools/'s entrypoint/glue code (``build_arg_parser()``, ``main()``) —
FastAPI's routing idiom is function-based, and these handlers only
translate an HTTP request into a call on a real ``core.Eidolon``
instance, never containing business logic themselves.

Every route is a POST with a single flat JSON object body (including
routes with only one field, via ``embed=True``) — a deliberate, uniform
choice, not REST resource semantics: ``EidolonClient`` is method-shaped,
not resource-shaped, so the wire format follows suit. Scalar parameters
are annotated with ``Body(...)`` so they land in that JSON body instead
of being inferred as query parameters (which would otherwise put the
full user message in a URL for the ``environment``-carrying routes).
Request/response bodies reuse ``core.models`` dataclasses directly —
FastAPI converts plain stdlib dataclasses via pydantic internally, with
no parallel DTO layer needed.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import Body, Depends, FastAPI

from core.eidolon_client import EidolonClient
from core.models import ContextBundle, EnvironmentSnapshot, PrefetchResult, TopicChange
from server.auth import require_token


def create_app(eidolon: EidolonClient) -> FastAPI:
    """Build the FastAPI app wrapping an already-constructed ``EidolonClient``.

    ``eidolon`` is a real ``core.Eidolon``, built once at startup by
    ``tools/serve.py`` the same way ``tools/cli.py`` builds one — this
    app contains no core logic itself, only HTTP<->facade translation.
    """
    app = FastAPI(title="Eidolon", dependencies=[Depends(require_token)])

    @app.post("/active-topic", response_model=int | None)
    def get_active_topic(
        peer_id: str = Body(...),
        session_id: str | None = Body(None),
    ) -> int | None:
        return eidolon.get_active_topic(peer_id, session_id)

    @app.post("/prepare-context", response_model=PrefetchResult)
    def prepare_context(
        message: str = Body(...),
        peer_id: str = Body(...),
        environment: EnvironmentSnapshot = Body(...),
        active_topic_id: int | None = Body(None),
        session_id: str | None = Body(None),
        request_id: str | None = Body(None),
    ) -> PrefetchResult:
        return eidolon.prepare_context(
            message, peer_id, environment, active_topic_id,
            session_id=session_id, request_id=request_id,
        )

    @app.post("/detect-topic", response_model=TopicChange)
    def detect_topic(
        message: str = Body(...),
        active_topic_id: int | None = Body(None),
        session_id: str | None = Body(None),
        request_id: str | None = Body(None),
    ) -> TopicChange:
        return eidolon.detect_topic(
            message, active_topic_id, session_id=session_id, request_id=request_id,
        )

    @app.post("/retrieve-context", response_model=ContextBundle)
    def retrieve_context(
        message: str = Body(...),
        peer_id: str = Body(...),
        environment: EnvironmentSnapshot = Body(...),
        topic_label: str | None = Body(None),
        is_new_topic: bool = Body(False),
        active_topic_id: int | None = Body(None),
        session_id: str | None = Body(None),
        request_id: str | None = Body(None),
    ) -> ContextBundle:
        return eidolon.retrieve_context(
            message, peer_id, environment, topic_label, is_new_topic, active_topic_id,
            session_id=session_id, request_id=request_id,
        )

    @app.post("/store-turn", response_model=int | None)
    def store_turn(
        user_message: str = Body(...),
        ai_response: str = Body(...),
        peer_id: str = Body(...),
        environment: EnvironmentSnapshot = Body(...),
        turn_id: int | None = Body(None),
        old_topic_id: int | None = Body(None),
        new_topic_label: str | None = Body(None),
        session_id: str | None = Body(None),
        request_id: str | None = Body(None),
    ) -> int | None:
        return eidolon.store_turn(
            user_message, ai_response, peer_id, environment, turn_id,
            old_topic_id, new_topic_label, session_id=session_id, request_id=request_id,
        )

    @app.post("/consolidate-facts", status_code=204)
    def consolidate_facts(
        peer_id: str = Body(...),
        request_id: str | None = Body(None),
        session_id: str | None = Body(None),
    ) -> None:
        eidolon.consolidate_facts(peer_id, request_id=request_id, session_id=session_id)

    @app.post("/close-stale-topics", response_model=list[int])
    def close_stale_topics(
        cutoff: datetime = Body(...),
        peer_id: str | None = Body(None),
        request_id: str | None = Body(None),
        session_id: str | None = Body(None),
    ) -> list[int]:
        return eidolon.close_stale_topics(
            cutoff, peer_id, request_id=request_id, session_id=session_id,
        )

    @app.post("/correct-fact", status_code=204)
    def correct_fact(
        peer_id: str = Body(...),
        category: str = Body(...),
        key: str = Body(...),
        value: str | None = Body(None),
        request_id: str | None = Body(None),
        session_id: str | None = Body(None),
    ) -> None:
        eidolon.correct_fact(
            peer_id, category, key, value, request_id=request_id, session_id=session_id,
        )

    @app.post("/inspect-cluster", response_model=list[dict])
    def inspect_cluster(cluster_id: int = Body(..., embed=True)) -> list[dict]:
        return eidolon.inspect_cluster(cluster_id)

    return app
