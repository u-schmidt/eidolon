"""FastAPI app exposing Eidolon's core facade over HTTP, plus (optionally)
the turn-detail dashboard.

One route per ``EidolonClient`` method (core/eidolon_client.py) — a
thin translation layer, not a second implementation of core's logic.
Route handlers are free functions rather than class methods: this
follows the same established exception AGENTS.md already carves out for
tools/'s entrypoint/glue code (``build_arg_parser()``, ``main()``) —
FastAPI's routing idiom is function-based, and these handlers only
translate an HTTP request into a call on a real ``core.Eidolon``
instance, never containing business logic themselves.

Every RPC route is a POST with a single flat JSON object body
(including routes with only one field, via ``embed=True``) — a
deliberate, uniform choice, not REST resource semantics:
``EidolonClient`` is method-shaped, not resource-shaped, so the wire
format follows suit. Scalar parameters are annotated with ``Body(...)``
so they land in that JSON body instead of being inferred as query
parameters (which would otherwise put the full user message in a URL
for the ``environment``-carrying routes). Request/response bodies reuse
``core.models`` dataclasses directly — FastAPI converts plain stdlib
dataclasses via pydantic internally, with no parallel DTO layer needed.

The RPC routes and the dashboard routes are two ``APIRouter``s on one
``FastAPI`` app, not two separate apps — same ``Eidolon``/storage
instance underneath either way, but each router carries its own auth
scheme (bearer token vs. HTTP Basic — see server/auth.py) since they
serve different audiences (a machine caller vs. a human in a browser).
Kept as one process rather than a second standalone dashboard
container so there's exactly one thing ever writing to the database
through core's real orchestration — see README.md's "Turn Detail
Dashboard" section for the fuller rationale.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Body, Depends, FastAPI
from fastapi.staticfiles import StaticFiles

from core.eidolon_client import EidolonClient
from core.models import ContextBundle, EnvironmentSnapshot, PrefetchResult, TopicChange
from server.auth import require_dashboard_password, require_token
from tools.dashboard.app import DASHBOARD_STATIC_DIR, create_dashboard_router
from tools.dashboard.browse_reader import BrowseReader
from tools.dashboard.cluster_dynamics_reader import ClusterDynamicsReader
from tools.dashboard.performance_reader import PerformanceReader
from tools.dashboard.prompt_quality_reader import PromptQualityReader
from tools.dashboard.retrieval_quality_reader import RetrievalQualityReader
from tools.dashboard.topic_lifecycle_reader import TopicLifecycleReader
from tools.dashboard.turn_reader import TurnReader


def create_app(
    eidolon: EidolonClient,
    turn_reader: TurnReader | None = None,
    performance_reader: PerformanceReader | None = None,
    retrieval_quality_reader: RetrievalQualityReader | None = None,
    prompt_quality_reader: PromptQualityReader | None = None,
    topic_lifecycle_reader: TopicLifecycleReader | None = None,
    cluster_dynamics_reader: ClusterDynamicsReader | None = None,
    browse_reader: BrowseReader | None = None,
) -> FastAPI:
    """Build the FastAPI app wrapping an already-constructed ``EidolonClient``.

    ``eidolon`` is a real ``core.Eidolon``, built once at startup by
    ``tools/serve.py`` the same way ``tools/cli.py`` builds one — this
    app contains no core logic itself, only HTTP<->facade translation.

    ``turn_reader``/``performance_reader``/``retrieval_quality_reader``/
    ``prompt_quality_reader``/``topic_lifecycle_reader``/
    ``cluster_dynamics_reader``/``browse_reader`` are optional together:
    when given, the dashboard is mounted under ``/dashboard``; when
    omitted (the default — e.g. tests that only care about the RPC
    contract), the app is exactly the RPC API and nothing else. Any one
    given without the others is a caller bug, not a state worth
    validating for.
    """
    app = FastAPI(title="Eidolon")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        """Unauthenticated liveness probe for container orchestration —
        deliberately outside ``rpc`` (no bearer token needed) since a
        Docker/Compose healthcheck has no way to supply one. Process-alive
        only, no storage round-trip: if the process is wedged or crashed,
        the probe already fails by timing out/refusing the connection."""
        return {"status": "ok"}

    rpc = APIRouter(dependencies=[Depends(require_token)])

    @rpc.post("/active-topic", response_model=int | None)
    def get_active_topic(
        peer_id: str = Body(...),
        session_id: str | None = Body(None),
    ) -> int | None:
        return eidolon.get_active_topic(peer_id, session_id)

    @rpc.post("/prepare-context", response_model=PrefetchResult)
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

    @rpc.post("/detect-topic", response_model=TopicChange)
    def detect_topic(
        message: str = Body(...),
        active_topic_id: int | None = Body(None),
        session_id: str | None = Body(None),
        request_id: str | None = Body(None),
    ) -> TopicChange:
        return eidolon.detect_topic(
            message, active_topic_id, session_id=session_id, request_id=request_id,
        )

    @rpc.post("/retrieve-context", response_model=ContextBundle)
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

    @rpc.post("/store-turn", response_model=int | None)
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

    @rpc.post("/consolidate-facts", status_code=204)
    def consolidate_facts(
        peer_id: str = Body(...),
        request_id: str | None = Body(None),
        session_id: str | None = Body(None),
    ) -> None:
        eidolon.consolidate_facts(peer_id, request_id=request_id, session_id=session_id)

    @rpc.post("/close-stale-topics", response_model=list[int])
    def close_stale_topics(
        cutoff: datetime = Body(...),
        peer_id: str | None = Body(None),
        request_id: str | None = Body(None),
        session_id: str | None = Body(None),
    ) -> list[int]:
        return eidolon.close_stale_topics(
            cutoff, peer_id, request_id=request_id, session_id=session_id,
        )

    @rpc.post("/correct-fact", status_code=204)
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

    @rpc.post("/inspect-cluster", response_model=list[dict])
    def inspect_cluster(cluster_id: int = Body(..., embed=True)) -> list[dict]:
        return eidolon.inspect_cluster(cluster_id)

    @rpc.post("/inspect-topic", response_model=list[dict])
    def inspect_topic(topic_id: int = Body(..., embed=True)) -> list[dict]:
        return eidolon.inspect_topic(topic_id)

    app.include_router(rpc)

    if (
        turn_reader is not None
        and performance_reader is not None
        and retrieval_quality_reader is not None
        and prompt_quality_reader is not None
        and topic_lifecycle_reader is not None
        and cluster_dynamics_reader is not None
        and browse_reader is not None
    ):
        app.include_router(
            create_dashboard_router(
                turn_reader, performance_reader, retrieval_quality_reader,
                prompt_quality_reader, topic_lifecycle_reader, cluster_dynamics_reader,
                browse_reader,
            ),
            prefix="/dashboard",
            dependencies=[Depends(require_dashboard_password)],
        )
        app.mount(
            "/dashboard/static",
            StaticFiles(directory=str(DASHBOARD_STATIC_DIR)),
            name="dashboard-static",
        )

    return app
