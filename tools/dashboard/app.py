"""FastAPI router for Eidolon's turn-detail dashboard.

A read-only, server-rendered (Jinja2 + htmx) view over one turn
(``request_id``) at a time — see ``tools/dashboard/turn_reader.py`` for
the data-access layer this only translates into HTML. Route handlers
are free functions rather than class methods, the same established
exception AGENTS.md already carves out for tools/'s and server/app.py's
entrypoint-style glue code.

An ``APIRouter``, not a standalone app: mounted onto ``server/app.py``'s
``create_app()`` under ``/dashboard``, sharing that process's single
``Eidolon``/storage instance rather than running as its own container
with its own database connection — see ``create_app()``'s docstring for
why. This router carries no auth dependency of its own; the app
composing it decides how to gate it (``server/auth.py``'s
``require_dashboard_password``).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from tools.dashboard.cluster_dynamics_reader import ClusterDynamicsReader
from tools.dashboard.performance_reader import PerformanceReader
from tools.dashboard.prompt_quality_reader import PromptQualityReader
from tools.dashboard.retrieval_quality_reader import RetrievalQualityReader
from tools.dashboard.topic_lifecycle_reader import TopicLifecycleReader
from tools.dashboard.turn_reader import TurnReader

_DASHBOARD_DIR = Path(__file__).parent
DASHBOARD_STATIC_DIR = _DASHBOARD_DIR / "static"
_templates = Jinja2Templates(directory=str(_DASHBOARD_DIR / "templates"))

# Fixed allow-list rather than free-form duration parsing — keeps the
# window <select> control trivial and avoids validating arbitrary
# duration strings from a query param.
_WINDOW_CUTOFFS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
}
_DEFAULT_WINDOW = "24h"


def create_dashboard_router(
    turn_reader: TurnReader,
    performance_reader: PerformanceReader,
    retrieval_quality_reader: RetrievalQualityReader,
    prompt_quality_reader: PromptQualityReader,
    topic_lifecycle_reader: TopicLifecycleReader,
    cluster_dynamics_reader: ClusterDynamicsReader,
) -> APIRouter:
    """Build the dashboard's routes wrapping already-constructed readers."""
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse, name="dashboard_index")
    def landing(request: Request) -> HTMLResponse:
        recent = turn_reader.recent_turns(limit=5)
        return _templates.TemplateResponse(
            request, "index.html", {"recent_turns": recent},
        )

    @router.get("/turns", response_class=HTMLResponse, name="dashboard_turns_list")
    def turns_list(request: Request, peer_id: str | None = None, limit: int = 50) -> HTMLResponse:
        turns = turn_reader.recent_turns(limit=limit, peer_id=peer_id)
        return _templates.TemplateResponse(
            request, "turns.html", {"turns": turns, "peer_id": peer_id or ""},
        )

    @router.get("/turns/{request_id}", response_class=HTMLResponse, name="dashboard_turn_detail")
    def turn_detail(request: Request, request_id: str) -> HTMLResponse:
        detail = turn_reader.get(request_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"No turn found for request_id={request_id!r}")
        return _templates.TemplateResponse(
            request, "turn_detail.html", {"detail": detail},
        )

    @router.get("/performance", response_class=HTMLResponse, name="dashboard_performance")
    def performance(request: Request, window: str = _DEFAULT_WINDOW) -> HTMLResponse:
        if window not in _WINDOW_CUTOFFS:
            raise HTTPException(status_code=400, detail=f"Unknown window {window!r} — use one of {list(_WINDOW_CUTOFFS)}")
        since = datetime.now(UTC) - _WINDOW_CUTOFFS[window]
        summary = performance_reader.summary(since)
        return _templates.TemplateResponse(
            request, "performance.html",
            {"summary": summary, "window": window, "windows": list(_WINDOW_CUTOFFS)},
        )

    @router.get("/retrieval-quality", response_class=HTMLResponse, name="dashboard_retrieval_quality")
    def retrieval_quality(request: Request, window: str = _DEFAULT_WINDOW) -> HTMLResponse:
        if window not in _WINDOW_CUTOFFS:
            raise HTTPException(status_code=400, detail=f"Unknown window {window!r} — use one of {list(_WINDOW_CUTOFFS)}")
        since = datetime.now(UTC) - _WINDOW_CUTOFFS[window]
        summary = retrieval_quality_reader.summary(since)
        return _templates.TemplateResponse(
            request, "retrieval_quality.html",
            {"summary": summary, "window": window, "windows": list(_WINDOW_CUTOFFS)},
        )

    @router.get("/prompt-quality", response_class=HTMLResponse, name="dashboard_prompt_quality")
    def prompt_quality(request: Request, window: str = _DEFAULT_WINDOW) -> HTMLResponse:
        if window not in _WINDOW_CUTOFFS:
            raise HTTPException(status_code=400, detail=f"Unknown window {window!r} — use one of {list(_WINDOW_CUTOFFS)}")
        since = datetime.now(UTC) - _WINDOW_CUTOFFS[window]
        summary = prompt_quality_reader.summary(since)
        return _templates.TemplateResponse(
            request, "prompt_quality.html",
            {"summary": summary, "window": window, "windows": list(_WINDOW_CUTOFFS)},
        )

    @router.get("/topic-lifecycle", response_class=HTMLResponse, name="dashboard_topic_lifecycle")
    def topic_lifecycle(request: Request, window: str = _DEFAULT_WINDOW) -> HTMLResponse:
        if window not in _WINDOW_CUTOFFS:
            raise HTTPException(status_code=400, detail=f"Unknown window {window!r} — use one of {list(_WINDOW_CUTOFFS)}")
        since = datetime.now(UTC) - _WINDOW_CUTOFFS[window]
        summary = topic_lifecycle_reader.summary(since)
        return _templates.TemplateResponse(
            request, "topic_lifecycle.html",
            {"summary": summary, "window": window, "windows": list(_WINDOW_CUTOFFS)},
        )

    @router.get("/cluster-dynamics", response_class=HTMLResponse, name="dashboard_cluster_dynamics")
    def cluster_dynamics(request: Request, window: str = _DEFAULT_WINDOW) -> HTMLResponse:
        if window not in _WINDOW_CUTOFFS:
            raise HTTPException(status_code=400, detail=f"Unknown window {window!r} — use one of {list(_WINDOW_CUTOFFS)}")
        since = datetime.now(UTC) - _WINDOW_CUTOFFS[window]
        summary = cluster_dynamics_reader.summary(since)
        return _templates.TemplateResponse(
            request, "cluster_dynamics.html",
            {"summary": summary, "window": window, "windows": list(_WINDOW_CUTOFFS)},
        )

    return router
