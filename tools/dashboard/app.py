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

from tools.dashboard.browse_reader import BrowseReader
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


def _resolve_window(window: str) -> datetime:
    """Validate a ``window`` query param against ``_WINDOW_CUTOFFS`` and
    return the corresponding ``since`` cutoff, or raise 400 — shared by
    every insight route so the same 3-line check isn't repeated per
    route."""
    if window not in _WINDOW_CUTOFFS:
        raise HTTPException(status_code=400, detail=f"Unknown window {window!r} — use one of {list(_WINDOW_CUTOFFS)}")
    return datetime.now(UTC) - _WINDOW_CUTOFFS[window]


def create_dashboard_router(
    turn_reader: TurnReader,
    performance_reader: PerformanceReader,
    retrieval_quality_reader: RetrievalQualityReader,
    prompt_quality_reader: PromptQualityReader,
    topic_lifecycle_reader: TopicLifecycleReader,
    cluster_dynamics_reader: ClusterDynamicsReader,
    browse_reader: BrowseReader,
) -> APIRouter:
    """Build the dashboard's routes wrapping already-constructed readers."""
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse, name="dashboard_index")
    def landing(request: Request) -> HTMLResponse:
        recent = turn_reader.recent_turns(limit=5)
        counts = browse_reader.overview_counts()
        # Fixed 24h window rather than a selector — this card is a "look
        # now, something might be wrong" nudge onto Prompt Quality, not
        # another window-scoped insight view. total_topic_detection_calls
        # fires on every turn, so ==0 means "no debug data this window",
        # not "zero errors" — same signal prompt_quality.html itself uses.
        quality = prompt_quality_reader.summary(datetime.now(UTC) - timedelta(hours=24))
        return _templates.TemplateResponse(
            request, "index.html",
            {
                "recent_turns": recent, "counts": counts,
                "fallback_continue_count": quality.continue_fallback.fallback_continue,
                "has_quality_debug_data": quality.total_topic_detection_calls > 0,
            },
        )

    @router.get("/turns", response_class=HTMLResponse, name="dashboard_turns_list")
    def turns_list(request: Request, workspace_id: str | None = None, limit: int = 50) -> HTMLResponse:
        turns = turn_reader.recent_turns(limit=limit, workspace_id=workspace_id)
        return _templates.TemplateResponse(
            request, "turns.html", {"turns": turns, "workspace_id": workspace_id or ""},
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
        since = _resolve_window(window)
        summary = performance_reader.summary(since)
        return _templates.TemplateResponse(
            request, "performance.html",
            {"summary": summary, "window": window, "windows": list(_WINDOW_CUTOFFS)},
        )

    @router.get("/retrieval-quality", response_class=HTMLResponse, name="dashboard_retrieval_quality")
    def retrieval_quality(
        request: Request, window: str = _DEFAULT_WINDOW, near_miss_kind: str | None = None,
    ) -> HTMLResponse:
        since = _resolve_window(window)
        near_miss_kind = near_miss_kind or None
        summary = retrieval_quality_reader.summary(since, near_miss_kind=near_miss_kind)
        return _templates.TemplateResponse(
            request, "retrieval_quality.html",
            {
                "summary": summary, "window": window, "windows": list(_WINDOW_CUTOFFS),
                "near_miss_kind": near_miss_kind or "",
            },
        )

    @router.get("/prompt-quality", response_class=HTMLResponse, name="dashboard_prompt_quality")
    def prompt_quality(request: Request, window: str = _DEFAULT_WINDOW) -> HTMLResponse:
        since = _resolve_window(window)
        summary = prompt_quality_reader.summary(since)
        return _templates.TemplateResponse(
            request, "prompt_quality.html",
            {"summary": summary, "window": window, "windows": list(_WINDOW_CUTOFFS)},
        )

    @router.get("/topic-lifecycle", response_class=HTMLResponse, name="dashboard_topic_lifecycle")
    def topic_lifecycle(request: Request, window: str = _DEFAULT_WINDOW) -> HTMLResponse:
        since = _resolve_window(window)
        summary = topic_lifecycle_reader.summary(since)
        return _templates.TemplateResponse(
            request, "topic_lifecycle.html",
            {"summary": summary, "window": window, "windows": list(_WINDOW_CUTOFFS)},
        )

    @router.get("/cluster-dynamics", response_class=HTMLResponse, name="dashboard_cluster_dynamics")
    def cluster_dynamics(request: Request, window: str = _DEFAULT_WINDOW) -> HTMLResponse:
        since = _resolve_window(window)
        summary = cluster_dynamics_reader.summary(since)
        return _templates.TemplateResponse(
            request, "cluster_dynamics.html",
            {"summary": summary, "window": window, "windows": list(_WINDOW_CUTOFFS)},
        )

    @router.get("/topics", response_class=HTMLResponse, name="dashboard_topics")
    def topics(
        request: Request, page: int = 1, status: str | None = None, workspace_id: str | None = None,
    ) -> HTMLResponse:
        # Empty-string query params (a submitted-but-blank filter field, or
        # a pagination link carrying the previous filter state) must mean
        # "no filter", same as an absent param — not a literal "" match.
        status = status or None
        workspace_id = workspace_id or None
        result = browse_reader.topics(page, status=status, workspace_id=workspace_id)
        return _templates.TemplateResponse(
            request, "browse_topics.html",
            {"page": result, "status": status or "", "workspace_id": workspace_id or ""},
        )

    @router.get("/facts", response_class=HTMLResponse, name="dashboard_facts")
    def facts(
        request: Request, page: int = 1, category: str | None = None, workspace_id: str | None = None,
        sort: str | None = None,
    ) -> HTMLResponse:
        category = category or None
        workspace_id = workspace_id or None
        sort = sort or None
        result = browse_reader.facts(page, category=category, workspace_id=workspace_id, sort_by=sort)
        return _templates.TemplateResponse(
            request, "browse_facts.html",
            {"page": result, "category": category or "", "workspace_id": workspace_id or "", "sort": sort or ""},
        )

    @router.get("/memory-items", response_class=HTMLResponse, name="dashboard_memory_items")
    def memory_items(
        request: Request, page: int = 1, item_type: str | None = None, topic_id: str | None = None,
        q: str | None = None,
    ) -> HTMLResponse:
        item_type = item_type or None
        q = q or None
        try:
            topic_id_int = int(topic_id) if topic_id else None
        except ValueError:
            # A non-numeric topic_id filter (e.g. stray text in the input
            # box) means "no valid filter", not a 500 — same treatment as
            # any other unmatched/blank filter on this page.
            topic_id_int = None
        result = browse_reader.memory_items(page, item_type=item_type, topic_id=topic_id_int, q=q)
        return _templates.TemplateResponse(
            request, "browse_memory_items.html",
            {"page": result, "item_type": item_type or "", "topic_id": topic_id_int, "q": q or ""},
        )

    @router.get("/clusters", response_class=HTMLResponse, name="dashboard_clusters")
    def clusters(request: Request, page: int = 1, workspace_id: str | None = None) -> HTMLResponse:
        workspace_id = workspace_id or None
        result = browse_reader.clusters(page, workspace_id=workspace_id)
        return _templates.TemplateResponse(
            request, "browse_clusters.html",
            {"page": result, "workspace_id": workspace_id or ""},
        )

    @router.get("/clusters/{cluster_id}", response_class=HTMLResponse, name="dashboard_cluster_detail")
    def cluster_detail(request: Request, cluster_id: int) -> HTMLResponse:
        detail = browse_reader.cluster_detail(cluster_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"No cluster found for id={cluster_id!r}")
        return _templates.TemplateResponse(
            request, "cluster_detail.html", {"detail": detail},
        )

    return router
