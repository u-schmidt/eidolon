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

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from tools.dashboard.turn_reader import TurnReader

_DASHBOARD_DIR = Path(__file__).parent
DASHBOARD_STATIC_DIR = _DASHBOARD_DIR / "static"
_templates = Jinja2Templates(directory=str(_DASHBOARD_DIR / "templates"))


def create_dashboard_router(turn_reader: TurnReader) -> APIRouter:
    """Build the dashboard's routes wrapping an already-constructed ``TurnReader``."""
    router = APIRouter()

    @router.get("/", response_class=HTMLResponse, name="dashboard_index")
    def index(request: Request, peer_id: str | None = None, limit: int = 50) -> HTMLResponse:
        turns = turn_reader.recent_turns(limit=limit, peer_id=peer_id)
        return _templates.TemplateResponse(
            request, "index.html", {"turns": turns, "peer_id": peer_id or ""},
        )

    @router.get("/turns/{request_id}", response_class=HTMLResponse, name="dashboard_turn_detail")
    def turn_detail(request: Request, request_id: str) -> HTMLResponse:
        detail = turn_reader.get(request_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"No turn found for request_id={request_id!r}")
        return _templates.TemplateResponse(
            request, "turn_detail.html", {"detail": detail},
        )

    return router
