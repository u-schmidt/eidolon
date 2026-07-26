"""FastAPI app for Eidolon's turn-detail dashboard.

A read-only, server-rendered (Jinja2 + htmx) view over one turn
(``request_id``) at a time — see ``tools/dashboard/turn_reader.py`` for
the data-access layer this only translates into HTML. Route handlers
are free functions rather than class methods, the same established
exception AGENTS.md already carves out for tools/'s and server/app.py's
entrypoint-style glue code.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from tools.dashboard.turn_reader import TurnReader

_DASHBOARD_DIR = Path(__file__).parent
_basic_auth = HTTPBasic(auto_error=False)


def _require_password(credentials: Annotated[HTTPBasicCredentials | None, Depends(_basic_auth)]) -> None:
    """Optional HTTP Basic auth, gated by EIDOLON_DASHBOARD_PASSWORD.

    Skipped entirely if that env var is unset — matches the
    "binds to 127.0.0.1 by default" localhost threat model
    ``tools/serve.py`` already uses for its own default host. HTTP
    Basic (not the deployment-token bearer scheme ``server/auth.py``
    uses) since this is a browser page a human opens directly, not an
    API client sending its own headers.
    """
    expected = os.environ.get("EIDOLON_DASHBOARD_PASSWORD", "")
    if not expected:
        return
    valid = credentials is not None and secrets.compare_digest(credentials.password, expected)
    if not valid:
        raise HTTPException(
            status_code=401, detail="Invalid or missing credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )


def create_dashboard_app(turn_reader: TurnReader) -> FastAPI:
    """Build the FastAPI app wrapping an already-constructed ``TurnReader``."""
    app = FastAPI(title="Eidolon — Turn Detail", dependencies=[Depends(_require_password)])
    app.mount("/static", StaticFiles(directory=str(_DASHBOARD_DIR / "static")), name="static")
    templates = Jinja2Templates(directory=str(_DASHBOARD_DIR / "templates"))

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, peer_id: str | None = None, limit: int = 50) -> HTMLResponse:
        turns = turn_reader.recent_turns(limit=limit, peer_id=peer_id)
        return templates.TemplateResponse(
            request, "index.html", {"turns": turns, "peer_id": peer_id or ""},
        )

    @app.get("/turns/{request_id}", response_class=HTMLResponse)
    def turn_detail(request: Request, request_id: str) -> HTMLResponse:
        detail = turn_reader.get(request_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"No turn found for request_id={request_id!r}")
        return templates.TemplateResponse(
            request, "turn_detail.html", {"detail": detail},
        )

    return app
