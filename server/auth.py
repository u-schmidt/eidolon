"""Auth dependencies for the Eidolon HTTP server.

Two independent schemes for two independently-mounted routers on the
same app (see server/app.py's create_app): ``require_token`` gates the
machine RPC API (bearer token, checked against
``EIDOLON_DEPLOYMENT_TOKEN`` — env var, not YAML, same precedent as
``EIDOLON_STORAGE_PASSWORD``, see core/config.py); ``require_dashboard_password``
gates the human-facing turn-detail dashboard (HTTP Basic — a browser
can prompt for that natively, unlike a bearer header). A
network-exposed core needs an auth boundary an in-process call never
did; see ROADMAP.md's "Core-as-a-service deployment mode" entry.
"""

from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials


def require_token(authorization: str = Header(default="")) -> None:
    """FastAPI dependency: reject any request without a matching bearer token.

    Fails closed on server misconfiguration — if EIDOLON_DEPLOYMENT_TOKEN
    isn't set, every request is rejected (500) rather than silently
    accepted, matching this project's existing "raise loudly on missing
    required config" precedent (see ConfigError in core/config.py).
    """
    expected = os.environ.get("EIDOLON_DEPLOYMENT_TOKEN", "")
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="Server misconfigured: EIDOLON_DEPLOYMENT_TOKEN is not set.",
        )
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")


_dashboard_basic_auth = HTTPBasic(auto_error=False)


def require_dashboard_password(
    credentials: Annotated[HTTPBasicCredentials | None, Depends(_dashboard_basic_auth)],
) -> None:
    """FastAPI dependency: optional HTTP Basic auth for the dashboard router.

    Skipped entirely if EIDOLON_DASHBOARD_PASSWORD is unset — matches
    the "binds to 127.0.0.1 by default" localhost threat model this
    server already uses for its own default host, fine for a page
    that's typically reached over SSH tunnel or directly on the host.
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
