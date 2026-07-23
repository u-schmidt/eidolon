"""Bearer-token auth dependency for the Eidolon HTTP server.

Checked against ``EIDOLON_DEPLOYMENT_TOKEN`` (env var, not YAML — same
precedent as ``EIDOLON_STORAGE_PASSWORD``, see core/config.py). A
network-exposed core needs an auth boundary an in-process call never
did; see ROADMAP.md's "Core-as-a-service deployment mode" entry.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException


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
