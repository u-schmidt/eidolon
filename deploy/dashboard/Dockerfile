# Eidolon turn-detail dashboard image.
#
# Runs tools/dashboard.py — a read-only web view over one turn at a
# time (see README.md's "Turn Detail Dashboard" section). Connects
# directly to the same storage a `server` deployment's eidolon-server
# uses (same EIDOLON_STORAGE_* env vars), independent of
# deployment.mode — it never goes through the EidolonClient RPC API.
#
# Build from the repo root:
#   docker build -f deploy/dashboard/Dockerfile -t eidolon-dashboard .

FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./
COPY core/ core/
COPY tools/dashboard.py tools/
COPY tools/dashboard/ tools/dashboard/

# dashboard: fastapi/uvicorn/jinja2, needed to run tools/dashboard.py.
# postgres: psycopg/pgvector — same "server + pgvector" preset as
# deploy/server/Dockerfile.
RUN uv sync --extra dashboard --extra postgres --no-dev --no-install-project

EXPOSE 8421

ENTRYPOINT ["uv", "run", "python", "tools/dashboard.py"]
CMD ["--config", "/app/eidolon.yaml", "--host", "0.0.0.0", "--port", "8421"]
