# Eidolon core-as-a-service server image.
#
# Runs tools/serve.py — core running as its own standalone HTTP process
# instead of being imported in-process by an adapter (see ROADMAP.md's
# "Core-as-a-service deployment mode" entry). Pairs with
# ../docker-compose.example.yml, which bundles this with a
# pgvector/pgvector:pg15 container for the "server + pgvector"
# deployment preset — the "running this for real" counterpart to the
# default "embedded + sqlite-vec" try-it-in-five-minutes path.
#
# Build from the repo root (the compose file does this automatically):
#   docker build -f deploy/server/Dockerfile -t eidolon-server .

FROM python:3.12-slim

WORKDIR /app

# uv is this project's own dependency manager (see uv.lock) — copied in
# directly from its official image rather than pip-installed, per uv's
# documented Docker pattern.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./
COPY core/ core/
COPY server/ server/
COPY client/ client/
COPY tools/serve.py tools/

# server: fastapi/uvicorn, needed to run tools/serve.py at all.
# postgres: psycopg/pgvector — this image is built for the "server +
# pgvector" preset specifically. LLM/embedder providers are all
# HTTP-based (ollama, lm_studio, openai_compatible, ...) and need no
# extra package beyond the base `requests` dependency; a local
# sentence_transformers embedder would need that extra added separately.
RUN uv sync --extra server --extra postgres --no-dev --no-install-project

EXPOSE 8420

ENTRYPOINT ["uv", "run", "python", "tools/serve.py"]
CMD ["--config", "/app/eidolon.yaml", "--host", "0.0.0.0", "--port", "8420"]
