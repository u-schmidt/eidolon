"""Shared pytest fixtures.

``store_backend`` is parametrized over both ``StorageProvider``
implementations (sqlite, postgres) so the same contract tests run
against both — see tests/core/test_storage_provider_contract.py. This
is the "integration tests only when necessary (database, API,
filesystem)" case AGENTS.md carves out: there's no way to verify
PostgresStore's actual SQL/pgvector behavior without a real Postgres.

The Postgres side needs Docker. testcontainers spins up one ephemeral
``pgvector/pgvector:pg15`` container per test *session* (not per test —
container startup dominates runtime otherwise), truncating all tables
between tests instead for isolation. sqlite-parametrized tests never
touch Docker at all — the container is only requested lazily, inside
the postgres branch below, so a sqlite-only test run has zero Docker
dependency.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Testcontainers' "Ryuk" resource-reaper sidecar container hangs
# indefinitely trying to start in some sandboxed/restricted Docker
# setups (confirmed in this project's dev environment) — we manage
# container lifecycle explicitly via fixture teardown anyway, so the
# reaper isn't needed. Must be set before testcontainers is imported.
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")

from testcontainers.postgres import PostgresContainer  # noqa: E402

from core.stores.postgres_store import PostgresStore  # noqa: E402
from core.stores.sqlite_store import SqliteStore  # noqa: E402


@pytest.fixture(scope="session")
def postgres_container():
    with PostgresContainer("pgvector/pgvector:pg15", username="eidolon", password="test", dbname="eidolon") as pg:
        yield pg


class SqliteBackend:
    """A connected SqliteStore plus a way to backdate a row's timestamp
    column — test setup for recency/staleness scenarios needs this, and
    each backend stores timestamps differently (naive UTC strings here)."""

    def __init__(self) -> None:
        tmp = tempfile.mkdtemp()
        self.store = SqliteStore(str(Path(tmp) / "test.db"), use_vec_index=False)
        self.store.connect()

    def backdate(self, table: str, row_id: int, column: str, when: datetime) -> None:
        formatted = when.strftime("%Y-%m-%d %H:%M:%S")
        self.store._conn.execute(f"UPDATE {table} SET {column} = ? WHERE id = ?", (formatted, row_id))
        self.store._conn.commit()

    def close(self) -> None:
        self.store.close()


class PostgresBackend:
    """A connected PostgresStore plus the same backdate() interface —
    real TIMESTAMPTZ columns accept a datetime directly, no formatting."""

    def __init__(self, container: PostgresContainer) -> None:
        self.store = PostgresStore(
            host=container.get_container_host_ip(),
            port=int(container.get_exposed_port(5432)),
            dbname=container.dbname,
            user=container.username,
            password=container.password,
            # Matches the dimensionality of the embedding vectors used
            # throughout the contract test suite (e.g. [1.0, 0.0, 0.0]) —
            # pgvector's VECTOR(n) column width is fixed, unlike SQLite's
            # dimension-agnostic BLOB storage.
            embedding_dim=3,
        )
        self.store.connect()

    def backdate(self, table: str, row_id: int, column: str, when: datetime) -> None:
        self.store._conn.execute(f"UPDATE {table} SET {column} = %s WHERE id = %s", (when, row_id))
        self.store._conn.commit()

    def truncate_all(self) -> None:
        with self.store._conn.cursor() as cur:
            cur.execute(
                "TRUNCATE topics, memory_items, messages, facts, "
                "topic_clusters, fact_consolidation_state RESTART IDENTITY CASCADE"
            )
        self.store._conn.commit()

    def close(self) -> None:
        self.store.close()


@pytest.fixture(params=["sqlite", "postgres"])
def store_backend(request):
    if request.param == "sqlite":
        backend = SqliteBackend()
        yield backend
        backend.close()
    else:
        container = request.getfixturevalue("postgres_container")
        backend = PostgresBackend(container)
        yield backend
        backend.truncate_all()
        backend.close()


@pytest.fixture
def store(store_backend: SqliteBackend | PostgresBackend):
    """The connected StorageProvider itself — for the majority of
    contract tests that don't need store_backend.backdate()."""
    return store_backend.store


@pytest.fixture(params=["embedded", "http"])
def make_eidolon_client(request):
    """Factory fixture: hand it an already-built ``Eidolon`` (backed by
    whatever storage/llm/embedder doubles a test needs) and get back an
    ``EidolonClient`` — either that same instance directly ("embedded"),
    or an ``EidolonHttpClient`` talking to it through a real FastAPI app
    via ``TestClient`` (in-process, no real sockets — "http"). Same
    underlying ``Eidolon`` either way, so a behavior difference between
    the two branches would mean the HTTP transport lost or mangled
    something — see tests/core/test_eidolon_client_contract.py.

    A factory rather than a fixed value (unlike ``store`` above) because
    each test typically needs its own canned LLM responses (NEW_TOPIC
    vs. CONTINUE, etc.), which the caller controls by constructing its
    own ``Eidolon`` before handing it in.
    """
    def _make(eidolon):
        if request.param == "embedded":
            return eidolon

        from fastapi.testclient import TestClient

        from client.http_client import EidolonHttpClient
        from server.app import create_app
        from server.auth import require_token

        app = create_app(eidolon)
        # Auth itself is already covered by server/'s own smoke tests —
        # not what this suite is checking — so it's bypassed here rather
        # than threading a token through every contract test.
        app.dependency_overrides[require_token] = lambda: None
        test_client = TestClient(app)
        # base_url="" — TestClient already knows its own base URL
        # internally and expects bare paths, unlike a real requests.Session.
        return EidolonHttpClient(base_url="", token="unused", timeout=5.0, session=test_client)

    return _make
