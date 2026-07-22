"""Postgres-specific storage backend tests.

Backend-agnostic behavior lives in tests/core/test_storage_provider_contract.py.
This file covers only what's genuinely specific to Postgres: the
schema_migrations version-tracking mechanism (SQLite's PRAGMA user_version
equivalent — see core/stores/postgres_migrations.py).
"""

from __future__ import annotations

import pytest

from core.stores.postgres_migrations import POSTGRES_MIGRATIONS, migrate
from tests.conftest import PostgresBackend


@pytest.fixture
def pg_backend(postgres_container):
    backend = PostgresBackend(postgres_container)
    yield backend
    backend.truncate_all()
    backend.close()


class TestSchemaMigrations:
    def test_all_migrations_are_recorded(self, pg_backend: PostgresBackend):
        rows = pg_backend.store._conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [r[0] for r in rows] == list(range(1, max(POSTGRES_MIGRATIONS) + 1))

    def test_migrate_is_idempotent_on_reconnect(self, pg_backend: PostgresBackend):
        """Reconnecting (e.g. a process restart) must not re-apply or
        fail on an already-applied migration."""
        migrate(pg_backend.store._conn, embedding_dim=3)  # must not raise
        row = pg_backend.store._conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()
        assert row[0] == max(POSTGRES_MIGRATIONS)

    def test_vector_extension_is_installed(self, pg_backend: PostgresBackend):
        row = pg_backend.store._conn.execute(
            "SELECT extname FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        assert row is not None
