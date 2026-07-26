"""Postgres-specific storage backend tests.

Backend-agnostic behavior lives in tests/core/test_storage_provider_contract.py.
This file covers only what's genuinely specific to Postgres: the
schema_migrations version-tracking mechanism (SQLite's PRAGMA user_version
equivalent — see core/stores/postgres_migrations.py), and connection
transaction hygiene (SQLite has no equivalent concept — it's
autocommit-per-statement at the OS-file-lock level).
"""

from __future__ import annotations

import psycopg
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


class TestTransactionHygiene:
    """A real bug, found live: PostgresStore.connect() used to leave the
    connection in manual-commit mode (autocommit=False) for its entire
    lifetime, and every read-only method (unlike the write methods,
    which all called .commit()) never closed out its own implicit
    transaction — so a long-lived connection sat "idle in transaction"
    indefinitely after any read, observed live blocking a later schema
    migration that needed an ACCESS EXCLUSIVE lock on `messages`. Fixed
    by flipping to autocommit=True once connect()'s own setup (which
    still needs its migration statements to commit together) is done —
    see PostgresStore.connect()'s docstring.

    Checked from a *second*, independent connection: asking
    pg_stat_activity about your own backend's state always reports
    "active" (you're mid-query, asking the question) — the whole point
    is observing the store's connection from outside it.
    """

    @staticmethod
    def _state_for_pid(observer: psycopg.Connection, pid: int) -> str | None:
        row = observer.execute(
            "SELECT state FROM pg_stat_activity WHERE pid = %s", (pid,)
        ).fetchone()
        return row[0] if row else None

    @staticmethod
    def _own_pid(store) -> int:
        return store._conn.execute("SELECT pg_backend_pid()").fetchone()[0]

    def test_read_only_call_leaves_no_open_transaction(
        self, pg_backend: PostgresBackend, postgres_container,
    ):
        store = pg_backend.store
        pid = self._own_pid(store)

        store.get_topic(999999)  # any read-only call; result irrelevant

        with psycopg.connect(
            host=postgres_container.get_container_host_ip(),
            port=int(postgres_container.get_exposed_port(5432)),
            dbname=postgres_container.dbname,
            user=postgres_container.username,
            password=postgres_container.password,
        ) as observer:
            assert self._state_for_pid(observer, pid) != "idle in transaction"

    def test_write_call_leaves_no_open_transaction(
        self, pg_backend: PostgresBackend, postgres_container,
    ):
        store = pg_backend.store
        pid = self._own_pid(store)

        store.create_topic("Test topic", peer_id="user")

        with psycopg.connect(
            host=postgres_container.get_container_host_ip(),
            port=int(postgres_container.get_exposed_port(5432)),
            dbname=postgres_container.dbname,
            user=postgres_container.username,
            password=postgres_container.password,
        ) as observer:
            assert self._state_for_pid(observer, pid) != "idle in transaction"

    def test_assign_topic_to_cluster_commits_both_statements(self, pg_backend: PostgresBackend):
        """Multi-statement write under autocommit — see
        assign_topic_to_cluster's docstring for why it needs an explicit
        ``with conn.transaction()``. Confirms both UPDATEs actually took
        effect together, not just that no exception was raised."""
        store = pg_backend.store
        tid = store.create_topic("Gym talk", peer_id="user")
        cid = store.create_cluster("user", "Fitness", "summary")

        store.assign_topic_to_cluster(tid, cid)

        assert store.get_topic(tid)["id"] == tid  # sanity: topic still there
        topic_row = store._conn.execute(
            "SELECT cluster_id FROM topics WHERE id = %s", (tid,)
        ).fetchone()
        assert topic_row[0] == cid
        cluster_row = store._conn.execute(
            "SELECT topic_count FROM topic_clusters WHERE id = %s", (cid,)
        ).fetchone()
        assert cluster_row[0] == 1
