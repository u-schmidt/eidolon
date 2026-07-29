"""SQLite-specific storage backend tests.

Backend-agnostic behavior lives in tests/core/test_storage_provider_contract.py
(runs against both SqliteStore and PostgresStore). This file covers only
what's genuinely specific to SQLite: PRAGMA/migration mechanics and the
optional sqlite-vec accelerated search path (pgvector has no equivalent
"index unavailable, fall back to brute force" duality — see
core/stores/postgres_store.py's module docstring).
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from core.models import MemoryItem, MemoryItemBudget
from core.stores.sqlite_store import SqliteStore


@pytest.fixture
def store():
    """Create a fresh SqliteStore in a temp directory for each test.

    use_vec_index=False so these tests deterministically exercise the
    brute-force scan regardless of whether sqlite-vec happens to be
    installed in the environment — the accelerated path has its own
    dedicated tests (TestVecIndexAcceleratedSearch below).
    """
    tmp = tempfile.mkdtemp()
    db_path = str(Path(tmp) / "test.db")
    s = SqliteStore(db_path, use_vec_index=False)
    s.connect()
    yield s
    s.close()


class TestSchema:
    def test_expected_indexes_exist(self, store: SqliteStore):
        names = {
            row[0]
            for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        assert {
            "idx_messages_topic",
            "idx_messages_workspace_role",
            "idx_topics_workspace_status",
            "idx_memory_items_subject_type",
        } <= names

    def test_user_version_is_current(self, store: SqliteStore):
        from core.stores.migrations import MIGRATIONS

        version = store._conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == max(MIGRATIONS)


class TestMigrationAtomicity:
    """A failing statement partway through one version's script must roll
    back that version's other statements too, not leave them durably
    committed while user_version stays behind — see migrate()'s docstring
    for how executescript's lack of implicit transaction control made
    that possible before this was fixed."""

    def test_failed_migration_rolls_back_ddl_and_version(self, monkeypatch, tmp_path):
        import core.stores.migrations as migrations_module

        db_path = str(tmp_path / "atomicity.db")
        db = sqlite3.connect(db_path)

        # Bring the DB to a known-good state at the real latest version.
        migrations_module.migrate(db)
        start_version = db.execute("PRAGMA user_version").fetchone()[0]

        broken_version = start_version + 1
        monkeypatch.setitem(
            migrations_module.MIGRATIONS,
            broken_version,
            "CREATE TABLE atomicity_probe (id INTEGER);\n"
            "ALTER TABLE this_table_does_not_exist ADD COLUMN x INTEGER;\n",
        )

        with pytest.raises(sqlite3.Error):
            migrations_module.migrate(db)

        db.close()

        # Fresh connection — only durable/committed state is visible here.
        verify = sqlite3.connect(db_path)
        assert verify.execute("PRAGMA user_version").fetchone()[0] == start_version
        assert (
            verify.execute(
                "SELECT name FROM sqlite_master WHERE name = 'atomicity_probe'"
            ).fetchall()
            == []
        )
        verify.close()

    def test_successful_migration_still_commits_ddl_and_version(self, monkeypatch, tmp_path):
        import core.stores.migrations as migrations_module

        db_path = str(tmp_path / "atomicity_ok.db")
        db = sqlite3.connect(db_path)
        migrations_module.migrate(db)
        start_version = db.execute("PRAGMA user_version").fetchone()[0]

        new_version = start_version + 1
        monkeypatch.setitem(
            migrations_module.MIGRATIONS, new_version, "CREATE TABLE atomicity_probe_ok (id INTEGER);"
        )
        migrations_module.migrate(db)
        db.close()

        verify = sqlite3.connect(db_path)
        assert verify.execute("PRAGMA user_version").fetchone()[0] == new_version
        assert verify.execute(
            "SELECT name FROM sqlite_master WHERE name = 'atomicity_probe_ok'"
        ).fetchall() != []
        verify.close()

    def test_active_topic_query_uses_index(self, store: SqliteStore):
        """The per-turn get_active_topic lookup must be an index seek, not
        a full table scan — regression guard for the topics index."""
        store.create_topic("Ongoing", workspace_id="user", session_id="web")
        plan = store._conn.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT id FROM topics WHERE workspace_id = ? AND session_id IS ? "
            "AND status = 'active' ORDER BY created_at DESC, id DESC LIMIT 1",
            ("user", "web"),
        ).fetchall()
        detail = " ".join(str(row[3]) for row in plan)
        assert "idx_topics_workspace_status" in detail
        assert "SCAN" not in detail or "USING INDEX" in detail


class TestVecIndexAcceleratedSearch:
    """Parallel coverage for the sqlite-vec-accelerated search path —
    proves it returns results equivalent to (not just different from)
    the brute-force scan, and exercises what's unique to it (workspace
    partitioning via the vector index itself, backfill on connect()).
    Skipped entirely if sqlite-vec isn't installed.
    """

    @pytest.fixture
    def accel_store(self):
        pytest.importorskip("sqlite_vec")
        tmp = tempfile.mkdtemp()
        s = SqliteStore(str(Path(tmp) / "test.db"))  # use_vec_index defaults True
        s.connect()
        assert s.vec_index.available, "sqlite-vec installed but failed to load in this environment"
        yield s
        s.close()

    def test_search_topics_accelerated_path_is_workspace_isolated(self, accel_store: SqliteStore):
        mine = accel_store.create_topic("Mine", summary="", workspace_id="user", embedding=[1.0, 0.0, 0.0])
        accel_store.close_topic(mine, "closed")
        theirs = accel_store.create_topic("Theirs", summary="", workspace_id="someone_else", embedding=[1.0, 0.0, 0.0])
        accel_store.close_topic(theirs, "closed")

        results = accel_store.search_topics(query_embedding=[1.0, 0.0, 0.0], workspace_id="user")

        assert [r["id"] for r in results] == [mine]

    def test_search_topics_accelerated_matches_brute_force_ranking(self, accel_store: SqliteStore):
        near = accel_store.create_topic("Near", summary="", workspace_id="user", embedding=[1.0, 0.0, 0.0])
        accel_store.close_topic(near, "closed")
        far = accel_store.create_topic("Far", summary="", workspace_id="user", embedding=[0.0, 1.0, 0.0])
        accel_store.close_topic(far, "closed")

        results = accel_store.search_topics(query_embedding=[0.9, 0.1, 0.0], workspace_id="user")

        assert [r["id"] for r in results] == [near, far]

    def test_search_memory_items_accelerated_path_finds_real_workspace_id(self, accel_store: SqliteStore):
        msg_id = accel_store.store_message(turn_id=1, role="user", content="hi", workspace_id="telegram:12345")
        item_id = accel_store.store_memory_item(MemoryItem(
            content="about a trip", item_type="essence", source_peer="user", subject_peer="user",
            attribution_type="self_statement", message_id=msg_id, embedding=[1.0, 0.0, 0.0],
        ))

        results = accel_store.search_memory_items(workspace_id="telegram:12345", query_embedding=[1.0, 0.0, 0.0])

        assert [item.id for item in results] == [item_id]

    def test_search_memory_items_rare_type_survives_knn_prefilter(self, accel_store: SqliteStore):
        """The KNN prefilter ranks by embedding similarity only, with no
        item_type awareness — a reserved surprise/feeling slot could
        otherwise go unfillable if the type's only candidate isn't
        semantically close enough to survive the prefilter cutoff. A
        small budget (total=1) keeps _knn_prefilter_k at its floor (50),
        so 55 much-closer fact candidates reliably push the far surprise
        item outside a naive top-50 scan — it must still come back via
        the dedicated bypass fetch."""
        msg_id = accel_store.store_message(turn_id=1, role="user", content="hi", workspace_id="user")
        for i in range(55):
            accel_store.store_memory_item(MemoryItem(
                content=f"near fact {i}", item_type="fact", source_peer="user", subject_peer="user",
                attribution_type="self_statement", message_id=msg_id, embedding=[1.0, 0.0, 0.0],
            ))
        surprise_id = accel_store.store_memory_item(MemoryItem(
            content="far surprise", item_type="surprise", source_peer="user", subject_peer="user",
            attribution_type="self_statement", message_id=msg_id, embedding=[0.05, 1.0, 0.0],
        ))

        results = accel_store.search_memory_items(
            workspace_id="user", query_embedding=[1.0, 0.0, 0.0],
            budget=MemoryItemBudget(fact=0, surprise=1, feeling=0, free=0),
        )

        assert [item.id for item in results] == [surprise_id]

    def test_search_clusters_accelerated_path_is_workspace_isolated(self, accel_store: SqliteStore):
        mine = accel_store.create_cluster("user", "Mine", "summary", embedding=[1.0, 0.0, 0.0])
        accel_store.create_cluster("someone_else", "Theirs", "summary", embedding=[1.0, 0.0, 0.0])

        results = accel_store.search_clusters(query_embedding=[1.0, 0.0, 0.0], workspace_id="user")

        assert [r["id"] for r in results] == [mine]

    def test_search_cluster_candidates_accelerated_path_excludes_self_and_clustered(self, accel_store: SqliteStore):
        new_topic = accel_store.create_topic("New", summary="", workspace_id="user", embedding=[1.0, 0.0, 0.0])
        accel_store.close_topic(new_topic, "closed")
        other_topic = accel_store.create_topic("Other", summary="", workspace_id="user", embedding=[1.0, 0.0, 0.0])
        accel_store.close_topic(other_topic, "closed")
        clustered_topic = accel_store.create_topic("Clustered", summary="", workspace_id="user", embedding=[1.0, 0.0, 0.0])
        accel_store.close_topic(clustered_topic, "closed")
        cid = accel_store.create_cluster("user", "Existing cluster", "summary", embedding=[1.0, 0.0, 0.0])
        accel_store.assign_topic_to_cluster(clustered_topic, cid)

        candidates = accel_store.search_cluster_candidates(new_topic, "user")

        kinds = {(c["candidate_kind"], c["id"]) for c in candidates}
        assert kinds == {("topic", other_topic), ("cluster", cid)}

    def test_write_paths_populate_the_index_immediately(self, accel_store: SqliteStore):
        """A topic created through create_topic (not a manual upsert) must
        be findable through the accelerated path right away — proves the
        write-path wiring, not just VecIndex in isolation."""
        tid = accel_store.create_topic("Bike", summary="", workspace_id="user", embedding=[1.0, 0.0, 0.0])
        accel_store.close_topic(tid, "closed")

        assert accel_store.vec_index.knn("vec_topics", "user", [1.0, 0.0, 0.0], k=5) == [
            (tid, pytest.approx(1.0))
        ]

    def test_backfill_on_connect_indexes_pre_existing_embeddings(self):
        """Data written while the index was unavailable must become
        searchable once sqlite-vec loads later — not silently invisible
        forever. See SqliteStore._backfill_vec_index."""
        pytest.importorskip("sqlite_vec")
        tmp = tempfile.mkdtemp()
        db_path = str(Path(tmp) / "test.db")

        without_index = SqliteStore(db_path, use_vec_index=False)
        without_index.connect()
        tid = without_index.create_topic("Bike", summary="", workspace_id="user", embedding=[1.0, 0.0, 0.0])
        without_index.close_topic(tid, "closed")
        without_index.close()

        with_index = SqliteStore(db_path)  # use_vec_index defaults True
        with_index.connect()
        try:
            assert with_index.vec_index.available
            hits = with_index.vec_index.knn("vec_topics", "user", [1.0, 0.0, 0.0], k=5)
            assert hits == [(tid, pytest.approx(1.0))]
        finally:
            with_index.close()

    def test_backfill_does_not_leave_an_uncommitted_transaction(self):
        """Real deployment failure: _backfill_vec_index's DELETE+INSERT
        statements open an implicit transaction; if connect() returns
        without committing it, a second, independent connection to the
        same file hits 'database is locked' immediately (SQLite's
        default busy_timeout is 0, no retry) — this is exactly what
        happened when EidolonMemoryProvider.initialize() raced a second
        connection against a real deployed DB. A second connection must
        be able to write right after connect() returns."""
        pytest.importorskip("sqlite_vec")
        tmp = tempfile.mkdtemp()
        db_path = str(Path(tmp) / "test.db")

        seed = SqliteStore(db_path, use_vec_index=False)
        seed.connect()
        seed.create_topic("Bike", summary="", workspace_id="user", embedding=[1.0, 0.0, 0.0])
        seed.close()

        store = SqliteStore(db_path)  # use_vec_index defaults True, triggers a real backfill
        store.connect()
        try:
            second = sqlite3.connect(db_path, timeout=0)
            try:
                second.execute(
                    "INSERT INTO messages (turn_id, role, content, workspace_id) VALUES (?, ?, ?, ?)",
                    (1, "user", "hello", "user"),
                )
                second.commit()
            finally:
                second.close()
        finally:
            store.close()
