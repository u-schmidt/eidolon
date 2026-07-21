"""Tests for VecIndex.

Requires the real sqlite-vec extension (skipped entirely if it isn't
installed) — this class is a thin wrapper around real vec0 SQL, not
worth mocking at the sqlite3 layer.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

pytest.importorskip("sqlite_vec")

from core.stores.vec_index import VecIndex  # noqa: E402


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def index(conn) -> VecIndex:
    idx = VecIndex(conn)
    assert idx.try_load() is True
    return idx


class TestTryLoad:
    def test_available_when_extension_loads(self, conn):
        idx = VecIndex(conn)
        assert idx.try_load() is True
        assert idx.available is True

    def test_unavailable_when_not_installed(self, conn):
        idx = VecIndex(conn)
        with patch.dict("sys.modules", {"sqlite_vec": None}):
            assert idx.try_load() is False
        assert idx.available is False

    def test_unavailable_when_extension_load_raises(self, conn):
        idx = VecIndex(conn)
        with patch("sqlite_vec.load", side_effect=Exception("failed to load native extension")):
            assert idx.try_load() is False
        assert idx.available is False

    def test_rediscovers_tables_from_a_prior_run(self, conn):
        first = VecIndex(conn)
        first.try_load()
        first.upsert("vec_topics", 1, "user_a", [1.0, 0.0, 0.0])
        conn.commit()

        second = VecIndex(conn)  # simulates a fresh process on the same DB file
        second.try_load()

        assert second.knn("vec_topics", "user_a", [1.0, 0.0, 0.0], k=5) == [(1, 1.0)]

    def test_rediscovery_ignores_vec0_shadow_tables(self, conn):
        """vec0 creates auxiliary shadow tables (_chunks, _rowids, etc.)
        alongside the real virtual table — rediscovery must not mistake
        those for independently-queryable vec0 tables."""
        first = VecIndex(conn)
        first.try_load()
        first.upsert("vec_topics", 1, "user_a", [1.0, 0.0, 0.0])
        conn.commit()

        second = VecIndex(conn)
        second.try_load()

        assert second._known_tables == {"vec_topics"}


class TestUpsertAndKnn:
    def test_knn_returns_similarity_not_distance(self, index: VecIndex, conn):
        index.upsert("vec_topics", 1, "user_a", [1.0, 0.0, 0.0])
        conn.commit()

        results = index.knn("vec_topics", "user_a", [1.0, 0.0, 0.0], k=5)

        assert results == [(1, pytest.approx(1.0))]  # identical vectors -> similarity 1.0

    def test_knn_orders_nearest_first(self, index: VecIndex, conn):
        index.upsert("vec_topics", 1, "user_a", [1.0, 0.0, 0.0])
        index.upsert("vec_topics", 2, "user_a", [0.0, 1.0, 0.0])  # orthogonal, less similar
        conn.commit()

        results = index.knn("vec_topics", "user_a", [1.0, 0.0, 0.0], k=5)

        assert [row_id for row_id, _ in results] == [1, 2]

    def test_knn_respects_k_limit(self, index: VecIndex, conn):
        for i in range(5):
            index.upsert("vec_topics", i, "user_a", [1.0, 0.0, 0.0])
        conn.commit()

        results = index.knn("vec_topics", "user_a", [1.0, 0.0, 0.0], k=2)

        assert len(results) == 2

    def test_partition_isolation_across_peers(self, index: VecIndex, conn):
        index.upsert("vec_topics", 1, "user_a", [1.0, 0.0, 0.0])
        index.upsert("vec_topics", 2, "user_b", [1.0, 0.0, 0.0])
        conn.commit()

        results = index.knn("vec_topics", "user_a", [1.0, 0.0, 0.0], k=5)

        assert [row_id for row_id, _ in results] == [1]

    def test_upsert_updates_existing_row_in_place(self, index: VecIndex, conn):
        index.upsert("vec_topics", 1, "user_a", [1.0, 0.0, 0.0])
        conn.commit()

        index.upsert("vec_topics", 1, "user_a", [0.0, 1.0, 0.0])
        conn.commit()

        results = index.knn("vec_topics", "user_a", [0.0, 1.0, 0.0], k=5)
        assert results == [(1, pytest.approx(1.0))]

    def test_delete_removes_row(self, index: VecIndex, conn):
        index.upsert("vec_topics", 1, "user_a", [1.0, 0.0, 0.0])
        conn.commit()

        index.delete("vec_topics", 1)
        conn.commit()

        assert index.knn("vec_topics", "user_a", [1.0, 0.0, 0.0], k=5) == []

    def test_delete_on_unknown_table_is_a_noop(self, index: VecIndex):
        index.delete("vec_memory_items", 1)  # must not raise

    def test_knn_on_never_upserted_table_returns_empty(self, index: VecIndex):
        assert index.knn("vec_memory_items", "user_a", [1.0, 0.0, 0.0], k=5) == []

    def test_different_tables_are_independent(self, index: VecIndex, conn):
        index.upsert("vec_topics", 1, "user_a", [1.0, 0.0, 0.0])
        index.upsert("vec_memory_items", 1, "user_a", [1.0, 0.0, 0.0])
        conn.commit()

        # deleting from one table must not affect the other
        index.delete("vec_topics", 1)
        conn.commit()

        assert index.knn("vec_topics", "user_a", [1.0, 0.0, 0.0], k=5) == []
        assert index.knn("vec_memory_items", "user_a", [1.0, 0.0, 0.0], k=5) == [(1, pytest.approx(1.0))]


class TestClusterFiltering:
    def test_unclustered_only_excludes_clustered_rows(self, index: VecIndex, conn):
        index.upsert("vec_topics", 1, "user_a", [1.0, 0.0, 0.0], cluster_id=None)
        index.upsert("vec_topics", 2, "user_a", [1.0, 0.0, 0.0], cluster_id=7)
        conn.commit()

        results = index.knn("vec_topics", "user_a", [1.0, 0.0, 0.0], k=5, unclustered_only=True)

        assert [row_id for row_id, _ in results] == [1]

    def test_without_unclustered_only_includes_everything(self, index: VecIndex, conn):
        index.upsert("vec_topics", 1, "user_a", [1.0, 0.0, 0.0], cluster_id=None)
        index.upsert("vec_topics", 2, "user_a", [1.0, 0.0, 0.0], cluster_id=7)
        conn.commit()

        results = index.knn("vec_topics", "user_a", [1.0, 0.0, 0.0], k=5)

        assert {row_id for row_id, _ in results} == {1, 2}


class TestSetClusterId:
    def test_updates_metadata_without_touching_embedding(self, index: VecIndex, conn):
        index.upsert("vec_topics", 1, "user_a", [1.0, 0.0, 0.0], cluster_id=None)
        conn.commit()

        index.set_cluster_id("vec_topics", 1, cluster_id=7)
        conn.commit()

        # embedding unchanged, no longer shows up as unclustered
        assert index.knn("vec_topics", "user_a", [1.0, 0.0, 0.0], k=5, unclustered_only=True) == []
        results = index.knn("vec_topics", "user_a", [1.0, 0.0, 0.0], k=5)
        assert results == [(1, pytest.approx(1.0))]

    def test_none_resets_to_unclustered_sentinel(self, index: VecIndex, conn):
        index.upsert("vec_topics", 1, "user_a", [1.0, 0.0, 0.0], cluster_id=7)
        conn.commit()

        index.set_cluster_id("vec_topics", 1, cluster_id=None)
        conn.commit()

        results = index.knn("vec_topics", "user_a", [1.0, 0.0, 0.0], k=5, unclustered_only=True)
        assert results == [(1, pytest.approx(1.0))]

    def test_noop_on_unknown_table(self, index: VecIndex):
        index.set_cluster_id("vec_memory_items", 1, cluster_id=7)  # must not raise


class TestDimensionHandling:
    def test_dimension_inferred_from_first_upsert(self, index: VecIndex, conn):
        index.upsert("vec_topics", 1, "user_a", [1.0, 0.0, 0.0, 0.0])  # dim=4
        conn.commit()

        results = index.knn("vec_topics", "user_a", [1.0, 0.0, 0.0, 0.0], k=5)
        assert results == [(1, pytest.approx(1.0))]

    def test_mismatched_dimension_raises(self, index: VecIndex, conn):
        index.upsert("vec_topics", 1, "user_a", [1.0, 0.0, 0.0, 0.0])
        conn.commit()

        with pytest.raises(sqlite3.OperationalError):
            index.upsert("vec_topics", 2, "user_a", [1.0, 0.0])
