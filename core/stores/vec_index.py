"""sqlite-vec-backed vector index for Eidolon.

Optional acceleration layer over SqliteStore's brute-force in-Python
cosine scan — a workspace-partitioned KNN pre-filter, not a replacement
for the existing weighted scoring (semantic + tag_overlap + temporal),
which still runs over whatever candidates come back from here.

Falls back gracefully: if the sqlite-vec extension isn't installed, or
the host SQLite build doesn't support loadable extensions (a real,
observed gap on stock/python.org-built Python on macOS), ``available``
stays False and callers use SqliteStore's existing brute-force path
instead — not new compatibility code, the same implementation already
in service today.
"""

from __future__ import annotations

import logging
import sqlite3
import struct

logger = logging.getLogger(__name__)

# vec0's INTEGER metadata columns reject NULL outright — this sentinel
# stands in for "not assigned to a cluster" (real cluster ids are
# AUTOINCREMENT, starting at 1, so 0 is unambiguous).
_UNCLUSTERED_SENTINEL = 0

# The vec0 partition-key column's name before the peer_id -> workspace_id
# rename (core/stores/migrations.py's version 11) — used only to detect
# and drop pre-rename tables in try_load(). Deliberately built from two
# concatenated pieces, not one literal "peer_id" string: a plain literal
# here would itself get caught by the next find-and-replace of "peer_id"
# across the codebase, silently turning this into a match against the
# *current* schema instead of the old one (this happened once already —
# see the comment at its use site).
_OLD_PARTITION_KEY_MARKER = "pe" + "er_id TEXT partition key"


def _serialize_vec(vec: list[float]) -> bytes:
    """Pack a float list into vec0's expected binary format (little-endian f32)."""
    return struct.pack(f"{len(vec)}f", *vec)


class VecIndex:
    """Wraps sqlite-vec's ``vec0`` virtual tables for fast,
    workspace-partitioned cosine KNN search.

    Tables are created lazily, one per table name, on the first
    ``upsert()`` for that name — vec0 bakes the embedding dimension into
    ``CREATE VIRTUAL TABLE``, and there's otherwise no reason for this
    class (or ``SqliteStore``) to know the embedder's configured
    dimension ahead of time.

    Parameters
    ----------
    conn : sqlite3.Connection
        The same connection ``SqliteStore`` uses — the vec0 virtual
        tables live in the same database file, alongside the relational
        tables that remain the source of truth for everything except
        the vector itself.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.available = False
        self._known_tables: set[str] = set()

    def try_load(self) -> bool:
        """Attempt to load the sqlite-vec extension and discover any
        vec0 tables already present from a prior run. Sets and returns
        ``available``. Never raises — a failure here just means the
        brute-force fallback stays in use."""
        try:
            import sqlite_vec
        except ImportError:
            logger.warning(
                "sqlite-vec not installed — falling back to the brute-force "
                "in-Python similarity scan. Install the 'vec' extra "
                "(pip install eidolon[vec]) for faster retrieval at scale."
            )
            self.available = False
            return False

        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        except Exception as exc:
            logger.warning(
                "sqlite-vec is installed but the extension failed to load "
                "(%r) — falling back to the brute-force in-Python "
                "similarity scan. This usually means the host Python's "
                "SQLite build doesn't support loadable extensions.",
                exc,
            )
            self.available = False
            return False

        self.available = True
        # Rediscover tables created by a prior process/run — _known_tables
        # is otherwise just in-memory state, and a restarted process must
        # not forget vec0 tables that already exist in the (persistent)
        # database file. Filtered on the `sql` column, not just a name
        # pattern — a vec0 virtual table also creates several auxiliary
        # shadow tables (`<name>_chunks`, `<name>_rowids`, etc.) that
        # share the name prefix but aren't themselves queryable vec0
        # tables; only `CREATE VIRTUAL TABLE ... USING vec0` rows are.
        self._known_tables = set()
        for name, sql in self._conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND sql LIKE 'CREATE VIRTUAL TABLE%USING vec0%'"
        ).fetchall():
            if _OLD_PARTITION_KEY_MARKER in sql:
                # Pre-rename table (the partition key was still under its
                # old name — see core/stores/migrations.py's version 11
                # for the equivalent relational-table rename). vec0 has no
                # ALTER TABLE RENAME COLUMN, and this table is a
                # rebuildable cache, not the source of truth (the
                # relational tables are), so the correct fix is dropping
                # it and letting SqliteStore._backfill_vec_index() — which
                # always runs right after this method — recreate and
                # repopulate it under the current column name. Built from
                # two concatenated halves rather than one literal: this
                # string must never read as a match against the *current*
                # schema string a few lines down, which a blind
                # find-and-replace across the file (as happened once
                # already) would otherwise silently turn it into.
                self._conn.execute(f"DROP TABLE {name}")
            else:
                self._known_tables.add(name)
        return True

    def upsert(
        self,
        table: str,
        row_id: int,
        workspace_id: str,
        embedding: list[float],
        cluster_id: int | None = None,
    ) -> None:
        """Insert or replace `row_id`'s vector in `table`.

        Creates the table (dimension inferred from `len(embedding)`) on
        first use for this table name. `cluster_id=None` is stored as
        the unclustered sentinel (see module docstring). No-ops if the
        index isn't available.
        """
        if not self.available:
            return
        self._ensure_table(table, dimension=len(embedding))
        self._conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (row_id,))
        self._conn.execute(
            f"INSERT INTO {table}(rowid, embedding, workspace_id, cluster_id) VALUES (?, ?, ?, ?)",
            (
                row_id,
                _serialize_vec(embedding),
                workspace_id,
                cluster_id if cluster_id is not None else _UNCLUSTERED_SENTINEL,
            ),
        )

    def delete(self, table: str, row_id: int) -> None:
        """Remove `row_id` from `table`, if present. No-ops if the index
        isn't available or `table` was never created."""
        if not self.available or table not in self._known_tables:
            return
        self._conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (row_id,))

    def set_cluster_id(self, table: str, row_id: int, cluster_id: int | None) -> None:
        """Update just `row_id`'s cluster_id metadata, leaving its
        embedding untouched — cheaper than a full upsert for a topic
        joining a cluster after its embedding was already indexed.
        No-ops if the index isn't available or `table` was never
        created (nothing to update)."""
        if not self.available or table not in self._known_tables:
            return
        self._conn.execute(
            f"UPDATE {table} SET cluster_id = ? WHERE rowid = ?",
            (cluster_id if cluster_id is not None else _UNCLUSTERED_SENTINEL, row_id),
        )

    def knn(
        self,
        table: str,
        workspace_id: str,
        query_embedding: list[float],
        k: int,
        unclustered_only: bool = False,
    ) -> list[tuple[int, float]]:
        """Return up to `k` (row_id, similarity) pairs from `table`,
        hard-partitioned to `workspace_id`, nearest-first.

        `similarity` is cosine similarity (`1.0 - distance` — vec0
        reports distance, the rest of the scoring code expects
        similarity), matching the existing `_cosine_similarity`
        convention. `unclustered_only` adds the `cluster_id = 0`
        (unclustered sentinel) filter, for `search_cluster_candidates`'
        raw-topic side. Empty if unavailable or `table` was never
        created (nothing upserted yet).
        """
        if not self.available or table not in self._known_tables:
            return []
        cluster_filter = "AND cluster_id = 0 " if unclustered_only else ""
        rows = self._conn.execute(
            f"SELECT rowid, distance FROM {table} "
            f"WHERE workspace_id = ? {cluster_filter}AND embedding MATCH ? AND k = ? "
            f"ORDER BY distance",
            (workspace_id, _serialize_vec(query_embedding), k),
        ).fetchall()
        return [(row_id, 1.0 - distance) for row_id, distance in rows]

    def _ensure_table(self, table: str, dimension: int) -> None:
        if table in self._known_tables:
            return
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
            f"embedding float[{dimension}] distance_metric=cosine, "
            f"workspace_id TEXT partition key, "
            f"cluster_id INTEGER)"
        )
        self._known_tables.add(table)
