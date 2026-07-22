"""One-time data migration: SQLite → Postgres/pgvector.

Reads every row from a SQLite Eidolon database (read-only — never
mutates the source, so it's safe to re-run against the same backup
copy repeatedly) and loads it into a Postgres database via PostgresStore's
schema (core/stores/postgres_schema.sql), preserving row ids exactly
(foreign keys — messages.topic_id, memory_items.topic_id/message_id,
topics.cluster_id — reference those exact ids, so re-numbering during
migration would mean rewriting every reference instead of copying them
verbatim).

Usage:
    # Dry run — validates connectivity/schema, prints source row counts,
    # writes nothing:
    uv run python tools/migrate_sqlite_to_postgres.py \\
        --sqlite-path ~/.eidolon/memory.db --pg-host 127.0.0.1 --pg-port 5432 \\
        --pg-dbname eidolon --pg-user eidolon --pg-password secret \\
        --embedding-dim 1024 --dry-run

    # Real run against a freshly created, empty target database:
    uv run python tools/migrate_sqlite_to_postgres.py \\
        --sqlite-path ~/.eidolon/memory.db --pg-host 127.0.0.1 --pg-port 5432 \\
        --pg-dbname eidolon --pg-user eidolon --pg-password secret \\
        --embedding-dim 1024 --truncate

--embedding-dim must match the configured embedder's dimension exactly
(models.embedder.dimension in eidolon.yaml) — pgvector's VECTOR(n)
column width is fixed at schema-creation time.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Ensure the project root is on sys.path for direct execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from core.stores.postgres_migrations import migrate as migrate_postgres_schema
from core.stores.sqlite_store import _deserialize_embedding

# Respects FK dependencies (see core/stores/postgres_schema.sql): facts and
# fact_consolidation_state have none; topic_clusters has none; topics
# references topic_clusters; messages references topics; memory_items
# references topics and messages.
_TABLE_ORDER = ["facts", "fact_consolidation_state", "topic_clusters", "topics", "messages", "memory_items"]
# Only these have an id-based IDENTITY sequence to re-seed after migration
# (fact_consolidation_state's primary key is peer_id, not id).
_IDENTITY_TABLES = ["facts", "topic_clusters", "topics", "messages", "memory_items"]
_EMBEDDED_TABLES = ["topics", "memory_items", "topic_clusters"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate an Eidolon SQLite database into Postgres/pgvector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sqlite-path", required=True, help="Path to the source SQLite database")
    parser.add_argument("--pg-host", required=True, help="Postgres host")
    parser.add_argument("--pg-port", type=int, default=5432, help="Postgres port")
    parser.add_argument("--pg-dbname", required=True, help="Postgres database name")
    parser.add_argument("--pg-user", required=True, help="Postgres user")
    parser.add_argument("--pg-password", required=True, help="Postgres password")
    parser.add_argument(
        "--embedding-dim", type=int, required=True,
        help="Embedder vector dimension — must match models.embedder.dimension exactly",
    )
    parser.add_argument(
        "--truncate", action="store_true",
        help="Truncate all target tables before loading. Required if the target already has rows in "
             "any migrated table; without it the script refuses to run rather than load on top of "
             "existing data.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate connectivity and apply/verify the schema, print source row counts, write no data.",
    )
    return parser


def _row_counts(conn, tables: list[str]) -> dict[str, int]:
    counts = {}
    for table in tables:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = row[0]
    return counts


def _print_counts(title: str, counts: dict[str, int]) -> None:
    print(f"\n{title}")
    for table, count in counts.items():
        print(f"  {table:<28} {count}")


def _migrate_facts(sqlite_conn: sqlite3.Connection, pg_conn: psycopg.Connection) -> None:
    rows = sqlite_conn.execute(
        "SELECT id, peer_id, category, key, value, source, importance, created_at FROM facts ORDER BY id"
    ).fetchall()
    for r in rows:
        pg_conn.execute(
            "INSERT INTO facts (id, peer_id, category, key, value, source, importance, created_at) "
            "OVERRIDING SYSTEM VALUE VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            r,
        )


def _migrate_fact_consolidation_state(sqlite_conn: sqlite3.Connection, pg_conn: psycopg.Connection) -> None:
    rows = sqlite_conn.execute(
        "SELECT peer_id, last_item_id, last_run_at FROM fact_consolidation_state"
    ).fetchall()
    for r in rows:
        pg_conn.execute(
            "INSERT INTO fact_consolidation_state (peer_id, last_item_id, last_run_at) VALUES (%s, %s, %s)",
            r,
        )


def _migrate_topic_clusters(sqlite_conn: sqlite3.Connection, pg_conn: psycopg.Connection) -> None:
    rows = sqlite_conn.execute(
        "SELECT id, peer_id, label, summary, tags, embedding, topic_count, created_at, updated_at "
        "FROM topic_clusters ORDER BY id"
    ).fetchall()
    for r in rows:
        embedding = _deserialize_embedding(r[5]) if r[5] is not None else None
        pg_conn.execute(
            "INSERT INTO topic_clusters (id, peer_id, label, summary, tags, embedding, topic_count, "
            "created_at, updated_at) OVERRIDING SYSTEM VALUE VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (r[0], r[1], r[2], r[3], Jsonb(json.loads(r[4])), embedding, r[6], r[7], r[8]),
        )


def _migrate_topics(sqlite_conn: sqlite3.Connection, pg_conn: psycopg.Connection) -> None:
    rows = sqlite_conn.execute(
        "SELECT id, label, summary, tags, embedding, status, peer_id, session_id, cluster_id, "
        "created_at, closed_at FROM topics ORDER BY id"
    ).fetchall()
    for r in rows:
        embedding = _deserialize_embedding(r[4]) if r[4] is not None else None
        pg_conn.execute(
            "INSERT INTO topics (id, label, summary, tags, embedding, status, peer_id, session_id, "
            "cluster_id, created_at, closed_at) OVERRIDING SYSTEM VALUE "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (r[0], r[1], r[2], Jsonb(json.loads(r[3])), embedding, r[5], r[6], r[7], r[8], r[9], r[10]),
        )


def _migrate_messages(sqlite_conn: sqlite3.Connection, pg_conn: psycopg.Connection) -> None:
    rows = sqlite_conn.execute(
        "SELECT id, turn_id, role, content, peer_id, topic_id, created_at FROM messages ORDER BY id"
    ).fetchall()
    for r in rows:
        pg_conn.execute(
            "INSERT INTO messages (id, turn_id, role, content, peer_id, topic_id, created_at) "
            "OVERRIDING SYSTEM VALUE VALUES (%s, %s, %s, %s, %s, %s, %s)",
            r,
        )


def _migrate_memory_items(sqlite_conn: sqlite3.Connection, pg_conn: psycopg.Connection) -> None:
    rows = sqlite_conn.execute(
        "SELECT id, content, item_type, source_peer, subject_peer, attribution_type, topic_id, "
        "message_id, embedding, created_at FROM memory_items ORDER BY id"
    ).fetchall()
    for r in rows:
        embedding = _deserialize_embedding(r[8]) if r[8] is not None else None
        pg_conn.execute(
            "INSERT INTO memory_items (id, content, item_type, source_peer, subject_peer, "
            "attribution_type, topic_id, message_id, embedding, created_at) OVERRIDING SYSTEM VALUE "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], embedding, r[9]),
        )


_MIGRATORS = {
    "facts": _migrate_facts,
    "fact_consolidation_state": _migrate_fact_consolidation_state,
    "topic_clusters": _migrate_topic_clusters,
    "topics": _migrate_topics,
    "messages": _migrate_messages,
    "memory_items": _migrate_memory_items,
}


def _reseed_sequences(pg_conn: psycopg.Connection) -> None:
    """After OVERRIDING SYSTEM VALUE inserts, the next auto-generated id
    must continue after the highest migrated id, not collide with it.
    Skips empty tables entirely — touching an empty table's sequence
    would skip id 1 for its first real insert."""
    for table in _IDENTITY_TABLES:
        row = pg_conn.execute(f"SELECT MAX(id) FROM {table}").fetchone()
        if row is not None and row[0] is not None:
            pg_conn.execute("SELECT setval(pg_get_serial_sequence(%s, 'id'), %s)", (table, row[0]))


def _verify_row_counts(sqlite_conn: sqlite3.Connection, pg_conn: psycopg.Connection) -> bool:
    sqlite_counts = _row_counts(sqlite_conn, _TABLE_ORDER)
    pg_counts = _row_counts(pg_conn, _TABLE_ORDER)
    print("\nVerification — row counts (source vs target):")
    ok = True
    for table in _TABLE_ORDER:
        match = sqlite_counts[table] == pg_counts[table]
        ok = ok and match
        status = "OK" if match else "MISMATCH"
        print(f"  {table:<28} sqlite={sqlite_counts[table]:<6} postgres={pg_counts[table]:<6} {status}")
    return ok


def _spot_check_embeddings(pg_conn: psycopg.Connection, table: str, sample_size: int = 5) -> bool:
    """A migrated embedding compared against itself via pgvector's <=>
    should be ~0 distance — a cheap correctness signal beyond row counts,
    catching e.g. a dimension mismatch or byte-order bug in the BLOB
    deserialization that a plain count wouldn't."""
    rows = pg_conn.execute(
        f"SELECT id, embedding <=> embedding AS self_distance FROM {table} "
        f"WHERE embedding IS NOT NULL ORDER BY random() LIMIT %s",
        (sample_size,),
    ).fetchall()
    if not rows:
        print(f"  {table:<28} no embedded rows to spot-check")
        return True
    ok = all(distance is not None and distance < 1e-6 for _, distance in rows)
    print(f"  {table:<28} spot-checked {len(rows)} embedded row(s) — {'OK' if ok else 'FAILED'}")
    if not ok:
        for row_id, distance in rows:
            if distance is None or distance >= 1e-6:
                print(f"    ⚠ id={row_id}: self-distance={distance} (expected ~0)")
    return ok


def main() -> None:
    args = build_arg_parser().parse_args()

    sqlite_path = Path(args.sqlite_path).expanduser().resolve()
    if not sqlite_path.exists():
        print(f"❌ SQLite source not found: {sqlite_path}")
        sys.exit(1)

    print(f"Source (read-only): {sqlite_path}")
    sqlite_conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    _print_counts("Source row counts:", _row_counts(sqlite_conn, _TABLE_ORDER))

    print(f"\nTarget: postgresql://{args.pg_user}@{args.pg_host}:{args.pg_port}/{args.pg_dbname}")
    pg_conn = psycopg.connect(
        host=args.pg_host, port=args.pg_port, dbname=args.pg_dbname,
        user=args.pg_user, password=args.pg_password, autocommit=False,
    )
    migrate_postgres_schema(pg_conn, args.embedding_dim)
    register_vector(pg_conn)
    pg_conn.commit()
    print("Schema applied/verified on target.")

    target_counts = _row_counts(pg_conn, _TABLE_ORDER)
    non_empty = [t for t, c in target_counts.items() if c > 0]
    if non_empty and not args.truncate:
        print(f"\n❌ Target already has rows in: {', '.join(non_empty)}")
        print("   Refusing to load on top of existing data without --truncate.")
        pg_conn.close()
        sqlite_conn.close()
        sys.exit(1)

    if args.dry_run:
        print("\n--dry-run: schema validated, connectivity confirmed, no data written.")
        pg_conn.close()
        sqlite_conn.close()
        return

    if non_empty:
        print(f"\n🧹 Truncating target tables: {', '.join(_TABLE_ORDER)}")
        pg_conn.execute(f"TRUNCATE {', '.join(_TABLE_ORDER)} RESTART IDENTITY CASCADE")
        pg_conn.commit()

    print("\nMigrating...")
    for table in _TABLE_ORDER:
        _MIGRATORS[table](sqlite_conn, pg_conn)
    _reseed_sequences(pg_conn)
    pg_conn.commit()
    print("Data loaded, sequences re-seeded.")

    ok = _verify_row_counts(sqlite_conn, pg_conn)
    print("\nEmbedding spot-checks:")
    for table in _EMBEDDED_TABLES:
        ok = _spot_check_embeddings(pg_conn, table) and ok

    sqlite_conn.close()
    pg_conn.close()

    if ok:
        print("\n✅ Migration complete and verified.")
    else:
        print("\n⚠  Migration completed but verification found issues — see output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
