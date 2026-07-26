"""Tests for the Postgres-backed tracer.

Mirrors tests/core/trace/test_sqlite_tracer.py's shape. Uses the same
session-scoped ``postgres_container`` fixture as the StorageProvider
contract suite (tests/conftest.py) rather than standing up a second
container — truncates ``trace_events`` between tests for isolation.
"""

from __future__ import annotations

import pytest

from core.trace.postgres_tracer import PostgresTracer


@pytest.fixture
def tracer(postgres_container):
    t = PostgresTracer(
        host=postgres_container.get_container_host_ip(),
        port=int(postgres_container.get_exposed_port(5432)),
        dbname=postgres_container.dbname,
        user=postgres_container.username,
        password=postgres_container.password,
    )
    t.connect()
    yield t
    with t._conn.cursor() as cur:
        cur.execute("TRUNCATE trace_events RESTART IDENTITY")
    t._conn.commit()
    t.close()


class TestPostgresTracer:
    def test_record_persists_event(self, tracer: PostgresTracer):
        tracer.record(
            "topic_detection",
            {"message": "hello", "result": "CONTINUE"},
            request_id="req1",
            session_id="sess1",
            duration_ms=42.5,
        )

        rows = tracer._conn.execute(
            "SELECT request_id, session_id, event_type, payload, duration_ms FROM trace_events"
        ).fetchall()
        assert len(rows) == 1
        request_id, session_id, event_type, payload, duration_ms = rows[0]
        assert request_id == "req1"
        assert session_id == "sess1"
        assert event_type == "topic_detection"
        assert payload == {"message": "hello", "result": "CONTINUE"}  # JSONB decodes automatically
        assert duration_ms == 42.5

    def test_record_without_optional_args(self, tracer: PostgresTracer):
        tracer.record("error", {"exception": "boom"})
        rows = tracer._conn.execute(
            "SELECT request_id, session_id, duration_ms FROM trace_events"
        ).fetchall()
        assert rows[0] == (None, None, None)

    def test_record_serializes_non_json_native_values(self, tracer: PostgresTracer):
        class Weird:
            def __str__(self):
                return "weird-value"

        tracer.record("error", {"thing": Weird()})
        row = tracer._conn.execute("SELECT payload FROM trace_events").fetchone()
        assert row[0] == {"thing": "weird-value"}

    def test_multiple_events_share_request_id(self, tracer: PostgresTracer):
        tracer.record("topic_detection", {}, request_id="req1")
        tracer.record("extraction", {}, request_id="req1")
        tracer.record("extraction", {}, request_id="req2")

        rows = tracer._conn.execute(
            "SELECT event_type FROM trace_events WHERE request_id = %s ORDER BY id", ("req1",)
        ).fetchall()
        assert [r[0] for r in rows] == ["topic_detection", "extraction"]

    def test_connect_is_idempotent_across_instances(self, postgres_container):
        # Reconnecting (e.g. a process restart) must not fail or wipe
        # existing data — mirrors SqliteTracer's own migration behavior.
        first = PostgresTracer(
            host=postgres_container.get_container_host_ip(),
            port=int(postgres_container.get_exposed_port(5432)),
            dbname=postgres_container.dbname,
            user=postgres_container.username,
            password=postgres_container.password,
        )
        first.connect()
        first.record("topic_detection", {})
        first.close()

        second = PostgresTracer(
            host=postgres_container.get_container_host_ip(),
            port=int(postgres_container.get_exposed_port(5432)),
            dbname=postgres_container.dbname,
            user=postgres_container.username,
            password=postgres_container.password,
        )
        second.connect()
        row = second._conn.execute("SELECT COUNT(*) FROM trace_events").fetchone()
        assert row[0] == 1
        with second._conn.cursor() as cur:
            cur.execute("TRUNCATE trace_events RESTART IDENTITY")
        second._conn.commit()
        second.close()

    def test_record_before_connect_raises(self):
        tracer = PostgresTracer(host="127.0.0.1", port=5432, dbname="x", user="x", password="x")
        with pytest.raises(RuntimeError):
            tracer.record("error", {})
