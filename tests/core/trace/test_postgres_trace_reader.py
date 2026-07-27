"""Tests for the Postgres-backed trace reader.

Mirrors tests/core/trace/test_postgres_tracer.py's shape — same
session-scoped ``postgres_container``, writes via a real
``PostgresTracer`` then reads back via ``PostgresTraceReader``, and
truncates ``trace_events`` between tests for isolation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.trace.postgres_trace_reader import PostgresTraceReader
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


@pytest.fixture
def reader(postgres_container):
    r = PostgresTraceReader(
        host=postgres_container.get_container_host_ip(),
        port=int(postgres_container.get_exposed_port(5432)),
        dbname=postgres_container.dbname,
        user=postgres_container.username,
        password=postgres_container.password,
    )
    r.connect()
    yield r
    r.close()


class TestPostgresTraceReader:
    def test_reads_back_written_events(self, tracer: PostgresTracer, reader: PostgresTraceReader):
        tracer.record(
            "topic_detection",
            {"message": "hello", "result": "CONTINUE"},
            request_id="req1",
            session_id="sess1",
            duration_ms=42.5,
        )

        events = reader.events_for_request("req1")
        assert len(events) == 1
        assert events[0]["event_type"] == "topic_detection"
        assert events[0]["payload"] == {"message": "hello", "result": "CONTINUE"}
        assert events[0]["duration_ms"] == 42.5
        assert events[0]["session_id"] == "sess1"

    def test_events_come_back_oldest_first(self, tracer: PostgresTracer, reader: PostgresTraceReader):
        tracer.record("topic_detection", {}, request_id="req1")
        tracer.record("extraction", {}, request_id="req1")

        events = reader.events_for_request("req1")
        assert [e["event_type"] for e in events] == ["topic_detection", "extraction"]

    def test_no_events_for_unknown_request(self, reader: PostgresTraceReader):
        assert reader.events_for_request("nope") == []

    def test_connect_creates_trace_events_even_if_no_tracer_ever_ran(self, postgres_container):
        # A reader may be the very first thing to touch trace_events in
        # a deployment that's never turned debug.enabled on — connect()
        # must not require a PostgresTracer to have run its migration first.
        r = PostgresTraceReader(
            host=postgres_container.get_container_host_ip(),
            port=int(postgres_container.get_exposed_port(5432)),
            dbname=postgres_container.dbname,
            user=postgres_container.username,
            password=postgres_container.password,
        )
        r.connect()
        assert r.events_for_request("anything") == []
        r.close()


class TestEventsOfTypeSince:
    def test_reads_back_matching_type_within_window(self, tracer: PostgresTracer, reader: PostgresTraceReader):
        tracer.record(
            "retrieval_scoring_summary", {"candidate_type": "topic"},
            request_id="req1", session_id="sess1", duration_ms=12.0,
        )

        rows = reader.events_of_type_since("retrieval_scoring_summary", datetime.now(UTC) - timedelta(hours=1))
        assert len(rows) == 1
        assert rows[0]["payload"] == {"candidate_type": "topic"}
        assert rows[0]["request_id"] == "req1"
        assert rows[0]["session_id"] == "sess1"
        assert rows[0]["duration_ms"] == 12.0

    def test_excludes_other_event_types(self, tracer: PostgresTracer, reader: PostgresTraceReader):
        tracer.record("topic_detection", {}, request_id="req1")

        rows = reader.events_of_type_since("retrieval_scoring_summary", datetime.now(UTC) - timedelta(hours=1))
        assert rows == []

    def test_excludes_rows_before_cutoff(self, tracer: PostgresTracer, reader: PostgresTraceReader):
        tracer.record("retrieval_scoring_summary", {}, request_id="req1")

        rows = reader.events_of_type_since("retrieval_scoring_summary", datetime.now(UTC) + timedelta(hours=1))
        assert rows == []

    def test_events_come_back_oldest_first(self, tracer: PostgresTracer, reader: PostgresTraceReader):
        tracer.record("retrieval_scoring_summary", {"n": 1}, request_id="req1")
        tracer.record("retrieval_scoring_summary", {"n": 2}, request_id="req2")

        rows = reader.events_of_type_since("retrieval_scoring_summary", datetime.now(UTC) - timedelta(hours=1))
        assert [r["payload"]["n"] for r in rows] == [1, 2]
