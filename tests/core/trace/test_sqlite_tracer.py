"""Tests for the SQLite-backed tracer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.trace.sqlite_tracer import SqliteTracer


@pytest.fixture
def tracer():
    tmp = tempfile.mkdtemp()
    db_path = str(Path(tmp) / "trace.db")
    t = SqliteTracer(db_path)
    t.connect()
    yield t
    t.close()


class TestSqliteTracer:
    def test_record_persists_event(self, tracer: SqliteTracer):
        tracer.record(
            "topic_detection",
            {"message": "hello", "result": "CONTINUE"},
            request_id="req1",
            session_id="sess1",
            duration_ms=42.5,
        )

        rows = tracer._conn.execute(
            "SELECT request_id, session_id, event_type, payload, duration_ms FROM events"
        ).fetchall()
        assert len(rows) == 1
        request_id, session_id, event_type, payload, duration_ms = rows[0]
        assert request_id == "req1"
        assert session_id == "sess1"
        assert event_type == "topic_detection"
        assert json.loads(payload) == {"message": "hello", "result": "CONTINUE"}
        assert duration_ms == 42.5

    def test_record_without_optional_args(self, tracer: SqliteTracer):
        tracer.record("error", {"exception": "boom"})
        rows = tracer._conn.execute(
            "SELECT request_id, session_id, duration_ms FROM events"
        ).fetchall()
        assert rows[0] == (None, None, None)

    def test_record_serializes_non_json_native_values(self, tracer: SqliteTracer):
        class Weird:
            def __str__(self):
                return "weird-value"

        tracer.record("error", {"thing": Weird()})
        row = tracer._conn.execute("SELECT payload FROM events").fetchone()
        assert json.loads(row[0]) == {"thing": "weird-value"}

    def test_multiple_events_share_request_id(self, tracer: SqliteTracer):
        tracer.record("topic_detection", {}, request_id="req1")
        tracer.record("extraction", {}, request_id="req1")
        tracer.record("extraction", {}, request_id="req2")

        rows = tracer._conn.execute(
            "SELECT event_type FROM events WHERE request_id = ? ORDER BY id", ("req1",)
        ).fetchall()
        assert [r[0] for r in rows] == ["topic_detection", "extraction"]

    def test_connect_is_idempotent_across_instances(self, tracer: SqliteTracer):
        # Reconnecting to the same file (e.g. process restart) must not fail
        # or wipe existing data — mirrors SqliteStore's migration behavior.
        tracer.record("topic_detection", {})
        path = tracer._path
        tracer.close()

        second = SqliteTracer(str(path))
        second.connect()
        rows = second._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert rows[0] == 1
        second.close()

    def test_record_before_connect_raises(self):
        tracer = SqliteTracer(str(Path(tempfile.mkdtemp()) / "trace.db"))
        with pytest.raises(RuntimeError):
            tracer.record("error", {})
