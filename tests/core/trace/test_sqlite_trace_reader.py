"""Tests for the SQLite-backed trace reader.

Covers both the normal read-back path (against a real ``SqliteTracer``
file) and the graceful-empty behavior when the trace file doesn't exist
at all — the key difference from ``tools/trace.py``'s own
``TraceReader``, which intentionally raises ``FileNotFoundError`` for a
CLI tool. A dashboard page must not crash just because a turn happened
to run with ``debug.enabled: false``.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.trace.sqlite_trace_reader import SqliteTraceReader
from core.trace.sqlite_tracer import SqliteTracer


@pytest.fixture
def trace_db_path():
    return str(Path(tempfile.mkdtemp()) / "trace.db")


class TestSqliteTraceReader:
    def test_reads_back_written_events(self, trace_db_path: str):
        tracer = SqliteTracer(trace_db_path)
        tracer.connect()
        tracer.record(
            "topic_detection", {"message": "hello"}, request_id="req1", session_id="sess1", duration_ms=42.5,
        )
        tracer.close()

        reader = SqliteTraceReader(trace_db_path)
        reader.connect()
        events = reader.events_for_request("req1")
        assert len(events) == 1
        assert events[0]["event_type"] == "topic_detection"
        assert events[0]["payload"] == {"message": "hello"}
        assert events[0]["duration_ms"] == 42.5
        assert events[0]["session_id"] == "sess1"
        reader.close()

    def test_events_come_back_oldest_first(self, trace_db_path: str):
        tracer = SqliteTracer(trace_db_path)
        tracer.connect()
        tracer.record("topic_detection", {}, request_id="req1")
        tracer.record("extraction", {}, request_id="req1")
        tracer.close()

        reader = SqliteTraceReader(trace_db_path)
        reader.connect()
        events = reader.events_for_request("req1")
        assert [e["event_type"] for e in events] == ["topic_detection", "extraction"]

    def test_connect_on_missing_file_leaves_reader_empty_not_raising(self, trace_db_path: str):
        # trace_db_path's parent dir exists but the file was never created
        # (debug.enabled was never turned on for this deployment).
        reader = SqliteTraceReader(trace_db_path)
        reader.connect()  # must not raise
        assert reader.events_for_request("anything") == []
        reader.close()

    def test_events_for_request_before_connect_returns_empty(self, trace_db_path: str):
        reader = SqliteTraceReader(trace_db_path)
        assert reader.events_for_request("anything") == []


class TestEventsOfTypeSince:
    def test_reads_back_matching_type_within_window(self, trace_db_path: str):
        tracer = SqliteTracer(trace_db_path)
        tracer.connect()
        tracer.record(
            "retrieval_scoring_summary", {"candidate_type": "topic"},
            request_id="req1", session_id="sess1", duration_ms=12.0,
        )
        tracer.close()

        reader = SqliteTraceReader(trace_db_path)
        reader.connect()
        rows = reader.events_of_type_since("retrieval_scoring_summary", datetime.now(UTC) - timedelta(hours=1))
        assert len(rows) == 1
        assert rows[0]["payload"] == {"candidate_type": "topic"}
        assert rows[0]["request_id"] == "req1"
        assert rows[0]["session_id"] == "sess1"
        assert rows[0]["duration_ms"] == 12.0

    def test_excludes_other_event_types(self, trace_db_path: str):
        tracer = SqliteTracer(trace_db_path)
        tracer.connect()
        tracer.record("topic_detection", {}, request_id="req1")
        tracer.close()

        reader = SqliteTraceReader(trace_db_path)
        reader.connect()
        rows = reader.events_of_type_since("retrieval_scoring_summary", datetime.now(UTC) - timedelta(hours=1))
        assert rows == []

    def test_excludes_rows_before_cutoff(self, trace_db_path: str):
        tracer = SqliteTracer(trace_db_path)
        tracer.connect()
        tracer.record("retrieval_scoring_summary", {}, request_id="req1")
        tracer.close()

        reader = SqliteTraceReader(trace_db_path)
        reader.connect()
        rows = reader.events_of_type_since("retrieval_scoring_summary", datetime.now(UTC) + timedelta(hours=1))
        assert rows == []

    def test_events_come_back_oldest_first(self, trace_db_path: str):
        tracer = SqliteTracer(trace_db_path)
        tracer.connect()
        tracer.record("retrieval_scoring_summary", {"n": 1}, request_id="req1")
        tracer.record("retrieval_scoring_summary", {"n": 2}, request_id="req2")
        tracer.close()

        reader = SqliteTraceReader(trace_db_path)
        reader.connect()
        rows = reader.events_of_type_since("retrieval_scoring_summary", datetime.now(UTC) - timedelta(hours=1))
        assert [r["payload"]["n"] for r in rows] == [1, 2]

    def test_before_connect_returns_empty(self, trace_db_path: str):
        reader = SqliteTraceReader(trace_db_path)
        assert reader.events_of_type_since("retrieval_scoring_summary", datetime.now(UTC)) == []
