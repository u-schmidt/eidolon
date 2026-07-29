"""Backend-agnostic MetricsReader contract tests.

Runs the same assertions against both SqliteMetricsReader and
PostgresMetricsReader (the ``metrics_reader``/``store_backend``
fixtures in tests/conftest.py are parametrized over both) — mirrors
tests/core/test_metrics_recorder_contract.py's own rationale. Writes go
through the real ``metrics_recorder`` fixture (the production write
path), reads through ``metrics_reader`` (the new read path this suite
covers) — end-to-end round trip, not a raw-SQL check.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from core.interfaces import MetricsReader, MetricsRecorder


class TestEventsForRequest:
    def test_reads_back_written_rows(self, metrics_recorder: MetricsRecorder, metrics_reader: MetricsReader):
        metrics_recorder.record_event(
            "error", "fact_consolidation", "boom",
            workspace_id="user", request_id="req1", session_id="sess1",
        )

        rows = metrics_reader.events_for_request("req1")
        assert len(rows) == 1
        assert rows[0]["severity"] == "error"
        assert rows[0]["step"] == "fact_consolidation"
        assert rows[0]["message"] == "boom"
        assert rows[0]["workspace_id"] == "user"
        assert rows[0]["session_id"] == "sess1"
        assert rows[0]["created_at"] is not None

    def test_filters_by_request_id(self, metrics_recorder: MetricsRecorder, metrics_reader: MetricsReader):
        metrics_recorder.record_event("error", "extraction", "one", request_id="req1")
        metrics_recorder.record_event("error", "extraction", "two", request_id="req2")

        assert len(metrics_reader.events_for_request("req1")) == 1
        assert len(metrics_reader.events_for_request("req2")) == 1

    def test_no_rows_for_unknown_request(self, metrics_reader: MetricsReader):
        assert metrics_reader.events_for_request("nope") == []


class TestLlmCallsForRequest:
    def test_reads_back_written_rows(self, metrics_recorder: MetricsRecorder, metrics_reader: MetricsReader):
        metrics_recorder.record_llm_call(
            "topic_detection", 123.4, True,
            workspace_id="user", request_id="req1", session_id="sess1",
        )

        rows = metrics_reader.llm_calls_for_request("req1")
        assert len(rows) == 1
        assert rows[0]["call_type"] == "topic_detection"
        assert rows[0]["duration_ms"] == 123.4
        assert rows[0]["succeeded"] is True

    def test_failed_call_reads_back_as_not_succeeded(
        self, metrics_recorder: MetricsRecorder, metrics_reader: MetricsReader,
    ):
        metrics_recorder.record_llm_call("extraction", 5.0, False, request_id="req1")
        rows = metrics_reader.llm_calls_for_request("req1")
        assert rows[0]["succeeded"] is False


class TestEmbeddingCallsForRequest:
    def test_reads_back_written_rows(self, metrics_recorder: MetricsRecorder, metrics_reader: MetricsReader):
        metrics_recorder.record_embedding_call(
            "memory_item", 5.1, 768, True,
            entity_type="memory_item", entity_id="42", request_id="req1",
        )

        rows = metrics_reader.embedding_calls_for_request("req1")
        assert len(rows) == 1
        assert rows[0]["purpose"] == "memory_item"
        assert rows[0]["duration_ms"] == 5.1
        assert rows[0]["dimension"] == 768
        assert rows[0]["succeeded"] is True
        assert rows[0]["entity_type"] == "memory_item"
        assert rows[0]["entity_id"] == "42"


class TestOrdering:
    def test_rows_come_back_oldest_first(self, metrics_recorder: MetricsRecorder, metrics_reader: MetricsReader):
        metrics_recorder.record_llm_call("topic_detection", 1.0, True, request_id="req1")
        metrics_recorder.record_llm_call("extraction", 2.0, True, request_id="req1")

        rows = metrics_reader.llm_calls_for_request("req1")
        assert [r["call_type"] for r in rows] == ["topic_detection", "extraction"]


class TestEventsSince:
    def test_reads_back_written_rows(self, metrics_recorder: MetricsRecorder, metrics_reader: MetricsReader):
        metrics_recorder.record_event(
            "error", "fact_consolidation", "boom",
            workspace_id="user", request_id="req1", session_id="sess1",
        )

        rows = metrics_reader.events_since(datetime.now(UTC) - timedelta(hours=1))
        assert len(rows) == 1
        assert rows[0]["severity"] == "error"
        assert rows[0]["step"] == "fact_consolidation"

    def test_excludes_rows_before_cutoff(self, metrics_recorder: MetricsRecorder, metrics_reader: MetricsReader):
        metrics_recorder.record_event("error", "extraction", "boom", request_id="req1")

        rows = metrics_reader.events_since(datetime.now(UTC) + timedelta(hours=1))
        assert rows == []


class TestLlmCallsSince:
    def test_reads_back_written_rows(self, metrics_recorder: MetricsRecorder, metrics_reader: MetricsReader):
        metrics_recorder.record_llm_call("topic_detection", 42.0, True, request_id="req1")

        rows = metrics_reader.llm_calls_since(datetime.now(UTC) - timedelta(hours=1))
        assert len(rows) == 1
        assert rows[0]["call_type"] == "topic_detection"
        assert rows[0]["duration_ms"] == 42.0

    def test_excludes_rows_before_cutoff(self, metrics_recorder: MetricsRecorder, metrics_reader: MetricsReader):
        metrics_recorder.record_llm_call("topic_detection", 42.0, True, request_id="req1")

        rows = metrics_reader.llm_calls_since(datetime.now(UTC) + timedelta(hours=1))
        assert rows == []


class TestEmbeddingCallsSince:
    def test_reads_back_written_rows(self, metrics_recorder: MetricsRecorder, metrics_reader: MetricsReader):
        metrics_recorder.record_embedding_call("memory_item", 5.1, 768, True, request_id="req1")

        rows = metrics_reader.embedding_calls_since(datetime.now(UTC) - timedelta(hours=1))
        assert len(rows) == 1
        assert rows[0]["purpose"] == "memory_item"

    def test_excludes_rows_before_cutoff(self, metrics_recorder: MetricsRecorder, metrics_reader: MetricsReader):
        metrics_recorder.record_embedding_call("memory_item", 5.1, 768, True, request_id="req1")

        rows = metrics_reader.embedding_calls_since(datetime.now(UTC) + timedelta(hours=1))
        assert rows == []
