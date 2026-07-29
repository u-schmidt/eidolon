"""Backend-agnostic MetricsRecorder contract tests.

Runs the same assertions against both SqliteMetricsRecorder and
PostgresMetricsRecorder (the ``metrics_recorder``/``store_backend``
fixtures in tests/conftest.py are parametrized over both) — mirrors
tests/core/test_storage_provider_contract.py's own rationale for
StorageProvider. Reads rows back via the underlying store's raw
connection (same reach-into-``_conn`` pattern
``store_backend.backdate()`` already uses) since MetricsRecorder itself
has no read methods — it's a write-only operational log.
"""

from __future__ import annotations

from core.interfaces import MetricsRecorder


def _rows(store_backend, table: str, order_by: str = "id") -> list[tuple]:
    # Both sqlite3.Connection and psycopg.Connection support .execute()
    # directly (same as test_postgres_store_specifics.py already relies on).
    result = store_backend.store._conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}")
    return result.fetchall()


class TestRecordEvent:
    def test_writes_a_readable_row(self, metrics_recorder: MetricsRecorder, store_backend):
        metrics_recorder.record_event(
            "error", "fact_consolidation", "boom",
            workspace_id="user", request_id="req1", session_id="sess1",
        )

        rows = _rows(store_backend, "events")
        assert len(rows) == 1
        # Columns: id, severity, step, message, workspace_id, request_id, session_id, created_at
        _id, severity, step, message, workspace_id, request_id, session_id, created_at = rows[0]
        assert severity == "error"
        assert step == "fact_consolidation"
        assert message == "boom"
        assert workspace_id == "user"
        assert request_id == "req1"
        assert session_id == "sess1"
        assert created_at is not None

    def test_nullable_fields_default_to_none(self, metrics_recorder: MetricsRecorder, store_backend):
        metrics_recorder.record_event("warning", "extraction", "hmm")

        rows = _rows(store_backend, "events")
        assert len(rows) == 1
        assert rows[0][4] is None  # workspace_id
        assert rows[0][5] is None  # request_id
        assert rows[0][6] is None  # session_id


class TestRecordLlmCall:
    def test_writes_a_readable_row(self, metrics_recorder: MetricsRecorder, store_backend):
        metrics_recorder.record_llm_call(
            "topic_detection", 123.4, True,
            workspace_id="user", request_id="req1", session_id="sess1",
        )

        rows = _rows(store_backend, "llm_calls")
        assert len(rows) == 1
        # Columns: id, call_type, duration_ms, succeeded, workspace_id, request_id, session_id, created_at
        _id, call_type, duration_ms, succeeded, workspace_id, request_id, session_id, created_at = rows[0]
        assert call_type == "topic_detection"
        assert duration_ms == 123.4
        assert bool(succeeded) is True
        assert workspace_id == "user"
        assert request_id == "req1"
        assert session_id == "sess1"

    def test_failed_call_recorded_as_not_succeeded(self, metrics_recorder: MetricsRecorder, store_backend):
        metrics_recorder.record_llm_call("extraction", 5.0, False)

        rows = _rows(store_backend, "llm_calls")
        assert bool(rows[0][3]) is False


class TestRecordEmbeddingCall:
    def test_writes_a_readable_row(self, metrics_recorder: MetricsRecorder, store_backend):
        metrics_recorder.record_embedding_call(
            "memory_item", 5.1, 768, True,
            entity_type="memory_item", entity_id="42",
            workspace_id="user", request_id="req1", session_id="sess1",
        )

        rows = _rows(store_backend, "embedding_calls")
        assert len(rows) == 1
        (
            _id, purpose, duration_ms, dimension, succeeded, entity_type, entity_id,
            workspace_id, request_id, session_id, created_at,
        ) = rows[0]
        assert purpose == "memory_item"
        assert duration_ms == 5.1
        assert dimension == 768
        assert bool(succeeded) is True
        assert entity_type == "memory_item"
        assert entity_id == "42"
        assert workspace_id == "user"
        assert request_id == "req1"
        assert session_id == "sess1"

    def test_entity_fields_are_nullable(self, metrics_recorder: MetricsRecorder, store_backend):
        metrics_recorder.record_embedding_call("retrieval_query", 3.0, 768, True)

        rows = _rows(store_backend, "embedding_calls")
        assert rows[0][5] is None  # entity_type
        assert rows[0][6] is None  # entity_id


class TestMultipleWrites:
    def test_rows_accumulate_across_calls(self, metrics_recorder: MetricsRecorder, store_backend):
        metrics_recorder.record_event("error", "extraction", "one")
        metrics_recorder.record_event("error", "extraction", "two")
        metrics_recorder.record_llm_call("extraction", 1.0, True)
        metrics_recorder.record_embedding_call("memory_item", 1.0, 768, True)

        assert len(_rows(store_backend, "events")) == 2
        assert len(_rows(store_backend, "llm_calls")) == 1
        assert len(_rows(store_backend, "embedding_calls")) == 1
