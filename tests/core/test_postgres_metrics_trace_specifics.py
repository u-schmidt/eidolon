"""Postgres-specific transaction-hygiene tests for PostgresMetricsReader,
PostgresMetricsRecorder, and PostgresTraceReader.

Same bug class as tests/core/test_postgres_store_specifics.py's own
TestTransactionHygiene — connect()ing with autocommit=False and never
committing/flipping to autocommit leaves a long-lived connection sitting
"idle in transaction" indefinitely after any call, observed live via the
new dashboard performance page (every request calls
PostgresMetricsReader.events_since()). Fixed by connecting with
autocommit=True directly (none of these three classes need
multi-statement atomicity — unlike PostgresStore's connect(), which
needs its migration sequence to commit atomically).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import psycopg
import pytest

from core.metrics.postgres_metrics_reader import PostgresMetricsReader
from core.metrics.postgres_metrics_recorder import PostgresMetricsRecorder
from core.trace.postgres_trace_reader import PostgresTraceReader
from tests.conftest import PostgresBackend


def _state_for_pid(observer: psycopg.Connection, pid: int) -> str | None:
    row = observer.execute(
        "SELECT state FROM pg_stat_activity WHERE pid = %s", (pid,)
    ).fetchone()
    return row[0] if row else None


def _own_pid(conn: psycopg.Connection) -> int:
    return conn.execute("SELECT pg_backend_pid()").fetchone()[0]


def _observer(postgres_container) -> psycopg.Connection:
    return psycopg.connect(
        host=postgres_container.get_container_host_ip(),
        port=int(postgres_container.get_exposed_port(5432)),
        dbname=postgres_container.dbname,
        user=postgres_container.username,
        password=postgres_container.password,
    )


@pytest.fixture
def connection_kwargs(postgres_container) -> dict:
    # events/llm_calls/embedding_calls only exist once the main schema
    # migrations have run — PostgresBackend's own connect() does that,
    # same as ProviderFactory.create_storage() does before any reader/
    # recorder connects in production.
    backend = PostgresBackend(postgres_container)
    kwargs = {
        "host": postgres_container.get_container_host_ip(),
        "port": int(postgres_container.get_exposed_port(5432)),
        "dbname": postgres_container.dbname,
        "user": postgres_container.username,
        "password": postgres_container.password,
    }
    yield kwargs
    backend.truncate_all()
    backend.close()


class TestMetricsReaderTransactionHygiene:
    def test_read_call_leaves_no_open_transaction(self, connection_kwargs, postgres_container):
        reader = PostgresMetricsReader(**connection_kwargs)
        reader.connect()
        pid = _own_pid(reader._conn)

        reader.events_for_request("nope")  # any read-only call; result irrelevant

        with _observer(postgres_container) as observer:
            assert _state_for_pid(observer, pid) != "idle in transaction"
        reader.close()

    def test_since_call_leaves_no_open_transaction(self, connection_kwargs, postgres_container):
        reader = PostgresMetricsReader(**connection_kwargs)
        reader.connect()
        pid = _own_pid(reader._conn)

        reader.events_since(datetime.now(UTC) - timedelta(hours=1))

        with _observer(postgres_container) as observer:
            assert _state_for_pid(observer, pid) != "idle in transaction"
        reader.close()


class TestMetricsRecorderTransactionHygiene:
    def test_write_call_leaves_no_open_transaction(self, connection_kwargs, postgres_container):
        recorder = PostgresMetricsRecorder(**connection_kwargs)
        recorder.connect()
        pid = _own_pid(recorder._conn)

        recorder.record_event("error", "extraction", "boom")

        with _observer(postgres_container) as observer:
            assert _state_for_pid(observer, pid) != "idle in transaction"
        recorder.close()


class TestTraceReaderTransactionHygiene:
    def test_read_call_leaves_no_open_transaction(self, connection_kwargs, postgres_container):
        reader = PostgresTraceReader(**connection_kwargs)
        reader.connect()  # runs migrate() internally
        pid = _own_pid(reader._conn)

        reader.events_for_request("nope")  # any read-only call; result irrelevant

        with _observer(postgres_container) as observer:
            assert _state_for_pid(observer, pid) != "idle in transaction"
        reader.close()
