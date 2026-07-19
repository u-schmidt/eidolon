"""Tests for tools/simulate_injection.py — InjectionSimulator."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from core.trace.sqlite_tracer import SqliteTracer
from tools.simulate_injection import InjectionSimulator, SimulationResult
from tools.trace import TraceReader


@pytest.fixture
def trace_db(tmp_path):
    """A trace.db with two turns: one with a full context_retrieval event,
    one that never reached retrieval (topic_detection error)."""
    db_path = str(tmp_path / "trace.db")
    tracer = SqliteTracer(db_path)
    tracer.connect()

    tracer.record(
        "context_retrieval",
        {
            "message": "was ist mein lieblingsprojekt?",
            "peer_id": "user",
            "topic_label": "Bello Features",
            "is_new_topic": False,
            "facts": [{"category": "work", "key": "role", "value": "engineer"}],
            "related_topics": [{"id": 1, "summary": "Discussed the EFB project", "tags": ["efb", "aviation"]}],
            "memory_items": [
                {"content": "User loves Bello", "item_type": "essence",
                 "source_peer": "user", "subject_peer": "user"}
            ],
        },
        request_id="req-full", session_id="sess1", duration_ms=2.0,
    )
    tracer.record(
        "error",
        {"step": "topic_detection", "exception": "boom"},
        request_id="req-error", session_id="sess1",
    )
    tracer.close()
    return db_path


class TestResolveRequestId:
    def test_uses_given_request_id_verbatim(self, trace_db):
        reader = TraceReader(trace_db)
        simulator = InjectionSimulator(reader)
        assert simulator.resolve_request_id("req-full", None) == "req-full"
        reader.close()

    def test_falls_back_to_most_recent_when_omitted(self, trace_db):
        reader = TraceReader(trace_db)
        simulator = InjectionSimulator(reader)
        assert simulator.resolve_request_id(None, None) == "req-error"
        reader.close()

    def test_returns_none_when_db_empty(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        tracer = SqliteTracer(db_path)
        tracer.connect()
        tracer.close()

        reader = TraceReader(db_path)
        simulator = InjectionSimulator(reader)
        assert simulator.resolve_request_id(None, None) is None
        reader.close()


class TestSimulate:
    def test_reconstructs_the_exact_injection_string(self, trace_db):
        reader = TraceReader(trace_db)
        simulator = InjectionSimulator(reader)

        result = simulator.simulate("req-full")
        reader.close()

        assert isinstance(result, SimulationResult)
        assert result.request_id == "req-full"
        assert result.user_message == "was ist mein lieblingsprojekt?"
        assert result.injection == (
            "[Current topic: Bello Features]\n\n"
            "[Known facts]\n- work: role = engineer\n\n"
            "[Related topics]\n- Discussed the EFB project (efb, aviation)\n\n"
            "[Memory fragments]\n- [essence] (user about user): User loves Bello"
        )

    def test_returns_none_when_no_context_retrieval_event(self, trace_db):
        reader = TraceReader(trace_db)
        simulator = InjectionSimulator(reader)

        result = simulator.simulate("req-error")
        reader.close()

        assert result is None


class TestSimulateEnvironmentAndAges:
    def test_reconstructs_environment_and_relative_ages(self, tmp_path):
        """Ages must be computed relative to the turn's own recorded time,
        not today — so replaying an old turn shows ages as they were then."""
        db_path = str(tmp_path / "trace.db")
        tracer = SqliteTracer(db_path)
        tracer.connect()

        three_weeks_ago = (datetime.now() - timedelta(days=21)).strftime("%Y-%m-%d %H:%M:%S")
        tracer.record(
            "context_retrieval",
            {
                "message": "hi",
                "peer_id": "user",
                "topic_label": None,
                "is_new_topic": False,
                "environment": {
                    "situation": "conversation", "location": None,
                    "time": datetime.now().isoformat(), "day_of_week": datetime.now().strftime("%A"),
                },
                "facts": [{"category": "work", "key": "role", "value": "engineer", "created_at": three_weeks_ago}],
                "related_topics": [],
                "memory_items": [],
            },
            request_id="req-env", session_id="sess1",
        )
        tracer.close()

        reader = TraceReader(db_path)
        simulator = InjectionSimulator(reader)
        result = simulator.simulate("req-env")
        reader.close()

        assert "[Environment]" in result.injection
        assert "3 weeks ago" in result.injection

    def test_old_trace_without_environment_or_timestamps_degrades_gracefully(self, trace_db):
        """Trace data recorded before this feature existed has no
        environment sub-fields and no per-item timestamps — must not
        crash, and must simply omit the new sections."""
        reader = TraceReader(trace_db)
        simulator = InjectionSimulator(reader)

        result = simulator.simulate("req-full")
        reader.close()

        assert "[Environment]" not in result.injection
        assert "ago" not in result.injection
        assert "today" not in result.injection
