"""Tests for tools/trace.py — TraceReader and TracePrinter."""

from __future__ import annotations

import pytest

from core.trace.sqlite_tracer import SqliteTracer
from tools.trace import TraceReader, TracePrinter, TurnSummary


@pytest.fixture
def trace_db(tmp_path):
    """A trace.db with one representative turn (every current event_type,
    written through the real SqliteTracer — not hand-written SQL)."""
    db_path = str(tmp_path / "trace.db")
    tracer = SqliteTracer(db_path)
    tracer.connect()

    tracer.record(
        "topic_detection",
        {"message": "hello", "current_topic_summary": None, "response": "CONTINUE",
         "result_type": "CONTINUE", "result_label": None},
        request_id="req1", session_id="sess1", duration_ms=5.0,
    )
    tracer.record(
        "embedding",
        {"purpose": "retrieval_query", "text": "hello", "dimension": 768},
        request_id="req1", session_id="sess1", duration_ms=1.0,
    )
    tracer.record(
        "retrieval_scoring_summary",
        {"candidate_type": "topic", "total_scored": 1, "included_count": 1,
         "top_candidates": [
             {"candidate_id": 1, "semantic_raw": 0.5, "semantic_weighted": 0.2,
              "tag_overlap_raw": 0.0, "tag_overlap_weighted": 0.0,
              "person_weighted": 0.0, "final_score": 0.2, "included": True},
         ]},
        request_id="req1", session_id="sess1",
    )
    tracer.record(
        "retrieval_scoring_summary",
        {"candidate_type": "memory_item", "total_scored": 1, "included_count": 1,
         "top_candidates": [
             {"candidate_id": 7, "semantic_raw": 0.6, "semantic_weighted": 0.24,
              "person_weighted": 0.2, "final_score": 0.44, "included": True},
         ]},
        request_id="req1", session_id="sess1",
    )
    tracer.record(
        "context_retrieval",
        {"message": "hello", "peer_id": "user", "topic_label": None, "is_new_topic": False,
         "tags": [], "environment": {"situation": "conversation", "location": None,
                                      "time": None, "day_of_week": None},
         "facts": [{"category": "fitness", "key": "gym", "value": "FitZone"}],
         "related_topics": [{"id": 1, "summary": "About X", "tags": []}],
         "memory_items": [{"content": "note", "item_type": "essence",
                            "source_peer": "user", "subject_peer": "user"}]},
        request_id="req1", session_id="sess1", duration_ms=2.0,
    )
    tracer.record(
        "extraction",
        {"user_message": "hello", "ai_response": "hi", "response": "[]",
         "items": [{"content": "note", "item_type": "essence", "attribution_type": "self_statement"}]},
        request_id="req1", session_id="sess1",
    )
    tracer.record(
        "turn_complete",
        {"user_message": "hello", "ai_response": "hi", "old_topic_id": None,
         "new_topic_id": None, "new_topic_label": None, "items_extracted": 1, "fact_items": 0},
        request_id="req1", session_id="sess1", duration_ms=10.0,
    )
    tracer.close()
    return db_path


class TestTraceReaderRecentTurns:
    def test_recent_turns_returns_one_turn(self, trace_db):
        reader = TraceReader(trace_db)
        turns = reader.recent_turns(limit=10)
        reader.close()

        assert len(turns) == 1
        assert isinstance(turns[0], TurnSummary)
        assert turns[0].request_id == "req1"
        assert turns[0].session_id == "sess1"
        assert turns[0].message_preview == "hello"

    def test_recent_turns_empty_db(self, tmp_path):
        db_path = str(tmp_path / "empty.db")
        tracer = SqliteTracer(db_path)
        tracer.connect()
        tracer.close()

        reader = TraceReader(db_path)
        assert reader.recent_turns(limit=10) == []
        reader.close()


class TestTracePrinterFormatSummary:
    def test_format_summary_topic_detection(self):
        event = {"event_type": "topic_detection",
                  "payload": {"result_type": "CONTINUE", "result_label": None}}
        assert TracePrinter().format_summary(event) == "matched against topic → CONTINUE"

    def test_format_summary_embedding(self):
        event = {"event_type": "embedding", "payload": {"purpose": "memory_item", "dimension": 1024}}
        assert TracePrinter().format_summary(event) == "embedded [memory_item] → dim 1024"

    def test_format_summary_unknown_event_type_falls_back_to_type(self):
        event = {"event_type": "mystery", "payload": {}}
        assert TracePrinter().format_summary(event) == "mystery"

    def test_format_summary_fact_consolidation(self):
        event = {
            "event_type": "fact_consolidation",
            "payload": {
                "upserts": [{"category": "location", "key": "city", "value": "Berlin"}],
                "deletes": [{"category": "lifestyle", "key": "smoking"}],
            },
        }
        assert TracePrinter().format_summary(event) == "consolidated → 1 upsert(s), 1 delete(s)"

    def test_format_summary_fact_correction(self):
        event = {
            "event_type": "fact_correction",
            "payload": {"peer_id": "user", "category": "location", "key": "city",
                        "value": "Berlin", "action": "upsert"},
        }
        assert TracePrinter().format_summary(event) == "corrected fact → upsert location.city"

    def test_format_summary_retrieval_scoring_summary(self):
        event = {
            "event_type": "retrieval_scoring_summary",
            "payload": {"candidate_type": "topic", "total_scored": 5, "included_count": 2,
                        "top_candidates": []},
        }
        assert TracePrinter().format_summary(event) == "topic scoring → 2/5 included"


class TestTracePrinterFormatDetail:
    def test_format_detail_topic_detection_shows_recent_message_count(self):
        event = {
            "event_type": "topic_detection",
            "duration_ms": 5.0,
            "created_at": "2026-07-20 12:00:00",
            "payload": {"message": "hi", "current_topic_summary": "General conversation",
                        "recent_message_count": 4, "response": "CONTINUE",
                        "result_type": "CONTINUE", "result_label": None},
        }
        detail = TracePrinter().format_detail(event)
        assert "recent_messages: 4" in detail

    def test_format_detail_includes_header_and_body(self, trace_db):
        reader = TraceReader(trace_db)
        events = reader.events_for_request("req1")
        reader.close()

        detail = TracePrinter().format_detail(events[0])
        assert "topic_detection" in detail
        assert "message: hello" in detail

    def test_format_detail_context_retrieval_shows_full_content(self, trace_db):
        reader = TraceReader(trace_db)
        events = reader.events_for_request("req1")
        reader.close()

        cr_event = next(e for e in events if e["event_type"] == "context_retrieval")
        detail = TracePrinter().format_detail(cr_event)
        assert "fitness.gym = FitZone" in detail
        assert "topic #1" in detail

    def test_format_detail_fact_consolidation(self):
        event = {
            "event_type": "fact_consolidation",
            "duration_ms": 12.5,
            "created_at": "2026-07-19 12:00:00",
            "payload": {
                "peer_id": "user",
                "new_item_count": 2,
                "total_facts_after": 3,
                "upserts": [{"category": "location", "key": "city", "value": "Berlin"}],
                "deletes": [{"category": "lifestyle", "key": "smoking"}],
                "response": '{"upserts": [...], "deletes": [...]}',
            },
        }
        detail = TracePrinter().format_detail(event)
        assert "location.city = Berlin" in detail
        assert "lifestyle.smoking" in detail
        assert "total_facts_after: 3" in detail

    def test_format_detail_fact_correction(self):
        event = {
            "event_type": "fact_correction",
            "duration_ms": 1.0,
            "created_at": "2026-07-19 12:00:00",
            "payload": {"peer_id": "user", "category": "location", "key": "city",
                        "value": "Berlin", "action": "upsert"},
        }
        detail = TracePrinter().format_detail(event)
        assert "action: upsert" in detail
        assert "location.city = Berlin" in detail

    def test_format_detail_retrieval_scoring_summary(self):
        event = {
            "event_type": "retrieval_scoring_summary",
            "duration_ms": 1.0,
            "created_at": "2026-07-19 12:00:00",
            "payload": {
                "candidate_type": "topic", "total_scored": 3, "included_count": 1,
                "top_candidates": [
                    {"candidate_id": 5, "final_score": 0.6, "semantic_raw": 0.5, "semantic_weighted": 0.2,
                     "tag_overlap_raw": 0.1, "tag_overlap_weighted": 0.03, "person_weighted": 0.2,
                     "included": True},
                    {"candidate_id": 6, "final_score": 0.1, "semantic_raw": 0.1, "semantic_weighted": 0.04,
                     "tag_overlap_raw": 0.0, "tag_overlap_weighted": 0.0, "person_weighted": 0.0,
                     "included": False},
                ],
            },
        }
        detail = TracePrinter().format_detail(event)
        assert "topic: 3 scored, 1 included (showing top 2)" in detail
        assert "[✓] #5" in detail
        assert "[ ] #6" in detail

    def test_format_detail_error_without_response(self):
        event = {
            "event_type": "error",
            "duration_ms": None,
            "created_at": "2026-07-19 12:00:00",
            "payload": {"step": "topic_detection", "exception": "RuntimeError('LLM backend unavailable')"},
        }
        detail = TracePrinter().format_detail(event)
        assert "step: topic_detection" in detail
        assert "exception: RuntimeError" in detail
        assert "response:" not in detail

    def test_format_detail_error_with_response(self):
        event = {
            "event_type": "error",
            "duration_ms": None,
            "created_at": "2026-07-19 12:00:00",
            "payload": {"step": "extraction", "exception": "ValueError('malformed extraction response')",
                        "response": "not json at all"},
        }
        detail = TracePrinter().format_detail(event)
        assert "step: extraction" in detail
        assert "response: not json at all" in detail

    def test_summary_error(self):
        event = {"event_type": "error", "payload": {"step": "extraction", "exception": "boom"}}
        assert TracePrinter().format_summary(event) == "ERROR in extraction"
