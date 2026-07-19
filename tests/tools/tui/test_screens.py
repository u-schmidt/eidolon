"""Tests for the trace explorer TUI screens — one full walk through the
drill-down stack (a -> g) plus targeted checks per screen.

Uses Textual's own App.run_test()/Pilot harness against a trace.db
populated via the real SqliteTracer (not hand-written SQL).
"""

from __future__ import annotations

import pytest
from textual.coordinate import Coordinate
from textual.widgets import DataTable

from core.trace.sqlite_tracer import SqliteTracer
from tools.tui.app import TraceExplorerApp
from tools.tui.screens.candidate_ranking import CandidateRankingScreen
from tools.tui.screens.phase_steps import PhaseStepsScreen
from tools.tui.screens.step_detail import StepDetailScreen
from tools.tui.screens.turn_detail import TurnDetailScreen
from tools.tui.screens.turn_list import TurnListScreen


@pytest.fixture
def trace_db(tmp_path):
    """One full turn: every prefetch event_type once, two scored
    memory_item candidates (different scores, to verify ranking order),
    and the sync_turn tail."""
    db_path = str(tmp_path / "trace.db")
    tracer = SqliteTracer(db_path)
    tracer.connect()

    tracer.record(
        "topic_detection",
        {"message": "hello there", "current_topic_summary": None,
         "response": "CONTINUE", "result_type": "CONTINUE", "result_label": None},
        request_id="req1", session_id="sess1", duration_ms=5.0,
    )
    tracer.record(
        "embedding",
        {"purpose": "retrieval_query", "text": "hello there", "dimension": 768},
        request_id="req1", session_id="sess1", duration_ms=1.0,
    )
    tracer.record(
        "retrieval_candidate_scored",
        {"candidate_type": "memory_item", "candidate_id": 1, "semantic_raw": 0.3,
         "semantic_weighted": 0.12, "person_weighted": 0.2, "final_score": 0.32, "included": False},
        request_id="req1", session_id="sess1",
    )
    tracer.record(
        "retrieval_candidate_scored",
        {"candidate_type": "memory_item", "candidate_id": 2, "semantic_raw": 0.7,
         "semantic_weighted": 0.28, "person_weighted": 0.2, "final_score": 0.48, "included": True},
        request_id="req1", session_id="sess1",
    )
    tracer.record(
        "retrieval_candidate_scored",
        {"candidate_type": "topic", "candidate_id": 9, "semantic_raw": 0.4,
         "semantic_weighted": 0.16, "tag_overlap_raw": 0.0, "tag_overlap_weighted": 0.0,
         "person_weighted": 0.0, "final_score": 0.16, "included": True},
        request_id="req1", session_id="sess1",
    )
    tracer.record(
        "context_retrieval",
        {"message": "hello there", "peer_id": "user", "topic_label": None, "is_new_topic": False,
         "tags": [], "environment": {"situation": "conversation", "location": None,
                                      "time": None, "day_of_week": None},
         "facts": [{"category": "fitness", "key": "gym", "value": "FitZone"}],
         "related_topics": [{"id": 9, "summary": "About X", "tags": []}],
         "memory_items": [{"content": "note", "item_type": "essence",
                            "source_peer": "user", "subject_peer": "user"}]},
        request_id="req1", session_id="sess1", duration_ms=2.0,
    )
    tracer.record(
        "embedding",
        {"purpose": "memory_item", "text": "note", "dimension": 768},
        request_id="req1", session_id="sess1", duration_ms=1.0,
    )
    tracer.record(
        "extraction",
        {"user_message": "hello there", "ai_response": "hi", "response": "[]",
         "items": [{"content": "note", "item_type": "essence", "attribution_type": "self_statement"}]},
        request_id="req1", session_id="sess1",
    )
    tracer.record(
        "turn_complete",
        {"user_message": "hello there", "ai_response": "hi", "old_topic_id": None,
         "new_topic_id": None, "new_topic_label": None, "items_extracted": 1, "fact_items": 0},
        request_id="req1", session_id="sess1", duration_ms=10.0,
    )
    tracer.close()
    return db_path


class TestFullDrillDown:
    """One continuous walk through a -> g, matching how a user would
    actually navigate."""

    @pytest.mark.asyncio
    async def test_walk_turn_list_to_candidate_ranking(self, trace_db):
        app = TraceExplorerApp(db_path=trace_db, limit=10)
        async with app.run_test() as pilot:
            await pilot.pause()

            # a) turn list
            assert isinstance(app.screen, TurnListScreen)
            table = app.screen.query_one("#turns", DataTable)
            assert table.row_count == 1
            assert table.get_cell_at(Coordinate(0, 1)) == "hello there"

            # b) select the turn -> c) 2-phase view
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, TurnDetailScreen)
            table = app.screen.query_one("#phases", DataTable)
            assert table.row_count == 2
            assert table.get_cell_at(Coordinate(0, 0)) == "Prefetch"
            # topic_detection, embedding(retrieval_query), 3x retrieval_candidate_scored, context_retrieval
            assert table.get_cell_at(Coordinate(0, 1)) == "6"
            assert table.get_cell_at(Coordinate(1, 0)) == "Sync Turn"

            # d) select prefetch -> e) step list
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, PhaseStepsScreen)
            table = app.screen.query_one("#steps", DataTable)
            assert table.row_count == 6
            assert table.get_cell_at(Coordinate(0, 0)) == "topic_detection"
            assert "CONTINUE" in table.get_cell_at(Coordinate(0, 1))

            # select context_retrieval (last row) -> f) step detail
            table.cursor_coordinate = Coordinate(5, 0)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, StepDetailScreen)
            drilldown = app.screen.query_one("#drilldown", DataTable)
            assert drilldown.row_count == 2
            assert drilldown.get_cell_at(Coordinate(0, 0)) == "Related Topics"
            assert drilldown.get_cell_at(Coordinate(1, 0)) == "Memory Items"
            assert drilldown.get_cell_at(Coordinate(1, 1)) == "2"  # 2 scored memory_items

            # g) drill into memory items ranking
            drilldown.cursor_coordinate = Coordinate(1, 0)
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, CandidateRankingScreen)
            table = app.screen.query_one("#candidates", DataTable)
            assert table.row_count == 2
            # sorted by final_score desc: candidate #2 (0.48) before #1 (0.32)
            assert table.get_cell_at(Coordinate(0, 0)) == "2"
            assert table.get_cell_at(Coordinate(0, 1)) == "0.480"
            assert table.get_cell_at(Coordinate(1, 0)) == "1"

            # Esc walks back up the stack
            await pilot.press("escape")
            await pilot.pause()
            assert isinstance(app.screen, StepDetailScreen)


class TestTurnDetailScreen:
    @pytest.mark.asyncio
    async def test_empty_sync_turn_phase_is_not_selectable(self, tmp_path):
        """A turn where only prefetch has run yet (sync_turn hasn't landed)
        must show "no events" and not navigate anywhere when selected."""
        db_path = str(tmp_path / "trace.db")
        tracer = SqliteTracer(db_path)
        tracer.connect()
        tracer.record(
            "topic_detection",
            {"message": "hi", "result_type": "CONTINUE", "result_label": None},
            request_id="req1", session_id="sess1",
        )
        tracer.close()

        app = TraceExplorerApp(db_path=db_path, limit=10)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("enter")  # into TurnDetailScreen
            await pilot.pause()
            table = app.screen.query_one("#phases", DataTable)
            assert "no events" in table.get_cell_at(Coordinate(1, 0))

            table.cursor_coordinate = Coordinate(1, 0)
            await pilot.press("enter")
            await pilot.pause()
            # Still on TurnDetailScreen — nothing to select for an empty phase.
            assert isinstance(app.screen, TurnDetailScreen)
