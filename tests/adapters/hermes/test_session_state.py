"""Tests for SessionState."""

from __future__ import annotations

from adapters.hermes.session_state import SessionState
from core.models import TopicChange


class TestSessionState:
    def test_defaults(self):
        state = SessionState()
        assert state.active_topic_id is None
        assert state.pending_topic_change is None
        assert state.pending_query is None
        assert state.turn_number == 0

    def test_holds_pending_topic_change(self):
        change = TopicChange(type="NEW_TOPIC", label="Test Topic")
        state = SessionState(
            active_topic_id=7, pending_topic_change=change, pending_query="hello"
        )
        assert state.active_topic_id == 7
        assert state.pending_topic_change is change
        assert state.pending_query == "hello"

    def test_holds_turn_number(self):
        state = SessionState(turn_number=12)
        assert state.turn_number == 12
