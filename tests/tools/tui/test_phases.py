"""Tests for tools/tui/phases.py — turn -> (prefetch, sync_turn) split."""

from __future__ import annotations

from tools.tui.phases import PREFETCH, SYNC_TURN, classify, split_by_phase


def _event(event_type: str, **payload) -> dict:
    return {"event_type": event_type, "payload": payload}


class TestClassify:
    def test_topic_detection_is_prefetch(self):
        assert classify(_event("topic_detection")) == PREFETCH

    def test_retrieval_candidate_scored_is_prefetch(self):
        assert classify(_event("retrieval_candidate_scored")) == PREFETCH

    def test_context_retrieval_is_prefetch(self):
        assert classify(_event("context_retrieval")) == PREFETCH

    def test_embedding_retrieval_query_is_prefetch(self):
        assert classify(_event("embedding", purpose="retrieval_query")) == PREFETCH

    def test_embedding_memory_item_is_sync_turn(self):
        assert classify(_event("embedding", purpose="memory_item")) == SYNC_TURN

    def test_embedding_topic_label_is_sync_turn(self):
        assert classify(_event("embedding", purpose="topic_label")) == SYNC_TURN

    def test_embedding_topic_summary_is_sync_turn(self):
        assert classify(_event("embedding", purpose="topic_summary")) == SYNC_TURN

    def test_extraction_is_sync_turn(self):
        assert classify(_event("extraction")) == SYNC_TURN

    def test_fact_consolidation_is_sync_turn(self):
        assert classify(_event("fact_consolidation")) == SYNC_TURN

    def test_fact_correction_is_sync_turn(self):
        assert classify(_event("fact_correction")) == SYNC_TURN

    def test_topic_summarization_is_sync_turn(self):
        assert classify(_event("topic_summarization")) == SYNC_TURN

    def test_turn_complete_is_sync_turn(self):
        assert classify(_event("turn_complete")) == SYNC_TURN

    def test_error_is_sync_turn(self):
        assert classify(_event("error")) == SYNC_TURN

    def test_unknown_event_type_defaults_to_sync_turn(self):
        assert classify(_event("something_new")) == SYNC_TURN


class TestSplitByPhase:
    def test_splits_a_full_turn(self):
        events = [
            _event("topic_detection"),
            _event("embedding", purpose="retrieval_query"),
            _event("retrieval_candidate_scored"),
            _event("context_retrieval"),
            _event("embedding", purpose="memory_item"),
            _event("extraction"),
            _event("turn_complete"),
        ]
        prefetch_events, sync_turn_events = split_by_phase(events)

        assert [e["event_type"] for e in prefetch_events] == [
            "topic_detection", "embedding", "retrieval_candidate_scored", "context_retrieval",
        ]
        assert [e["event_type"] for e in sync_turn_events] == [
            "embedding", "extraction", "turn_complete",
        ]

    def test_every_event_is_bucketed_exactly_once(self):
        events = [_event("topic_detection"), _event("turn_complete"), _event("error")]
        prefetch_events, sync_turn_events = split_by_phase(events)
        assert len(prefetch_events) + len(sync_turn_events) == len(events)

    def test_empty_events_list(self):
        assert split_by_phase([]) == ([], [])
