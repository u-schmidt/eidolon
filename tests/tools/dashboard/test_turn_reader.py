"""Unit tests for TurnReader — the composition layer behind the
turn-detail dashboard. Uses MockStorage plus simple in-memory fakes for
MetricsReader/TraceReader rather than real backends (that round-trip
coverage lives in tests/core/test_metrics_reader_contract.py and
tests/core/trace/test_*_trace_reader.py)."""

from __future__ import annotations

from core.models import MemoryItem
from tests.mocks import MockStorage
from tools.dashboard.turn_reader import TurnReader


class FakeMetricsReader:
    def __init__(self) -> None:
        self.events: dict[str, list[dict]] = {}
        self.llm_calls: dict[str, list[dict]] = {}
        self.embedding_calls: dict[str, list[dict]] = {}

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def events_for_request(self, request_id: str) -> list[dict]:
        return self.events.get(request_id, [])

    def llm_calls_for_request(self, request_id: str) -> list[dict]:
        return self.llm_calls.get(request_id, [])

    def embedding_calls_for_request(self, request_id: str) -> list[dict]:
        return self.embedding_calls.get(request_id, [])

    def events_since(self, since):
        return []

    def llm_calls_since(self, since):
        return []

    def embedding_calls_since(self, since):
        return []


class FakeTraceReader:
    def __init__(self) -> None:
        self.events: dict[str, list[dict]] = {}

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def events_for_request(self, request_id: str) -> list[dict]:
        return self.events.get(request_id, [])


def _build_turn_reader() -> tuple[TurnReader, MockStorage, FakeMetricsReader, FakeTraceReader]:
    storage = MockStorage()
    metrics = FakeMetricsReader()
    trace = FakeTraceReader()
    return TurnReader(storage, metrics, trace), storage, metrics, trace


class TestGet:
    def test_unknown_request_id_returns_none(self):
        reader, _storage, _metrics, _trace = _build_turn_reader()
        assert reader.get("nope") is None

    def test_skeleton_mode_when_no_trace_data(self):
        reader, storage, metrics, _trace = _build_turn_reader()
        tid = storage.create_topic("Gym talk", workspace_id="u1")
        storage.store_message(turn_id=1, role="user", content="gym was crowded", workspace_id="u1", topic_id=tid, request_id="req1")
        storage.store_message(turn_id=1, role="ai", content="sounds rough", workspace_id="ai", topic_id=tid, request_id="req1")
        metrics.llm_calls["req1"] = [
            {"call_type": "topic_detection", "duration_ms": 42.5, "succeeded": True,
             "workspace_id": "u1", "session_id": None, "created_at": "t1"},
        ]
        metrics.events["req1"] = [
            {"severity": "error", "step": "extraction", "message": "boom",
             "workspace_id": "u1", "session_id": None, "created_at": "t2"},
        ]

        detail = reader.get("req1")
        assert detail is not None
        assert detail.rich_available is False
        assert detail.trace_steps == []
        assert detail.header.user_message == "gym was crowded"
        assert detail.header.ai_response == "sounds rough"
        assert detail.header.topic is not None
        assert detail.header.topic.label == "Gym talk"

        kinds = {(s.kind, s.label) for s in detail.skeleton_steps}
        assert ("llm_call", "Detect topic change") in kinds
        assert ("event", "Extract memory fragments") in kinds

    def test_rich_mode_when_trace_data_present(self):
        reader, storage, metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        storage.store_message(turn_id=1, role="ai", content="hello", workspace_id="ai", request_id="req1")
        trace.events["req1"] = [
            {"event_type": "topic_detection", "payload": {"message": "hi"}, "duration_ms": 5.0,
             "session_id": "sess1", "created_at": "t1"},
        ]

        detail = reader.get("req1")
        assert detail is not None
        assert detail.rich_available is True
        assert len(detail.trace_steps) == 1
        assert detail.trace_steps[0].event_type == "topic_detection"
        assert detail.header.session_id == "sess1"

    def test_turn_without_topic_id_has_no_topic_ref(self):
        reader, storage, _metrics, _trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")

        detail = reader.get("req1")
        assert detail is not None
        assert detail.header.topic is None


class TestTimingWaterfall:
    def test_phase_classification_and_offsets(self):
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        trace.events["req1"] = [
            {
                "event_type": "prefetch_complete", "payload": {}, "duration_ms": 1000.0,
                "session_id": None, "created_at": "2026-01-01T00:00:01.000000",
            },
            {
                "event_type": "turn_complete", "payload": {}, "duration_ms": 500.0,
                "session_id": None, "created_at": "2026-01-01T00:00:03.000000",
            },
        ]

        detail = reader.get("req1")
        assert detail is not None
        prefetch_step = next(s for s in detail.trace_steps if s.event_type == "prefetch_complete")
        sync_step = next(s for s in detail.trace_steps if s.event_type == "turn_complete")

        assert prefetch_step.phase == "prefetch"
        assert sync_step.phase == "sync_turn"
        # prefetch_complete starts earliest (00:00:00.000) -> offset 0;
        # turn_complete starts at 00:00:02.500 -> 2500ms after the anchor.
        assert prefetch_step.start_offset_ms == 0.0
        assert sync_step.start_offset_ms == 2500.0
        # turn_complete: starts at +2500ms, runs 500ms -> ends at +3000ms,
        # the latest end of any step, so that's the timeline scale.
        assert detail.timeline_total_ms == 3000.0

    def test_start_offset_none_when_created_at_unparseable(self):
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        trace.events["req1"] = [
            {"event_type": "topic_detection", "payload": {}, "duration_ms": 5.0,
             "session_id": None, "created_at": "not-a-timestamp"},
        ]

        detail = reader.get("req1")
        assert detail is not None
        assert detail.trace_steps[0].start_offset_ms is None

    def test_prefetch_and_sync_duration_pulled_from_wrapping_events(self):
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        trace.events["req1"] = [
            {"event_type": "prefetch_complete", "payload": {}, "duration_ms": 120.0,
             "session_id": None, "created_at": "t1"},
            {"event_type": "extraction", "payload": {}, "duration_ms": 999.0,
             "session_id": None, "created_at": "t2"},
            {"event_type": "turn_complete", "payload": {}, "duration_ms": 80.0,
             "session_id": None, "created_at": "t3"},
        ]

        detail = reader.get("req1")
        assert detail is not None
        assert detail.prefetch_duration_ms == 120.0
        assert detail.sync_duration_ms == 80.0

    def test_prefetch_and_sync_duration_none_when_wrapping_events_absent(self):
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        trace.events["req1"] = [
            {"event_type": "topic_detection", "payload": {}, "duration_ms": 5.0,
             "session_id": None, "created_at": "t1"},
        ]

        detail = reader.get("req1")
        assert detail is not None
        assert detail.prefetch_duration_ms is None
        assert detail.sync_duration_ms is None

    def test_llm_gap_computed_from_wrapping_events(self):
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        trace.events["req1"] = [
            {"event_type": "prefetch_complete", "payload": {}, "duration_ms": 1000.0,
             "session_id": None, "created_at": "2026-01-01T00:00:01.000000"},
            {"event_type": "turn_complete", "payload": {}, "duration_ms": 500.0,
             "session_id": None, "created_at": "2026-01-01T00:00:03.000000"},
        ]

        detail = reader.get("req1")
        assert detail is not None
        # prefetch: 0 -> 1000ms. turn_complete: starts +2500ms, ends +3000ms.
        # Gap is the idle window between them: 2500 - 1000 = 1500ms.
        assert detail.llm_gap_ms == 1500.0

    def test_llm_gap_none_when_prefetch_duration_missing(self):
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        trace.events["req1"] = [
            {"event_type": "turn_complete", "payload": {}, "duration_ms": 80.0,
             "session_id": None, "created_at": "t1"},
        ]

        detail = reader.get("req1")
        assert detail is not None
        assert detail.llm_gap_ms is None

    def test_llm_gap_none_when_sync_duration_missing(self):
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        trace.events["req1"] = [
            {"event_type": "prefetch_complete", "payload": {}, "duration_ms": 120.0,
             "session_id": None, "created_at": "t1"},
        ]

        detail = reader.get("req1")
        assert detail is not None
        assert detail.llm_gap_ms is None

    def test_llm_gap_clamped_to_zero_for_overlapping_timestamps(self):
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        # Pathological: turn_complete's own start (00:00:00.000) precedes
        # prefetch_complete's end (00:00:01.000) — plain subtraction would
        # go negative; llm_gap_ms must clamp to 0.0 rather than show
        # negative "AI response" time.
        trace.events["req1"] = [
            {"event_type": "prefetch_complete", "payload": {}, "duration_ms": 1000.0,
             "session_id": None, "created_at": "2026-01-01T00:00:01.000000"},
            {"event_type": "turn_complete", "payload": {}, "duration_ms": 500.0,
             "session_id": None, "created_at": "2026-01-01T00:00:00.500000"},
        ]

        detail = reader.get("req1")
        assert detail is not None
        assert detail.timeline_total_ms == 1000.0
        assert detail.llm_gap_ms == 0.0

    def test_user_wait_is_prefetch_plus_gap_excluding_sync(self):
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        trace.events["req1"] = [
            {"event_type": "prefetch_complete", "payload": {}, "duration_ms": 1000.0,
             "session_id": None, "created_at": "2026-01-01T00:00:01.000000"},
            {"event_type": "turn_complete", "payload": {}, "duration_ms": 500.0,
             "session_id": None, "created_at": "2026-01-01T00:00:03.000000"},
        ]

        detail = reader.get("req1")
        assert detail is not None
        # prefetch (1000ms) + gap (1500ms) = 2500ms — the 500ms sync
        # duration is excluded, since store_turn() runs async after the
        # reply already shipped and never delays the user.
        assert detail.user_wait_ms == 2500.0

    def test_user_wait_none_when_gap_unavailable(self):
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        trace.events["req1"] = [
            {"event_type": "prefetch_complete", "payload": {}, "duration_ms": 120.0,
             "session_id": None, "created_at": "t1"},
        ]

        detail = reader.get("req1")
        assert detail is not None
        assert detail.llm_gap_ms is None
        assert detail.user_wait_ms is None


class TestCandidateLabels:
    """Retrieval-scoring-summary candidates should resolve to a human
    label/content preview, not just a bare candidate_type/candidate_id —
    see TurnReader._annotate_candidate_labels."""

    def test_topic_candidate_gets_label(self):
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        topic_id = storage.create_topic("Bello Features", summary="")
        trace.events["req1"] = [{
            "event_type": "retrieval_scoring_summary",
            "payload": {
                "candidate_type": "topic", "total_scored": 1, "included_count": 1,
                "top_candidates": [{"candidate_type": "topic", "candidate_id": topic_id}],
            },
            "duration_ms": 1.0, "session_id": None, "created_at": "t1",
        }]

        detail = reader.get("req1")
        candidate = detail.trace_steps[0].payload["top_candidates"][0]
        assert candidate["label"] == "Bello Features"

    def test_cluster_candidate_gets_label(self):
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        cluster_id = storage.create_cluster(workspace_id="u1", label="Fitness", summary="")
        trace.events["req1"] = [{
            "event_type": "retrieval_scoring_summary",
            "payload": {
                "candidate_type": "cluster", "total_scored": 1, "included_count": 1,
                "top_candidates": [{"candidate_type": "cluster", "candidate_id": cluster_id}],
            },
            "duration_ms": 1.0, "session_id": None, "created_at": "t1",
        }]

        detail = reader.get("req1")
        candidate = detail.trace_steps[0].payload["top_candidates"][0]
        assert candidate["label"] == "Fitness"

    def test_memory_item_candidate_gets_content_preview(self):
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        item_id = storage.store_memory_item(MemoryItem(
            content="User likes hiking", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
        ))
        trace.events["req1"] = [{
            "event_type": "retrieval_scoring_summary",
            "payload": {
                "candidate_type": "memory_item", "total_scored": 1, "included_count": 1,
                "top_candidates": [{"candidate_type": "memory_item", "candidate_id": item_id}],
            },
            "duration_ms": 1.0, "session_id": None, "created_at": "t1",
        }]

        detail = reader.get("req1")
        candidate = detail.trace_steps[0].payload["top_candidates"][0]
        assert candidate["label"] == "User likes hiking"

    def test_cluster_candidate_kind_overrides_candidate_type(self):
        """search_cluster_candidates scores both topics and clusters under
        candidate_type="cluster_candidate" — candidate_kind disambiguates
        which one each row actually is (see retrieval_scorer.py)."""
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        topic_id = storage.create_topic("Gym talk", summary="")
        trace.events["req1"] = [{
            "event_type": "retrieval_scoring_summary",
            "payload": {
                "candidate_type": "cluster_candidate", "total_scored": 1, "included_count": 1,
                "top_candidates": [
                    {"candidate_type": "cluster_candidate", "candidate_kind": "topic", "candidate_id": topic_id},
                ],
            },
            "duration_ms": 1.0, "session_id": None, "created_at": "t1",
        }]

        detail = reader.get("req1")
        candidate = detail.trace_steps[0].payload["top_candidates"][0]
        assert candidate["label"] == "Gym talk"

    def test_missing_candidate_resolves_to_none(self):
        reader, storage, _metrics, trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="hi", workspace_id="u1", request_id="req1")
        trace.events["req1"] = [{
            "event_type": "retrieval_scoring_summary",
            "payload": {
                "candidate_type": "topic", "total_scored": 1, "included_count": 0,
                "top_candidates": [{"candidate_type": "topic", "candidate_id": 999}],
            },
            "duration_ms": 1.0, "session_id": None, "created_at": "t1",
        }]

        detail = reader.get("req1")
        candidate = detail.trace_steps[0].payload["top_candidates"][0]
        assert candidate["label"] is None


class TestRecentTurns:
    def test_lists_stored_turns(self):
        reader, storage, _metrics, _trace = _build_turn_reader()
        storage.store_message(turn_id=1, role="user", content="first", workspace_id="u1", request_id="req1")
        storage.store_message(turn_id=2, role="user", content="second", workspace_id="u1", request_id="req2")

        turns = reader.recent_turns(10)
        assert [t.request_id for t in turns] == ["req2", "req1"]

    def test_preview_is_truncated(self):
        reader, storage, _metrics, _trace = _build_turn_reader()
        long_message = "x" * 200
        storage.store_message(turn_id=1, role="user", content=long_message, workspace_id="u1", request_id="req1")

        turns = reader.recent_turns(10)
        assert len(turns[0].preview) <= 81  # 80 chars + ellipsis
        assert turns[0].preview.endswith("…")
