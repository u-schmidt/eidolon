"""Composes storage + the operational-metrics/trace readers into a
single "everything that happened during this turn" view.

``request_id`` already threads through an entire turn today — minted
once at the top of ``Eidolon.store_turn`` (see core/__init__.py) and
passed into every downstream step, including whatever it conditionally
triggers (fact consolidation, topic close/summarize, relabel, cluster
consolidation). So building a turn's full picture is just: look up
whatever rows exist for that id across storage/metrics/trace — their
mere presence or absence already answers "did this step run."

Lives in ``tools/``, not ``core/`` — mirrors the precedent that
``tools/trace.py``'s ``TraceReader``/``TracePrinter`` (read + format
glue for the CLI trace tool) already live outside core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from core.interfaces import MetricsReader, StorageProvider, TraceReader
from tools.dashboard.phases import classify as _classify_phase

_PREVIEW_MAX_CHARS = 80

# Human-readable labels for the raw event_type (rich trace)/call_type
# (llm_calls)/step (events) identifiers that show up in a turn's step
# list — confirmed by grepping every tracer.record()/record_llm_call()/
# record_event() call site in core/. Deliberately a display-only map:
# the raw identifier stays on TraceStep.event_type/SkeletonStep.kind for
# anything that still needs it (e.g. the retrieval_scoring_summary
# template dispatch). Unmapped identifiers fall back to the raw string
# rather than disappearing — same principle tools/dashboard/phases.py's
# classify() already uses for its own unknown-event-type fallback.
_STEP_LABELS: dict[str, str] = {
    "topic_detection": "Detect topic change",
    "topic_relabel": "Relabel topic",
    "topic_relabel_trigger": "Check relabel trigger",
    "topic_summarization": "Summarize & close topic",
    "topic_summary_embedding": "Embed topic summary",
    "extraction": "Extract memory fragments",
    "retrieval_scoring_summary": "Score & retrieve candidates",
    "context_retrieval": "Assemble injected context",
    "fact_consolidation": "Consolidate facts",
    "fact_correction": "Manual fact correction",
    "cluster_consolidation": "Consolidate into cluster",
    "cluster_summary_embedding": "Embed cluster summary",
    "stale_topic_cleanup": "Stale-topic sweep",
    "memory_item_embedding": "Embed memory item",
    "turn_complete": "Turn complete (sync)",
    "prefetch_complete": "Prefetch complete",
    "error": "Error",
}

# "embedding"/embedding_call are recorded for 4 different purposes (see
# core/__init__.py, core/retrieval/associative_retriever.py,
# core/summarization/topic_summarizer.py,
# core/consolidation/cluster_consolidator.py) — without this, every
# embedding step in a turn's list looks identical.
_EMBEDDING_PURPOSE_LABELS: dict[str, str] = {
    "memory_item": "Embed memory item",
    "retrieval_query": "Embed retrieval query",
    "topic_summary": "Embed topic summary",
    "cluster_summary": "Embed cluster summary",
}


def _preview(text: str) -> str:
    text = text or ""
    return text if len(text) <= _PREVIEW_MAX_CHARS else text[:_PREVIEW_MAX_CHARS] + "…"


def _friendly_label(raw: str) -> str:
    return _STEP_LABELS.get(raw, raw)


def _embedding_label(purpose: str | None) -> str:
    if purpose in _EMBEDDING_PURPOSE_LABELS:
        return _EMBEDDING_PURPOSE_LABELS[purpose]
    return "Embed" if purpose is None else f"Embed ({purpose})"


def _trace_step_label(event_type: str, payload: dict) -> str:
    if event_type == "embedding":
        return _embedding_label(payload.get("purpose"))
    return _friendly_label(event_type)


@dataclass
class TurnListItem:
    """One row in the recent-turns picker."""

    request_id: str
    peer_id: str | None
    topic_id: int | None
    preview: str
    created_at: object


@dataclass
class TopicRef:
    id: int
    label: str
    summary: str


@dataclass
class TurnHeader:
    request_id: str
    peer_id: str | None
    session_id: str | None
    created_at: object
    user_message: str | None
    ai_response: str | None
    topic: TopicRef | None


@dataclass
class SkeletonStep:
    """A step reconstructed from the always-on metrics tables only —
    type/duration/success, no rich payload."""

    kind: str  # "event" | "llm_call" | "embedding_call"
    label: str  # step / call_type / purpose
    duration_ms: float | None
    succeeded: bool | None
    detail: str | None  # error message, when kind == "event"
    created_at: object


@dataclass
class TraceStep:
    """A step reconstructed from the debug-gated rich trace — full payload."""

    event_type: str
    payload: dict
    duration_ms: float | None
    created_at: object
    display_label: str
    phase: str  # PREFETCH | SYNC_TURN, via tools.dashboard.phases.classify()
    start_offset_ms: float | None  # ms from the turn's earliest timed step; None if unknown


@dataclass
class TurnDetail:
    header: TurnHeader
    rich_available: bool
    trace_steps: list[TraceStep] = field(default_factory=list)
    skeleton_steps: list[SkeletonStep] = field(default_factory=list)
    # Single authoritative numbers pulled from prefetch_complete/
    # turn_complete's own duration_ms — deliberately NOT a sum of the
    # phase's individual step durations (turn_complete's duration
    # already contains extraction's/embedding's, which are also traced
    # separately — summing would double-count). None when that wrapping
    # event never fired (older turns, or a turn traced before this
    # existed).
    prefetch_duration_ms: float | None = None
    sync_duration_ms: float | None = None
    # The real max(start_offset_ms + duration_ms) across every timed step —
    # the x-axis scale for the timing waterfall. Deliberately not derived
    # from prefetch_duration_ms + sync_duration_ms: either can be None
    # (an older turn, or one phase's wrapping event missing) while
    # individual steps still carry real timing, so this is computed
    # straight from the steps themselves to never under-scale the chart.
    timeline_total_ms: float | None = None
    # Wall-clock time Eidolon spent doing nothing between the two phases —
    # the window Hermes/the LLM owns (sync_turn() is called only after the
    # reply has already gone out, see adapters/hermes/memory_provider.py's
    # "Sync Turn (called ASYNC after each LLM response)" section comment).
    # Not a new measurement: algebraically equal to turn_complete's start
    # minus prefetch_complete's end, since prefetch_complete's own start
    # IS timeline_total_ms's anchor by construction (it wraps every
    # prefetch sub-step, so nothing starts earlier). Computed as a
    # subtraction of fields already above rather than re-deriving
    # timestamps. Clamped to 0.0 — never negative — to absorb timestamp-
    # rounding noise (DB column resolution vs. perf_counter() sub-ms
    # precision). None unless all three of prefetch_duration_ms/
    # sync_duration_ms/timeline_total_ms are present.
    llm_gap_ms: float | None = None
    # What the user actually waited for: prefetch (blocks Hermes's LLM
    # call — see prefetch()'s docstring) plus the AI-response gap above.
    # Deliberately excludes sync_duration_ms — store_turn() runs async,
    # AFTER the reply has already shipped (its own docstring: "Called
    # AFTER the LLM responds (async, does not block the turn)"), so
    # fact extraction/consolidation/etc. never delays the user even
    # though it's real Eidolon compute time. None unless llm_gap_ms is
    # (same availability as that field).
    user_wait_ms: float | None = None


class TurnReader:
    """Read-only composition over ``StorageProvider``, ``MetricsReader``,
    and ``TraceReader`` — the data-access layer behind the turn-detail
    dashboard (see ``tools/dashboard/app.py``).
    """

    def __init__(
        self,
        storage: StorageProvider,
        metrics_reader: MetricsReader,
        trace_reader: TraceReader,
    ) -> None:
        self._storage = storage
        self._metrics_reader = metrics_reader
        self._trace_reader = trace_reader

    def recent_turns(self, limit: int = 20, peer_id: str | None = None) -> list[TurnListItem]:
        """List recent turns for a picker UI, most recent first.

        Sourced straight from ``messages`` (via
        ``StorageProvider.get_recent_turns``) — always-on, no debug trace
        needed, unlike ``tools/trace.py``'s equivalent, which can only
        list turns the debug trace happened to capture.
        """
        return [
            TurnListItem(
                request_id=row["request_id"],
                peer_id=row["peer_id"],
                topic_id=row["topic_id"],
                preview=_preview(row["content"]),
                created_at=row["created_at"],
            )
            for row in self._storage.get_recent_turns(limit, peer_id=peer_id)
        ]

    def get(self, request_id: str) -> TurnDetail | None:
        """Build the full picture of one turn, or ``None`` if
        ``request_id`` matches no stored message (unknown turn — includes
        turns stored before the ``messages.request_id`` column existed).
        """
        messages = self._storage.get_messages_by_request_id(request_id)
        if not messages:
            return None

        user_row = next((m for m in messages if m["role"] == "user"), None)
        ai_row = next((m for m in messages if m["role"] == "ai"), None)
        topic_id = (user_row or messages[0]).get("topic_id")
        topic_ref = None
        if topic_id is not None:
            topic = self._storage.get_topic(topic_id)
            if topic is not None:
                topic_ref = TopicRef(id=topic["id"], label=topic["label"], summary=topic["summary"])

        header = TurnHeader(
            request_id=request_id,
            peer_id=(user_row or messages[0]).get("peer_id"),
            session_id=None,
            created_at=(user_row or messages[0]).get("created_at"),
            user_message=user_row["content"] if user_row else None,
            ai_response=ai_row["content"] if ai_row else None,
            topic=topic_ref,
        )

        trace_events = self._trace_reader.events_for_request(request_id)
        anchor = self._earliest_start(trace_events)
        trace_steps = [
            TraceStep(
                event_type=e["event_type"],
                payload=e["payload"],
                duration_ms=e["duration_ms"],
                created_at=e["created_at"],
                display_label=_trace_step_label(e["event_type"], e["payload"]),
                phase=_classify_phase(e),
                start_offset_ms=self._start_offset_ms(e, anchor),
            )
            for e in trace_events
        ]
        for step in trace_steps:
            if step.event_type == "retrieval_scoring_summary":
                self._annotate_candidate_labels(step.payload)

        # Single authoritative per-phase totals — see TurnDetail's
        # docstring comment for why this isn't a sum of step durations.
        prefetch_duration_ms = next(
            (s.duration_ms for s in trace_steps if s.event_type == "prefetch_complete"), None,
        )
        sync_duration_ms = next(
            (s.duration_ms for s in trace_steps if s.event_type == "turn_complete"), None,
        )
        timeline_total_ms = max(
            (s.start_offset_ms + s.duration_ms for s in trace_steps
             if s.start_offset_ms is not None and s.duration_ms is not None),
            default=None,
        )
        llm_gap_ms = None
        if prefetch_duration_ms is not None and sync_duration_ms is not None and timeline_total_ms is not None:
            llm_gap_ms = max(0.0, timeline_total_ms - prefetch_duration_ms - sync_duration_ms)
        user_wait_ms = None
        if prefetch_duration_ms is not None and llm_gap_ms is not None:
            user_wait_ms = prefetch_duration_ms + llm_gap_ms

        events = self._metrics_reader.events_for_request(request_id)
        llm_calls = self._metrics_reader.llm_calls_for_request(request_id)
        embedding_calls = self._metrics_reader.embedding_calls_for_request(request_id)
        header.session_id = next(
            (
                s["session_id"] for s in (*trace_events, *events, *llm_calls, *embedding_calls)
                if s["session_id"]
            ),
            None,
        )

        skeleton_steps = self._build_skeleton_steps(events, llm_calls, embedding_calls)

        return TurnDetail(
            header=header,
            rich_available=bool(trace_steps),
            trace_steps=trace_steps,
            skeleton_steps=skeleton_steps,
            prefetch_duration_ms=prefetch_duration_ms,
            sync_duration_ms=sync_duration_ms,
            timeline_total_ms=timeline_total_ms,
            llm_gap_ms=llm_gap_ms,
            user_wait_ms=user_wait_ms,
        )

    def _annotate_candidate_labels(self, payload: dict) -> None:
        """Mutate a ``retrieval_scoring_summary`` payload in place, adding a
        ``label`` key to each ``top_candidates`` entry — the candidate's
        topic/cluster label or a memory-item content preview, instead of a
        bare ``candidate_type``/``candidate_id`` pair.

        ``candidate_kind`` disambiguates ``cluster_candidate`` rows (which
        score both topics and clusters in one call — see
        ``core/stores/retrieval_scorer.py``'s ``finalize()``); every other
        row's own ``candidate_type`` is the kind directly.
        """
        for candidate in payload.get("top_candidates", []):
            kind = candidate.get("candidate_kind") or candidate.get("candidate_type")
            candidate["label"] = self._resolve_candidate_label(kind, candidate["candidate_id"])

    def _resolve_candidate_label(self, kind: str | None, candidate_id: int) -> str | None:
        if kind == "topic":
            topic = self._storage.get_topic(candidate_id)
            return topic["label"] if topic else None
        if kind == "cluster":
            cluster = self._storage.get_cluster(candidate_id)
            return cluster["label"] if cluster else None
        if kind == "memory_item":
            item = self._storage.get_memory_item(candidate_id)
            return _preview(item["content"]) if item else None
        return None

    @staticmethod
    def _parse_timestamp(value: object) -> datetime | None:
        """``created_at`` is a raw string for SqliteTraceReader rows and a
        real ``datetime`` for PostgresTraceReader rows — same dual-shape
        handling as ``tools/dashboard/topic_lifecycle_reader.py``'s
        ``_parse_timestamp``."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    @classmethod
    def _step_start(cls, event: dict) -> datetime | None:
        """A step's start time, derived from its (end) ``created_at`` minus
        ``duration_ms`` — ``created_at`` is stamped when ``tracer.record()``
        is called, which every instrumented call site does right after
        computing its own ``duration_ms`` via ``time.perf_counter()``, so
        it's effectively the step's end time. ``None`` when either piece
        is missing (e.g. an ``error`` event, which carries no duration)."""
        end = cls._parse_timestamp(event["created_at"])
        duration_ms = event["duration_ms"]
        if end is None or duration_ms is None:
            return None
        return end - timedelta(milliseconds=duration_ms)

    @classmethod
    def _earliest_start(cls, events: list[dict]) -> datetime | None:
        """The turn's timeline anchor — the earliest step start time
        among events that have one. ``None`` if no event carries timing
        (e.g. skeleton-only turns never reach this at all, and a rich
        turn with only untimed ``error`` events would too)."""
        starts = [s for s in (cls._step_start(e) for e in events) if s is not None]
        return min(starts) if starts else None

    @classmethod
    def _start_offset_ms(cls, event: dict, anchor: datetime | None) -> float | None:
        """Milliseconds from ``anchor`` to this event's own start —
        the x-axis position for the turn-detail waterfall. ``None`` when
        either the turn has no anchor at all, or this specific event has
        no start time of its own."""
        if anchor is None:
            return None
        start = cls._step_start(event)
        if start is None:
            return None
        return (start - anchor).total_seconds() * 1000

    @staticmethod
    def _build_skeleton_steps(events: list[dict], llm_calls: list[dict], embedding_calls: list[dict]) -> list[SkeletonStep]:
        steps: list[SkeletonStep] = []
        for e in events:
            steps.append(SkeletonStep(
                kind="event", label=_friendly_label(e["step"]), duration_ms=None,
                succeeded=False, detail=e["message"], created_at=e["created_at"],
            ))
        for c in llm_calls:
            steps.append(SkeletonStep(
                kind="llm_call", label=_friendly_label(c["call_type"]), duration_ms=c["duration_ms"],
                succeeded=c["succeeded"], detail=None, created_at=c["created_at"],
            ))
        for c in embedding_calls:
            steps.append(SkeletonStep(
                kind="embedding_call", label=_embedding_label(c["purpose"]), duration_ms=c["duration_ms"],
                succeeded=c["succeeded"], detail=None, created_at=c["created_at"],
            ))
        steps.sort(key=lambda s: str(s.created_at))
        return steps
