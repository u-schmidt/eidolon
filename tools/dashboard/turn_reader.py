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

from core.interfaces import MetricsReader, StorageProvider, TraceReader

_PREVIEW_MAX_CHARS = 80


def _preview(text: str) -> str:
    text = text or ""
    return text if len(text) <= _PREVIEW_MAX_CHARS else text[:_PREVIEW_MAX_CHARS] + "…"


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


@dataclass
class TurnDetail:
    header: TurnHeader
    rich_available: bool
    trace_steps: list[TraceStep] = field(default_factory=list)
    skeleton_steps: list[SkeletonStep] = field(default_factory=list)


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
        trace_steps = [
            TraceStep(
                event_type=e["event_type"],
                payload=e["payload"],
                duration_ms=e["duration_ms"],
                created_at=e["created_at"],
            )
            for e in trace_events
        ]

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
        )

    @staticmethod
    def _build_skeleton_steps(events: list[dict], llm_calls: list[dict], embedding_calls: list[dict]) -> list[SkeletonStep]:
        steps: list[SkeletonStep] = []
        for e in events:
            steps.append(SkeletonStep(
                kind="event", label=e["step"], duration_ms=None,
                succeeded=False, detail=e["message"], created_at=e["created_at"],
            ))
        for c in llm_calls:
            steps.append(SkeletonStep(
                kind="llm_call", label=c["call_type"], duration_ms=c["duration_ms"],
                succeeded=c["succeeded"], detail=None, created_at=c["created_at"],
            ))
        for c in embedding_calls:
            steps.append(SkeletonStep(
                kind="embedding_call", label=c["purpose"], duration_ms=c["duration_ms"],
                succeeded=c["succeeded"], detail=None, created_at=c["created_at"],
            ))
        steps.sort(key=lambda s: str(s.created_at))
        return steps
