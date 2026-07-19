#!/usr/bin/env python3
"""Reconstruct the exact injection string prefetch() returned for a turn.

Replays a turn's already-recorded context_retrieval trace event through
ContextFormatter — the same formatter prefetch() itself uses — so the
result is exactly what got handed to the LLM, without re-running
retrieval or needing a live Eidolon instance.

Usage:
    python tools/simulate_injection.py
    python tools/simulate_injection.py --request-id 92687c94
    python tools/simulate_injection.py --session <session_id>
    python tools/simulate_injection.py --db ~/.eidolon/trace.db
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure the project root is on sys.path for direct execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core.context_formatter import ContextFormatter
from core.models import ContextBundle, Fact, MemoryItem, RelatedTopic
from tools.trace import TraceReader


@dataclass
class SimulationResult:
    """The reconstructed injection string for one turn, plus its context."""

    request_id: str
    user_message: str
    injection: str


class InjectionSimulator:
    """Rebuilds the injection string prefetch() returned for a past turn
    from trace.db's context_retrieval event, via ContextFormatter — the
    same code path prefetch() itself uses, so this can never drift from
    what actually happened.
    """

    def __init__(self, reader: TraceReader) -> None:
        self._reader = reader

    def resolve_request_id(self, request_id: str | None, session_id: str | None) -> str | None:
        """Return `request_id` if given, else the most recent one available
        (optionally restricted to `session_id`)."""
        if request_id:
            return request_id
        recent = self._reader.recent_request_ids(session_id=session_id, limit=1)
        return recent[0] if recent else None

    def simulate(self, request_id: str) -> SimulationResult | None:
        """Return the reconstructed injection for `request_id`, or None if
        that turn has no context_retrieval event (prefetch never ran, or
        it errored before retrieval)."""
        events = self._reader.events_for_request(request_id)
        payload = next(
            (e["payload"] for e in events if e["event_type"] == "context_retrieval"), None
        )
        if payload is None:
            return None

        bundle = self._bundle_from_payload(payload)
        return SimulationResult(
            request_id=request_id,
            user_message=payload.get("message", ""),
            injection=ContextFormatter.format(bundle),
        )

    @staticmethod
    def _bundle_from_payload(payload: dict) -> ContextBundle:
        facts = [
            Fact(peer_id=payload.get("peer_id", ""), category=f["category"], key=f["key"],
                 value=f["value"], source="")
            for f in payload.get("facts", [])
        ]
        related_topics = [
            RelatedTopic(topic_id=t["id"], summary=t["summary"], tags=t.get("tags", []))
            for t in payload.get("related_topics", [])
        ]
        memory_items = [
            MemoryItem(content=mi["content"], item_type=mi["item_type"],
                       source_peer=mi["source_peer"], subject_peer=mi["subject_peer"])
            for mi in payload.get("memory_items", [])
        ]
        return ContextBundle(
            facts=facts,
            current_topic_label=payload.get("topic_label"),
            related_topics=related_topics,
            memory_items=memory_items,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct the exact injection string prefetch() returned for a turn",
    )
    parser.add_argument("--db", default="~/.eidolon/trace.db", help="Trace database path")
    parser.add_argument("--request-id", default=None, help="Specific request_id (default: most recent)")
    parser.add_argument("--session", default=None, help="Restrict 'most recent' lookup to one session_id")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        reader = TraceReader(args.db)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    simulator = InjectionSimulator(reader)
    request_id = simulator.resolve_request_id(args.request_id, args.session)
    if request_id is None:
        reader.close()
        print("No trace events found" + (f" for session {args.session}" if args.session else "") + ".")
        return

    result = simulator.simulate(request_id)
    reader.close()

    if result is None:
        print(f"No context_retrieval event for request_id={request_id} "
              f"(prefetch may not have run, or nothing was retrieved).")
        return

    print(f"request_id: {result.request_id}")
    print(f"user message: {result.user_message!r}")
    print()
    print("=" * 70)
    print("Injection string returned to prefetch():")
    print("=" * 70)
    print(result.injection if result.injection else "(empty string)")
    print("=" * 70)


if __name__ == "__main__":
    main()
