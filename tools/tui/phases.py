"""Splits one turn's trace events into its two phases.

A turn's events all share one request_id (see the request_id parameter
on ``core/__init__.py``'s public methods) but come from two separate
calls in time: prefetch() before the LLM answers, sync_turn() after.
There's no separate correlation field for which phase an event belongs
to — it doesn't need one, since the two phases produce disjoint, static
sets of event_types. This module is that static classification.
"""

from __future__ import annotations

PREFETCH = "prefetch"
SYNC_TURN = "sync_turn"

_PREFETCH_EVENT_TYPES = {
    "topic_detection", "retrieval_scoring_summary", "context_retrieval", "prefetch_complete",
}
_SYNC_TURN_EVENT_TYPES = {
    "extraction", "fact_consolidation", "fact_correction", "topic_summarization",
    "turn_complete", "error",
}

# "embedding" events happen in both phases — purpose disambiguates.
_PREFETCH_EMBEDDING_PURPOSES = {"retrieval_query"}


def classify(event: dict) -> str:
    """Return PREFETCH or SYNC_TURN for one trace event."""
    event_type = event["event_type"]
    if event_type == "embedding":
        purpose = event["payload"].get("purpose")
        return PREFETCH if purpose in _PREFETCH_EMBEDDING_PURPOSES else SYNC_TURN
    if event_type in _PREFETCH_EVENT_TYPES:
        return PREFETCH
    if event_type in _SYNC_TURN_EVENT_TYPES:
        return SYNC_TURN
    return SYNC_TURN  # unknown event types default to sync_turn, never silently dropped


def split_by_phase(events: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split a turn's events into (prefetch_events, sync_turn_events)."""
    prefetch_events: list[dict] = []
    sync_turn_events: list[dict] = []
    for event in events:
        if classify(event) == PREFETCH:
            prefetch_events.append(event)
        else:
            sync_turn_events.append(event)
    return prefetch_events, sync_turn_events
