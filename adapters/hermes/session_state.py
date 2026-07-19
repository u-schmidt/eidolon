"""Per-session state for the Eidolon Hermes adapter.

Hermes gives each ``EidolonMemoryProvider`` instance its own lifecycle per
session (see ``agent_init.py``/``gateway/run.py``'s per-session agent
cache), but a single instance can still see more than one ``session_id``
over its lifetime (session resumption, branching, context-compression
handoffs). This container holds what the adapter needs to track per
``session_id`` so those handoffs — and, within one session, the handoff
between ``prefetch()`` and ``sync_turn()`` — stay isolated and correct.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.models import TopicChange


@dataclass
class SessionState:
    """State the Hermes adapter tracks for one Hermes ``session_id``.

    Attributes
    ----------
    active_topic_id : int | None
        The currently open topic for this session, as last known to the
        adapter (mirrors ``topics.status='active'`` in storage).
    pending_topic_change : TopicChange | None
        The topic decision computed by ``prefetch()`` for the turn that
        hasn't been synced yet. Consumed by ``sync_turn()`` instead of
        recomputing it, provided ``pending_query`` still matches.
    pending_query : str | None
        The user message ``pending_topic_change`` was computed for — lets
        ``sync_turn()`` verify the cached decision is actually for this
        turn before trusting it.
    """

    active_topic_id: int | None = None
    pending_topic_change: TopicChange | None = None
    pending_query: str | None = None
