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
    pending_request_id : str | None
        The trace ``request_id`` ``prefetch()`` minted for this turn's
        ``detect_topic()``/``retrieve_context()`` calls. Reused by
        ``sync_turn()``'s ``store_turn()`` call (under the same
        ``pending_query`` validity check) so a whole turn's trace events
        share one id instead of three unrelated ones.
    turn_number : int
        Hermes' own per-agent turn counter (``agent._user_turn_count``,
        handed to us via ``MemoryProvider.on_turn_start()``), cached here
        so ``sync_turn()`` can use it as ``messages.turn_id``. Not derived
        ourselves — Hermes already hydrates it correctly from conversation
        history on a resumed session, which we'd otherwise have to
        reimplement. Defaults to 0, matching a fresh SessionState.

        Why this lives on SessionState rather than being read straight
        off ``on_turn_start``'s ``turn_number`` argument in ``sync_turn()``:
        Hermes' own counter is scoped to the *agent object*, not the
        logical conversation — it does NOT reset on ``/reset``/``/new``
        (confirmed by reading Hermes' source: no call site resets
        ``agent._user_turn_count``, only ``on_session_switch(reset=True)``
        fires, and that's documented as the providers' own cue to reset
        *their* copy). So we cache Hermes' value here and let
        ``on_session_switch(reset=True)`` blow this whole ``SessionState``
        away (see ``EidolonMemoryProvider.on_session_switch``) instead of
        trusting Hermes' raw number to already be reset — otherwise a
        genuinely new conversation could start at turn_id 48 instead of 1,
        and — since ``get_memory_items_by_turn`` filters on turn_id alone,
        not per-topic — collide with an unrelated older topic that also
        happened to reach turn 48 before the reset.

        Known blind spot: ``on_turn_start()`` is called with no
        ``session_id`` at all (checked against the live Hermes source —
        neither a parameter nor a kwarg), so
        ``EidolonMemoryProvider.on_turn_start()`` has to attribute the
        call to ``self._current_session_id`` rather than being told
        directly. That's safe as long as Hermes only ever runs one
        session's turns at a time through this provider instance — true
        for every switch path in the current source (resume/branch/reset
        all call ``on_session_switch()`` before the next turn's
        ``on_turn_start()``, never interleaved) — but would misattribute
        a turn number to the wrong session if Hermes ever ran two
        sessions' turns concurrently through one provider instance. If
        turn ids start looking shuffled between concurrent sessions,
        this assumption is the first thing to re-check.
    """

    active_topic_id: int | None = None
    pending_topic_change: TopicChange | None = None
    pending_query: str | None = None
    pending_request_id: str | None = None
    turn_number: int = 0
