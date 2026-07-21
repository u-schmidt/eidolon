"""Eval script schema — the parsed shape of a scripted golden conversation.

Pure data containers — no logic, no provider dependencies. Mirrors the
grouping convention of ``core/models.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryItemHint:
    """One expected memory item for a turn — tier 2, scored by embedding
    similarity against the turn's actual extracted items, never asserted
    exactly (see ``tools/eval/scorer.py``)."""

    item_type: str
    content_hint: str
    subject_peer: str | None = None


@dataclass
class TurnExpectation:
    """What a script author expects from a single turn.

    ``topic_change`` is tier 1 (hard assert) when set; ``None`` means
    "don't check this turn's topic decision". ``topic_label_hint`` and
    ``memory_items`` are tier 2 — reported, not asserted.
    """

    topic_change: str | None = None  # "CONTINUE" | "NEW_TOPIC" | None
    topic_label_hint: str | None = None
    memory_items: list[MemoryItemHint] = field(default_factory=list)


@dataclass
class EvalTurn:
    """One scripted turn: a fake user message and a fake AI reply, fed to
    ``store_turn`` exactly as an adapter would after the real LLM responds."""

    user: str
    ai: str
    expect: TurnExpectation = field(default_factory=TurnExpectation)


@dataclass
class FactExpectation:
    """A fact expected in the peer's consolidated profile at script end —
    tier 1, checked against ``storage.get_facts(peer_id)``."""

    category: str
    key: str
    value: str


@dataclass
class TopicHint:
    """A topic expected to exist at script end — tier 2, report-only."""

    label_hint: str


@dataclass
class EndExpectation:
    """What a script author expects once every turn has been played."""

    facts: list[FactExpectation] = field(default_factory=list)
    topics: list[TopicHint] = field(default_factory=list)


@dataclass
class EvalScript:
    """A whole scripted conversation for one (throwaway) peer."""

    name: str
    peer_id: str
    turns: list[EvalTurn]
    language: str = "en"
    expect_end: EndExpectation = field(default_factory=EndExpectation)
