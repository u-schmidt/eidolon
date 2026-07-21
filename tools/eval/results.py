"""Eval run results — the parsed outcome of playing an EvalScript.

Pure data containers, mirroring the input schema in ``tools/eval/models.py``.
Rendering lives separately in ``tools/eval/report_formatter.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.models import MemoryItem
from tools.eval.models import FactExpectation, MemoryItemHint, TopicHint


@dataclass
class MemoryItemMatch:
    """A tier-2 hint paired with its best-matching actual item, if any."""

    hint: MemoryItemHint
    score: float
    matched_item: MemoryItem | None
    type_match: bool


@dataclass
class TurnResult:
    """Scored outcome of one turn."""

    turn_number: int
    user_message: str
    topic_change_actual: str
    topic_change_expected: str | None = None
    topic_change_pass: bool | None = None  # None when not asserted
    topic_label_hint: str | None = None
    topic_label_actual: str | None = None
    memory_item_matches: list[MemoryItemMatch] = field(default_factory=list)
    unmatched_actual_items: list[MemoryItem] = field(default_factory=list)


@dataclass
class FactCheckResult:
    """Whether one expected fact was found in the consolidated profile."""

    expected: FactExpectation
    found: bool


@dataclass
class EndResult:
    """Scored outcome of the script's end-of-run expectations."""

    fact_results: list[FactCheckResult] = field(default_factory=list)
    expected_topics: list[TopicHint] = field(default_factory=list)
    actual_topics: list[dict] = field(default_factory=list)


@dataclass
class EvalReport:
    """The full outcome of one eval run — one script against one
    configured model/embedder pair."""

    script_name: str
    peer_id: str
    model_identity: str
    embedder_identity: str
    db_path: str
    turn_results: list[TurnResult] = field(default_factory=list)
    end_result: EndResult = field(default_factory=EndResult)
