"""Eidolon core data types.

All dataclasses used across the core library.
Pure data containers — no logic, no provider dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class TopicChange:
    """Result of topic detection for a single user message."""

    type: Literal["CONTINUE", "NEW_TOPIC"]
    label: str | None = None  # required when type == "NEW_TOPIC"


@dataclass
class EnvironmentSnapshot:
    """Situational context at a point in time.

    Captures where/when/how the conversation is happening.
    Refreshed each turn by the adapter.
    """

    location: str | None = None
    time: datetime | None = None
    day_of_week: str | None = None
    situation: str | None = None  # e.g. "at home", "at cinema"
    rules: list[str] = field(default_factory=list)  # e.g. ["don't phone"]


@dataclass
class MemoryItem:
    """A single extracted fragment from a conversation turn.

    Never stores the full transcript — only essence, facts, surprises, or feelings.
    Every item carries perspective attribution (source + subject).
    """

    content: str
    item_type: Literal["essence", "fact", "surprise", "feeling"] = "essence"
    source_peer: str = ""
    subject_peer: str = ""
    attribution_type: str = ""
    id: int | None = None
    topic_id: int | None = None
    message_id: int | None = None
    embedding: list[float] | None = None
    timestamp: datetime | None = None
    context: EnvironmentSnapshot | None = None  # V2


@dataclass
class MemoryItemBudget:
    """Per-item-type slot budget for one ``search_memory_items`` call.

    Reserved slots per type, plus a free pool open to any type (including
    types with no reservation, e.g. "essence" — it's redundant with the
    always-shown topic summary, so it only competes for free slots). An
    empty/exhausted reserved tier's leftover capacity rolls into the free
    pool rather than going unused — see
    ``RetrievalScorer.finalize_tiered``.
    """

    fact: int = 4
    surprise: int = 1
    feeling: int = 1
    free: int = 4

    @property
    def total(self) -> int:
        return self.fact + self.surprise + self.feeling + self.free


@dataclass
class RelatedTopic:
    """A past topic related to the current conversation."""

    topic_id: int
    summary: str
    tags: list[str] = field(default_factory=list)
    cluster_label: str | None = None
    timestamp: datetime | None = None
    label: str | None = None


@dataclass
class ClusterSummary:
    """A consolidated narrative for a recurring topic thread.

    Built by ``ClusterConsolidator`` from two or more related topics
    that would otherwise compete individually in retrieval — see
    ``core/consolidation/cluster_consolidator.py``. ``summary`` is a
    short, present-tense "current state" snapshot, rewritten fresh each
    consolidation pass rather than accumulated; ``milestones`` is a
    small, curated list of the thread's genuine directional shifts over
    time (e.g. "started as an image viewer", "pivoted to a generation
    helper") — kept separate so the current state never has to compress
    away the thread's history to stay short.
    """

    cluster_id: int
    label: str
    summary: str
    topic_count: int
    milestones: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    recent_topics: list[RelatedTopic] = field(default_factory=list)


@dataclass
class Fact:
    """A structured, near-constant piece of information about a workspace.

    ``entity_name`` is blank for an ordinary workspace-level fact (the
    ``facts`` table) and set to a lowercase named-entity string for a
    per-entity profile card fact (the ``entity_facts`` table) — see
    ``core/consolidation/entity_profile_consolidator.py``. One dataclass
    for both rather than a parallel ``EntityFact`` type since the shape
    is otherwise identical.
    """

    workspace_id: str
    category: str
    key: str
    value: str
    source: str  # who contributed this fact
    importance: str = "medium"  # "low" | "medium" | "high"
    created_at: datetime | None = None
    entity_name: str = ""


@dataclass
class ContextBundle:
    """Everything the core returns for prompt injection.

    This is an *offer* — the LLM can use it or ignore it.
    """

    facts: list[Fact] = field(default_factory=list)
    environment: EnvironmentSnapshot | None = None
    current_topic_label: str | None = None
    related_topics: list[RelatedTopic] = field(default_factory=list)
    recent_topics: list[RelatedTopic] = field(default_factory=list)
    cluster_summaries: list[ClusterSummary] = field(default_factory=list)
    memory_items: list[MemoryItem] = field(default_factory=list)
    entity_facts: list[Fact] = field(default_factory=list)


@dataclass
class PrefetchResult:
    """The topic decision and retrieved context for one prefetch call.

    Returned by ``Eidolon.prepare_context()`` — bundles both pieces so a
    caller (e.g. an HTTP-backed ``EidolonClient``) gets them in a single
    round trip instead of two separate calls, while still letting it
    cache ``topic_change`` for a later ``store_turn()`` the way
    ``detect_topic()``+``retrieve_context()`` called separately always
    has.
    """

    topic_change: TopicChange
    bundle: ContextBundle


@dataclass
class Topic:
    """A conversation segment with summary, tags, and optional embedding."""

    id: int | None = None
    label: str = ""
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    embedding: list[float] | None = None
    status: Literal["active", "closed"] = "active"
    workspace_id: str | None = None
    created_at: datetime | None = None
    closed_at: datetime | None = None
