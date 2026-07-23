"""Stdlib-only JSON (de)serialization for core.models dataclasses.

The server (server/app.py) uses FastAPI/pydantic to handle this
automatically, but the client side deliberately does NOT reach for
pydantic to mirror that — the Hermes container only ever installs
client/, never server/'s fastapi/pydantic dependency, and doing so here
would silently reintroduce it into the exact venv this feature exists
to keep dependency-free. Plain ``dataclasses.asdict()`` plus a manual
datetime-ISO8601 walk outgoing; manual dict-to-dataclass reconstruction
incoming.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any

from core.models import (
    ClusterSummary,
    ContextBundle,
    EnvironmentSnapshot,
    Fact,
    MemoryItem,
    PrefetchResult,
    RelatedTopic,
    TopicChange,
)


def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses/datetimes into plain JSON-safe values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def _parse_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def environment_from_dict(d: dict[str, Any] | None) -> EnvironmentSnapshot | None:
    if d is None:
        return None
    return EnvironmentSnapshot(
        location=d.get("location"),
        time=_parse_datetime(d.get("time")),
        day_of_week=d.get("day_of_week"),
        situation=d.get("situation"),
        rules=d.get("rules", []),
    )


def fact_from_dict(d: dict[str, Any]) -> Fact:
    return Fact(
        peer_id=d["peer_id"],
        category=d["category"],
        key=d["key"],
        value=d["value"],
        source=d["source"],
        importance=d.get("importance", "medium"),
        created_at=_parse_datetime(d.get("created_at")),
    )


def related_topic_from_dict(d: dict[str, Any]) -> RelatedTopic:
    return RelatedTopic(
        topic_id=d["topic_id"],
        summary=d["summary"],
        tags=d.get("tags", []),
        cluster_label=d.get("cluster_label"),
        timestamp=_parse_datetime(d.get("timestamp")),
        label=d.get("label"),
    )


def cluster_summary_from_dict(d: dict[str, Any]) -> ClusterSummary:
    return ClusterSummary(
        cluster_id=d["cluster_id"],
        label=d["label"],
        summary=d["summary"],
        topic_count=d["topic_count"],
        recent_topics=[related_topic_from_dict(t) for t in d.get("recent_topics", [])],
    )


def memory_item_from_dict(d: dict[str, Any]) -> MemoryItem:
    return MemoryItem(
        content=d["content"],
        item_type=d.get("item_type", "essence"),
        source_peer=d.get("source_peer", ""),
        subject_peer=d.get("subject_peer", ""),
        attribution_type=d.get("attribution_type", ""),
        id=d.get("id"),
        topic_id=d.get("topic_id"),
        message_id=d.get("message_id"),
        embedding=d.get("embedding"),
        timestamp=_parse_datetime(d.get("timestamp")),
        context=environment_from_dict(d.get("context")),
    )


def context_bundle_from_dict(d: dict[str, Any]) -> ContextBundle:
    return ContextBundle(
        facts=[fact_from_dict(f) for f in d.get("facts", [])],
        environment=environment_from_dict(d.get("environment")),
        current_topic_label=d.get("current_topic_label"),
        related_topics=[related_topic_from_dict(t) for t in d.get("related_topics", [])],
        recent_topics=[related_topic_from_dict(t) for t in d.get("recent_topics", [])],
        cluster_summaries=[cluster_summary_from_dict(c) for c in d.get("cluster_summaries", [])],
        memory_items=[memory_item_from_dict(m) for m in d.get("memory_items", [])],
    )


def topic_change_from_dict(d: dict[str, Any]) -> TopicChange:
    return TopicChange(type=d["type"], label=d.get("label"))


def prefetch_result_from_dict(d: dict[str, Any]) -> PrefetchResult:
    return PrefetchResult(
        topic_change=topic_change_from_dict(d["topic_change"]),
        bundle=context_bundle_from_dict(d["bundle"]),
    )
