"""Composes the trace reader + storage into an aggregate cluster-dynamics
view — whether topic clustering tends to match an existing cluster,
create a brand-new one, or find no confident match at all, and which
clusters are seeing the most consolidation activity.

Needs debug.enabled trace data (cluster_consolidation events) — no new
StorageProvider or TraceReader capability was needed to build this page,
unlike the other four dashboard readers: cluster_consolidation's existing
payload (match_index + per-candidate kind + resulting_cluster_id) already
carries everything, and StorageProvider.get_cluster already exists for
label/topic_count lookups. Lives in tools/, not core/, for the same
reason the other dashboard readers do.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from core.interfaces import StorageProvider, TraceReader


@dataclass
class ClusterDecisionStats:
    total_calls: int
    matched_existing: int
    new_cluster_created: int
    no_match: int


@dataclass
class ActiveCluster:
    cluster_id: int
    label: str | None
    total_topic_count: int
    activity_count: int


@dataclass
class ClusterDynamicsSummary:
    since: datetime
    decisions: ClusterDecisionStats
    most_active_clusters: list[ActiveCluster] = field(default_factory=list)


class ClusterDynamicsReader:
    """Read-only aggregation over TraceReader/StorageProvider for the
    dashboard's cluster-dynamics page (see tools/dashboard/app.py)."""

    def __init__(self, trace_reader: TraceReader, storage: StorageProvider) -> None:
        self._trace_reader = trace_reader
        self._storage = storage

    def summary(self, since: datetime, top_n: int = 15) -> ClusterDynamicsSummary:
        events = self._trace_reader.events_of_type_since("cluster_consolidation", since)

        return ClusterDynamicsSummary(
            since=since,
            decisions=self._decision_stats(events),
            most_active_clusters=self._most_active_clusters(events, top_n),
        )

    @staticmethod
    def _decision_kind(payload: dict) -> str:
        """One of "matched_existing" / "new_cluster_created" / "no_match".

        A valid match_index whose candidate kind is "cluster" matched an
        existing cluster; kind "topic" means two raw topics got promoted
        into a brand-new cluster (see
        core/consolidation/cluster_consolidator.py:170-186). An invalid/
        missing match_index means no clustering decision was made at all
        this call — not "new", a third, distinct bucket.
        """
        match_index = payload.get("match_index")
        candidates = payload.get("candidates", [])
        if not isinstance(match_index, int) or not (1 <= match_index <= len(candidates)):
            return "no_match"
        kind = candidates[match_index - 1].get("kind")
        return "matched_existing" if kind == "cluster" else "new_cluster_created"

    @classmethod
    def _decision_stats(cls, events: list[dict]) -> ClusterDecisionStats:
        counts = Counter(cls._decision_kind(e["payload"]) for e in events)
        return ClusterDecisionStats(
            total_calls=len(events),
            matched_existing=counts["matched_existing"],
            new_cluster_created=counts["new_cluster_created"],
            no_match=counts["no_match"],
        )

    def _most_active_clusters(self, events: list[dict], top_n: int) -> list[ActiveCluster]:
        activity: Counter[int] = Counter()
        for e in events:
            cluster_id = e["payload"].get("resulting_cluster_id")
            if cluster_id is not None:
                activity[cluster_id] += 1

        result = []
        for cluster_id, count in activity.most_common(top_n):
            cluster = self._storage.get_cluster(cluster_id)
            result.append(ActiveCluster(
                cluster_id=cluster_id,
                label=cluster["label"] if cluster else None,
                total_topic_count=cluster["topic_count"] if cluster else 0,
                activity_count=count,
            ))
        return result
