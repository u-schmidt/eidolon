"""Composes the trace reader + storage into an aggregate prompt/pipeline
quality view — how often topic detection falls back to CONTINUE on an
unparseable response, whether topic relabeling flip-flops on the same
topic, the extracted item_type/attribution_type mix, facts whose category
falls outside the configured list, facts consolidation repeatedly
overwrites, and how often the user manually corrects the pipeline.

Needs ``debug.enabled`` trace data (``topic_detection``/``topic_relabel``/
``extraction``/``fact_consolidation``/``fact_correction`` events) for
everything except the fact-category-outliers check, which reads the
current facts table directly — unlike
``tools/dashboard/performance_reader.py``, this is *not* an always-on
page. Lives in ``tools/``, not ``core/``, for the same reason
``tools/dashboard/turn_reader.py``/``performance_reader.py``/
``retrieval_quality_reader.py`` do — read + aggregation glue for a
dashboard view, not core orchestration logic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

from core.interfaces import StorageProvider, TraceReader


@dataclass
class ContinueFallbackStats:
    total_calls: int
    literal_continue: int
    fallback_continue: int
    new_topic: int


@dataclass
class RelabelFlipFlop:
    topic_id: int
    latest_label: str | None
    relabel_count: int
    distinct_labels: int


@dataclass
class TypeCount:
    type_name: str
    count: int


@dataclass
class FactCategoryOutlier:
    peer_id: str
    category: str
    key: str
    value: str


@dataclass
class FactChurn:
    category: str
    key: str
    upsert_count: int
    distinct_values: int


@dataclass
class CorrectedFact:
    category: str
    key: str
    correction_count: int


@dataclass
class FactCorrectionStats:
    total_corrections: int
    upserts: int
    deletes: int
    top_corrected: list[CorrectedFact] = field(default_factory=list)


@dataclass
class PromptQualitySummary:
    since: datetime
    total_topic_detection_calls: int
    continue_fallback: ContinueFallbackStats
    fact_correction: FactCorrectionStats
    relabel_flip_flops: list[RelabelFlipFlop] = field(default_factory=list)
    item_type_counts: list[TypeCount] = field(default_factory=list)
    attribution_type_counts: list[TypeCount] = field(default_factory=list)
    fact_category_outliers: list[FactCategoryOutlier] = field(default_factory=list)
    fact_churn: list[FactChurn] = field(default_factory=list)


class PromptQualityReader:
    """Read-only aggregation over ``TraceReader``/``StorageProvider`` for
    the dashboard's prompt/pipeline-quality page (see
    ``tools/dashboard/app.py``).
    """

    def __init__(
        self, trace_reader: TraceReader, storage: StorageProvider, configured_fact_categories: list[str],
    ) -> None:
        self._trace_reader = trace_reader
        self._storage = storage
        self._configured_fact_categories = set(configured_fact_categories)

    def summary(self, since: datetime, top_n: int = 15) -> PromptQualitySummary:
        detection_events = self._trace_reader.events_of_type_since("topic_detection", since)
        relabel_events = self._trace_reader.events_of_type_since("topic_relabel", since)
        extraction_events = self._trace_reader.events_of_type_since("extraction", since)
        consolidation_events = self._trace_reader.events_of_type_since("fact_consolidation", since)
        correction_events = self._trace_reader.events_of_type_since("fact_correction", since)

        return PromptQualitySummary(
            since=since,
            total_topic_detection_calls=len(detection_events),
            continue_fallback=self._continue_fallback_stats(detection_events),
            relabel_flip_flops=self._relabel_flip_flops(relabel_events, top_n),
            item_type_counts=self._type_counts(extraction_events, "item_type"),
            attribution_type_counts=self._type_counts(extraction_events, "attribution_type"),
            fact_category_outliers=self._fact_category_outliers(),
            fact_churn=self._fact_churn(consolidation_events, top_n),
            fact_correction=self._fact_correction_stats(correction_events, top_n),
        )

    @staticmethod
    def _continue_fallback_stats(detection_events: list[dict]) -> ContinueFallbackStats:
        """``fallback_continue`` = the LLM's raw response wasn't the
        literal string "CONTINUE" but still got parsed into one — see
        ``core/detection/topic_detector.py``'s ``_parse_detection_response``,
        which defaults to CONTINUE for any unrecognized text."""
        literal_continue = 0
        continue_total = 0
        new_topic = 0
        for e in detection_events:
            payload = e["payload"]
            if payload.get("result_type") == "CONTINUE":
                continue_total += 1
                if payload.get("response", "").strip().upper() == "CONTINUE":
                    literal_continue += 1
            elif payload.get("result_type") == "NEW_TOPIC":
                new_topic += 1
        return ContinueFallbackStats(
            total_calls=len(detection_events),
            literal_continue=literal_continue,
            fallback_continue=continue_total - literal_continue,
            new_topic=new_topic,
        )

    @staticmethod
    def _relabel_flip_flops(relabel_events: list[dict], top_n: int) -> list[RelabelFlipFlop]:
        labels_by_topic: dict[int, list[str]] = {}
        for e in relabel_events:
            payload = e["payload"]
            label = payload.get("label")
            if not label:
                continue
            labels_by_topic.setdefault(payload["topic_id"], []).append(label)

        flip_flops = [
            RelabelFlipFlop(
                topic_id=topic_id,
                latest_label=labels[-1],
                relabel_count=len(labels),
                distinct_labels=len(set(labels)),
            )
            for topic_id, labels in labels_by_topic.items()
            if len(set(labels)) > 1
        ]
        flip_flops.sort(key=lambda f: (f.distinct_labels, f.relabel_count), reverse=True)
        return flip_flops[:top_n]

    @staticmethod
    def _type_counts(extraction_events: list[dict], field_name: str) -> list[TypeCount]:
        counts: Counter[str] = Counter()
        for e in extraction_events:
            for item in e["payload"].get("items", []):
                value = item.get(field_name)
                if value:
                    counts[value] += 1
        return [
            TypeCount(type_name=name, count=count)
            for name, count in counts.most_common()
        ]

    def _fact_category_outliers(self) -> list[FactCategoryOutlier]:
        """Reflects the *current* facts table, not the selected time
        window — a config-compliance check, not window activity."""
        return [
            FactCategoryOutlier(peer_id=f.peer_id, category=f.category, key=f.key, value=f.value)
            for f in self._storage.get_all_facts()
            if f.category not in self._configured_fact_categories
        ]

    @staticmethod
    def _fact_churn(consolidation_events: list[dict], top_n: int) -> list[FactChurn]:
        upsert_counts: Counter[tuple[str, str]] = Counter()
        values_seen: dict[tuple[str, str], set[str]] = {}
        for e in consolidation_events:
            for entry in e["payload"].get("upserts", []):
                key = (str(entry.get("category", "")).strip().lower(), str(entry.get("key", "")).strip().lower())
                upsert_counts[key] += 1
                values_seen.setdefault(key, set()).add(str(entry.get("value", "")).strip())

        churn = [
            FactChurn(category=category, key=key, upsert_count=count, distinct_values=len(values_seen[(category, key)]))
            for (category, key), count in upsert_counts.items()
            if count > 1
        ]
        churn.sort(key=lambda c: c.upsert_count, reverse=True)
        return churn[:top_n]

    @staticmethod
    def _fact_correction_stats(correction_events: list[dict], top_n: int) -> FactCorrectionStats:
        upserts = sum(1 for e in correction_events if e["payload"].get("action") == "upsert")
        deletes = sum(1 for e in correction_events if e["payload"].get("action") == "delete")
        counts: Counter[tuple[str, str]] = Counter(
            (e["payload"]["category"], e["payload"]["key"]) for e in correction_events
        )
        top_corrected = [
            CorrectedFact(category=category, key=key, correction_count=count)
            for (category, key), count in counts.most_common(top_n)
        ]
        return FactCorrectionStats(
            total_corrections=len(correction_events),
            upserts=upserts,
            deletes=deletes,
            top_corrected=top_corrected,
        )
