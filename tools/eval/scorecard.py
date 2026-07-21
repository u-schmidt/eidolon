"""Scorecard — the reduced numeric summary of an EvalReport.

Extracted out of ``EvalReportFormatter`` so the same computation can be
reused by anything that only needs the aggregate numbers (e.g. a
cross-run report analyzer) without reconstructing the full nested
``EvalReport`` object graph.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.eval.results import EvalReport


@dataclass
class Scorecard:
    """Aggregate scoring numbers for one eval run. Averages are ``None``
    when nothing was scored (the markdown/table "n/a" case) rather than
    a sentinel float, so callers can't mistake "no data" for a real 0.0."""

    topic_change_checked: int
    topic_change_passed: int
    avg_fact_similarity: float | None
    avg_cluster_similarity: float | None
    avg_memory_item_similarity: float | None
    memory_item_type_checked: int
    memory_item_type_matched: int

    @classmethod
    def from_report(cls, report: EvalReport) -> Scorecard:
        checked = [tr for tr in report.turn_results if tr.topic_change_pass is not None]

        fact_scores = [m.score for m in report.end_result.fact_matches]
        cluster_scores = [m.score for m in report.end_result.cluster_matches]
        item_scores = [m.score for tr in report.turn_results for m in tr.memory_item_matches]
        typed = [
            m.type_match for tr in report.turn_results for m in tr.memory_item_matches if m.matched_item
        ]

        return cls(
            topic_change_checked=len(checked),
            topic_change_passed=sum(1 for tr in checked if tr.topic_change_pass),
            avg_fact_similarity=sum(fact_scores) / len(fact_scores) if fact_scores else None,
            avg_cluster_similarity=sum(cluster_scores) / len(cluster_scores) if cluster_scores else None,
            avg_memory_item_similarity=sum(item_scores) / len(item_scores) if item_scores else None,
            memory_item_type_checked=len(typed),
            memory_item_type_matched=sum(typed),
        )
