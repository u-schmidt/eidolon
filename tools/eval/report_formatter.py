"""Renders an EvalReport to markdown and persists it (+ a JSON copy) to disk."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from tools.eval.results import EndResult, EvalReport, TurnResult
from tools.eval.scorecard import Scorecard

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


class EvalReportFormatter:
    """Formats an ``EvalReport`` as markdown and writes it (plus a
    machine-readable JSON copy, for later cross-run comparison) to disk."""

    def to_markdown(self, report: EvalReport) -> str:
        lines = [
            f"# Eval report: {report.script_name}",
            "",
            f"- workspace_id: `{report.workspace_id}`",
            f"- model: {report.model_identity}",
            f"- embedder: {report.embedder_identity}",
            f"- db: `{report.db_path}`",
            f"- git commit: `{report.git_commit}`",
        ]
        if report.tag:
            lines.append(f"- tag: `{report.tag}`")
        lines.append("")
        lines.extend(self._scorecard(report))
        lines.append("")
        lines.append("## Turns")
        lines.append("")
        for turn_result in report.turn_results:
            lines.extend(self._turn_section(turn_result))
        lines.append("## End of script")
        lines.append("")
        lines.extend(self._end_section(report.end_result))
        return "\n".join(lines)

    def write(self, report: EvalReport, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        base = f"{_SLUG_RE.sub('_', report.script_name)}__{_SLUG_RE.sub('_', report.model_identity)}__{timestamp}"

        md_path = directory / f"{base}.md"
        json_path = directory / f"{base}.json"
        md_path.write_text(self.to_markdown(report), encoding="utf-8")
        payload = {**asdict(report), "scorecard": asdict(Scorecard.from_report(report))}
        json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return md_path

    def _scorecard(self, report: EvalReport) -> list[str]:
        sc = Scorecard.from_report(report)
        topic_accuracy = (
            f"{sc.topic_change_passed}/{sc.topic_change_checked}" if sc.topic_change_checked else "n/a"
        )
        avg_fact_similarity = f"{sc.avg_fact_similarity:.2f}" if sc.avg_fact_similarity is not None else "n/a"
        avg_cluster_similarity = (
            f"{sc.avg_cluster_similarity:.2f}" if sc.avg_cluster_similarity is not None else "n/a"
        )
        avg_similarity = (
            f"{sc.avg_memory_item_similarity:.2f}" if sc.avg_memory_item_similarity is not None else "n/a"
        )
        type_match_rate = (
            f"{sc.memory_item_type_matched}/{sc.memory_item_type_checked}"
            if sc.memory_item_type_checked
            else "n/a"
        )

        return [
            "## Scorecard",
            "",
            f"- topic-change accuracy: {topic_accuracy}",
            f"- end-fact similarity (avg): {avg_fact_similarity}",
            f"- end-cluster similarity (avg): {avg_cluster_similarity}",
            f"- memory-item similarity (avg): {avg_similarity}",
            f"- memory-item type match: {type_match_rate}",
        ]

    def _turn_section(self, turn_result: TurnResult) -> list[str]:
        lines = [f"### Turn {turn_result.turn_number}", "", f"> {turn_result.user_message}", ""]

        if turn_result.topic_change_expected is not None:
            mark = "✅" if turn_result.topic_change_pass else "❌"
            lines.append(
                f"{mark} topic_change: expected `{turn_result.topic_change_expected}`, "
                f"got `{turn_result.topic_change_actual}`"
            )
        else:
            lines.append(f"- topic_change: `{turn_result.topic_change_actual}` (not checked)")

        if turn_result.topic_label_hint:
            lines.append(
                f'- label hint: "{turn_result.topic_label_hint}" '
                f'vs actual: "{turn_result.topic_label_actual}"'
            )

        for match in turn_result.memory_item_matches:
            type_mark = "✅" if match.type_match else "⚠️"
            actual_desc = (
                f'[{match.matched_item.item_type}] "{match.matched_item.content}"'
                if match.matched_item
                else "(none)"
            )
            lines.append(
                f'- hint [{match.hint.item_type}] "{match.hint.content_hint}" '
                f"→ score {match.score:.2f} {type_mark} matched: {actual_desc}"
            )

        if turn_result.unmatched_actual_items:
            lines.append("- unmatched actual items (not claimed by any hint):")
            for item in turn_result.unmatched_actual_items:
                lines.append(f'  - [{item.item_type}] "{item.content}"')

        lines.append("")
        return lines

    def _end_section(self, end: EndResult) -> list[str]:
        lines = []
        for fact_match in end.fact_matches:
            expected = fact_match.expected
            actual_desc = (
                f'{fact_match.matched_fact.category}/{fact_match.matched_fact.key} '
                f'= "{fact_match.matched_fact.value}"'
                if fact_match.matched_fact
                else "(none)"
            )
            lines.append(
                f'- fact hint: {expected.category}/{expected.key} = "{expected.value}" '
                f"→ score {fact_match.score:.2f} matched: {actual_desc}"
            )

        if end.actual_facts:
            lines.append("")
            lines.append("Full consolidated facts profile (review for stray duplicates):")
            for fact in end.actual_facts:
                lines.append(f'- {fact.category}/{fact.key} = "{fact.value}" (source: {fact.source})')

        if end.expected_topics or end.actual_topics:
            lines.append("")
            lines.append("Expected topic hints:")
            for topic_hint in end.expected_topics:
                lines.append(f"- {topic_hint.label_hint}")
            lines.append("")
            lines.append("Actual closed topics:")
            for topic in end.actual_topics:
                lines.append(f"- {topic['label']}: {topic['summary']}")

        for cluster_match in end.cluster_matches:
            expected_cluster = cluster_match.expected
            actual_desc = (
                f"{cluster_match.matched_cluster['label']} "
                f"(holding {cluster_match.matched_cluster['topic_count']} topics): "
                f"{cluster_match.matched_cluster['summary']}"
                if cluster_match.matched_cluster
                else "(none)"
            )
            lines.append(
                f'- cluster hint: "{expected_cluster.label_hint}" → score {cluster_match.score:.2f} '
                f"matched: {actual_desc}"
            )

        if end.actual_clusters:
            lines.append("")
            lines.append("Full cluster list (review for stray/unexpected clusters):")
            for cluster in end.actual_clusters:
                lines.append(
                    f"- {cluster['label']} (holding {cluster['topic_count']} topics): {cluster['summary']}"
                )

        return lines
