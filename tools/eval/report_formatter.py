"""Renders an EvalReport to markdown and persists it (+ a JSON copy) to disk."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from tools.eval.results import EndResult, EvalReport, TurnResult

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


class EvalReportFormatter:
    """Formats an ``EvalReport`` as markdown and writes it (plus a
    machine-readable JSON copy, for later cross-run comparison) to disk."""

    def to_markdown(self, report: EvalReport) -> str:
        lines = [
            f"# Eval report: {report.script_name}",
            "",
            f"- peer_id: `{report.peer_id}`",
            f"- model: {report.model_identity}",
            f"- embedder: {report.embedder_identity}",
            f"- db: `{report.db_path}`",
            "",
        ]
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
        json_path.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")
        return md_path

    def _scorecard(self, report: EvalReport) -> list[str]:
        checked = [tr for tr in report.turn_results if tr.topic_change_pass is not None]
        topic_accuracy = (
            f"{sum(1 for tr in checked if tr.topic_change_pass)}/{len(checked)}" if checked else "n/a"
        )

        fact_scores = [m.score for m in report.end_result.fact_matches]
        avg_fact_similarity = f"{sum(fact_scores) / len(fact_scores):.2f}" if fact_scores else "n/a"

        scores = [m.score for tr in report.turn_results for m in tr.memory_item_matches]
        avg_similarity = f"{sum(scores) / len(scores):.2f}" if scores else "n/a"

        typed = [
            m.type_match for tr in report.turn_results for m in tr.memory_item_matches if m.matched_item
        ]
        type_match_rate = f"{sum(typed)}/{len(typed)}" if typed else "n/a"

        return [
            "## Scorecard",
            "",
            f"- topic-change accuracy: {topic_accuracy}",
            f"- end-fact similarity (avg): {avg_fact_similarity}",
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

        return lines
