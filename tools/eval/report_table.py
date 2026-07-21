"""Renders a list of PersistedReports as a fixed-width text table."""

from __future__ import annotations

from tools.eval.report_index import PersistedReport

_NA = "-"

_COLUMNS = (
    ("recorded", 16),
    ("script", 28),
    ("model", 22),
    ("tag", 12),
    ("commit", 13),
    ("topic_acc", 9),
    ("fact_sim", 8),
    ("cluster_sim", 11),
    ("item_sim", 8),
    ("type_match", 10),
)


def _fit(value: str, width: int) -> str:
    if len(value) <= width:
        return value.ljust(width)
    return value[: width - 1] + "…"


class ReportTableFormatter:
    """Formats PersistedReports as one flat, fixed-width table — no
    grouping or aggregation, just every run in the given order."""

    def format(self, reports: list[PersistedReport]) -> str:
        header = "  ".join(_fit(name, width) for name, width in _COLUMNS)
        separator = "  ".join("-" * width for _, width in _COLUMNS)
        rows = [self._row(report) for report in reports]
        return "\n".join([header, separator, *rows])

    def _row(self, report: PersistedReport) -> str:
        sc = report.scorecard
        values = [
            report.recorded_at.strftime("%Y-%m-%d %H:%M"),
            report.script_name,
            report.model_identity,
            report.tag or _NA,
            report.git_commit,
            self._ratio(sc.topic_change_passed, sc.topic_change_checked) if sc else _NA,
            self._avg(sc.avg_fact_similarity) if sc else _NA,
            self._avg(sc.avg_cluster_similarity) if sc else _NA,
            self._avg(sc.avg_memory_item_similarity) if sc else _NA,
            self._ratio(sc.memory_item_type_matched, sc.memory_item_type_checked) if sc else _NA,
        ]
        return "  ".join(_fit(value, width) for value, (_, width) in zip(values, _COLUMNS, strict=True))

    def _ratio(self, passed: int, checked: int) -> str:
        return f"{passed}/{checked}" if checked else _NA

    def _avg(self, value: float | None) -> str:
        return f"{value:.2f}" if value is not None else _NA
