"""Tests for ReportTableFormatter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from tools.eval.report_index import PersistedReport
from tools.eval.report_table import ReportTableFormatter
from tools.eval.scorecard import Scorecard


def _report(**overrides) -> PersistedReport:
    defaults = {
        "path": Path("/tmp/r.json"),
        "recorded_at": datetime(2026, 7, 21, 14, 30),
        "script_name": "bike saga",
        "model_identity": "lm_studio/deepseek-v4",
        "git_commit": "abc1234",
        "tag": "prompt-17",
        "scorecard": Scorecard(
            topic_change_checked=2, topic_change_passed=2,
            avg_fact_similarity=0.9, avg_cluster_similarity=0.85,
            avg_memory_item_similarity=0.8,
            memory_item_type_checked=1, memory_item_type_matched=1,
        ),
    }
    defaults.update(overrides)
    return PersistedReport(**defaults)


class TestFormat:
    def test_includes_header_and_row_values(self):
        table = ReportTableFormatter().format([_report()])

        assert "script" in table
        assert "2026-07-21 14:30" in table
        assert "bike saga" in table
        assert "prompt-17" in table
        assert "abc1234" in table
        assert "2/2" in table
        assert "0.90" in table

    def test_legacy_report_without_scorecard_shows_placeholders(self):
        report = _report(scorecard=None, tag=None, git_commit="unknown")

        table = ReportTableFormatter().format([report])

        rows = table.splitlines()[2]
        assert "-" in rows

    def test_long_values_are_truncated_to_column_width(self):
        report = _report(script_name="a" * 50)

        table = ReportTableFormatter().format([report])

        assert "a" * 50 not in table
        assert "…" in table

    def test_empty_report_list_yields_header_only(self):
        table = ReportTableFormatter().format([])

        assert len(table.splitlines()) == 2
