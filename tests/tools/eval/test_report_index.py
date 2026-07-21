"""Tests for ReportIndex — loading persisted report JSON into PersistedReport."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tools.eval.report_index import ReportIndex


def _write(directory: Path, name: str, data: dict) -> None:
    (directory / name).write_text(json.dumps(data), encoding="utf-8")


class TestLoad:
    def test_loads_a_full_schema_report(self):
        directory = Path(tempfile.mkdtemp())
        _write(directory, "a.json", {
            "script_name": "bike script",
            "model_identity": "lm_studio/deepseek-v4",
            "git_commit": "abc1234",
            "tag": "prompt-17",
            "scorecard": {
                "topic_change_checked": 2, "topic_change_passed": 2,
                "avg_fact_similarity": 0.9, "avg_cluster_similarity": None,
                "avg_memory_item_similarity": 0.8,
                "memory_item_type_checked": 1, "memory_item_type_matched": 1,
            },
        })

        reports = ReportIndex().load(directory)

        assert len(reports) == 1
        report = reports[0]
        assert report.script_name == "bike script"
        assert report.git_commit == "abc1234"
        assert report.tag == "prompt-17"
        assert report.scorecard is not None
        assert report.scorecard.avg_fact_similarity == 0.9
        assert report.scorecard.avg_cluster_similarity is None

    def test_legacy_report_missing_new_keys_degrades_gracefully(self):
        # Reports written before git_commit/tag/scorecard existed have
        # none of those keys at all — must not raise.
        directory = Path(tempfile.mkdtemp())
        _write(directory, "legacy.json", {
            "script_name": "old script",
            "model_identity": "ollama/minimax",
        })

        reports = ReportIndex().load(directory)

        assert len(reports) == 1
        report = reports[0]
        assert report.git_commit == "unknown"
        assert report.tag is None
        assert report.scorecard is None

    def test_loads_multiple_reports_sorted_by_filename(self):
        directory = Path(tempfile.mkdtemp())
        _write(directory, "b.json", {"script_name": "s2", "model_identity": "m"})
        _write(directory, "a.json", {"script_name": "s1", "model_identity": "m"})

        reports = ReportIndex().load(directory)

        assert [r.script_name for r in reports] == ["s1", "s2"]
