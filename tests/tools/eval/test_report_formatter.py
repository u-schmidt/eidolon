"""Tests for EvalReportFormatter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from core.models import Fact, MemoryItem
from tools.eval.models import FactExpectation, MemoryItemHint, TopicHint
from tools.eval.report_formatter import EvalReportFormatter
from tools.eval.results import EndResult, EvalReport, FactMatch, MemoryItemMatch, TurnResult


def _report() -> EvalReport:
    matched_item = MemoryItem(
        id=1, content="user wants a bike", item_type="essence",
        source_peer="user", subject_peer="user", attribution_type="self_statement",
    )
    unmatched_item = MemoryItem(
        id=2, content="user has 2000 euro", item_type="fact",
        source_peer="user", subject_peer="user", attribution_type="self_statement",
    )
    turn = TurnResult(
        turn_number=1,
        user_message="I want a bike",
        topic_change_actual="NEW_TOPIC",
        topic_change_expected="NEW_TOPIC",
        topic_change_pass=True,
        topic_label_hint="buying a bike",
        topic_label_actual="Bike",
        memory_item_matches=[
            MemoryItemMatch(
                hint=MemoryItemHint(item_type="essence", content_hint="wants a bike"),
                score=0.91,
                matched_item=matched_item,
                type_match=True,
            )
        ],
        unmatched_actual_items=[unmatched_item],
    )
    end = EndResult(
        fact_matches=[
            FactMatch(
                expected=FactExpectation(category="preference", key="bike_budget", value="2000 EUR"),
                score=0.95,
                matched_fact=Fact(
                    peer_id="eval-peer", category="finance", key="budget_item",
                    value="2000 euros", source="ai",
                ),
            )
        ],
        expected_topics=[TopicHint(label_hint="buying a bike")],
        actual_topics=[{"label": "Bike", "summary": "User wants a bike"}],
    )
    return EvalReport(
        script_name="bike script",
        peer_id="eval-peer",
        model_identity="lm_studio/deepseek-v4",
        embedder_identity="lm_studio/nomic-embed-text",
        db_path="/tmp/eval.db",
        turn_results=[turn],
        end_result=end,
    )


class TestToMarkdown:
    def test_includes_header_and_model_identity(self):
        markdown = EvalReportFormatter().to_markdown(_report())

        assert "# Eval report: bike script" in markdown
        assert "lm_studio/deepseek-v4" in markdown
        assert "lm_studio/nomic-embed-text" in markdown

    def test_scorecard_reflects_results(self):
        markdown = EvalReportFormatter().to_markdown(_report())

        assert "topic-change accuracy: 1/1" in markdown
        assert "end-fact similarity (avg): 0.95" in markdown
        assert "memory-item similarity (avg): 0.91" in markdown
        assert "memory-item type match: 1/1" in markdown

    def test_reports_unmatched_actual_items(self):
        markdown = EvalReportFormatter().to_markdown(_report())

        assert "unmatched actual items" in markdown
        assert "user has 2000 euro" in markdown

    def test_reports_fact_match_even_with_different_category_key(self):
        markdown = EvalReportFormatter().to_markdown(_report())

        assert 'preference/bike_budget = "2000 EUR"' in markdown
        assert 'finance/budget_item = "2000 euros"' in markdown
        assert "score 0.95" in markdown

    def test_scorecard_handles_empty_report_as_not_applicable(self):
        empty = EvalReport(
            script_name="empty", peer_id="p", model_identity="m", embedder_identity="e", db_path="db",
        )

        markdown = EvalReportFormatter().to_markdown(empty)

        assert "topic-change accuracy: n/a" in markdown
        assert "end-fact similarity (avg): n/a" in markdown
        assert "memory-item similarity (avg): n/a" in markdown


class TestWrite:
    def test_writes_markdown_and_json_files(self):
        directory = tempfile.mkdtemp()

        md_path = EvalReportFormatter().write(_report(), directory)

        assert md_path.exists()
        assert md_path.suffix == ".md"
        assert "bike_script" in md_path.name
        json_path = md_path.with_suffix(".json")
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["script_name"] == "bike script"

    def test_creates_missing_directory(self):
        directory = Path(tempfile.mkdtemp()) / "nested" / "reports"

        md_path = EvalReportFormatter().write(_report(), directory)

        assert md_path.exists()
