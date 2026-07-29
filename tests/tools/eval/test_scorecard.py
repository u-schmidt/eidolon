"""Tests for Scorecard.from_report."""

from __future__ import annotations

from core.models import Fact, MemoryItem
from tools.eval.models import ClusterHint, FactExpectation, MemoryItemHint
from tools.eval.results import ClusterMatch, EndResult, EvalReport, FactMatch, MemoryItemMatch, TurnResult
from tools.eval.scorecard import Scorecard


def _empty_report() -> EvalReport:
    return EvalReport(
        script_name="s", workspace_id="p", model_identity="m", embedder_identity="e", db_path="db",
        git_commit="abc1234",
    )


class TestFromReport:
    def test_empty_report_yields_all_none_or_zero(self):
        sc = Scorecard.from_report(_empty_report())

        assert sc.topic_change_checked == 0
        assert sc.topic_change_passed == 0
        assert sc.avg_fact_similarity is None
        assert sc.avg_cluster_similarity is None
        assert sc.avg_memory_item_similarity is None
        assert sc.memory_item_type_checked == 0
        assert sc.memory_item_type_matched == 0

    def test_topic_change_ignores_unchecked_turns(self):
        report = _empty_report()
        report.turn_results = [
            TurnResult(turn_number=1, user_message="a", topic_change_actual="NEW_TOPIC",
                       topic_change_expected="NEW_TOPIC", topic_change_pass=True),
            TurnResult(turn_number=2, user_message="b", topic_change_actual="CONTINUE",
                       topic_change_expected=None, topic_change_pass=None),
            TurnResult(turn_number=3, user_message="c", topic_change_actual="CONTINUE",
                       topic_change_expected="CONTINUE", topic_change_pass=False),
        ]

        sc = Scorecard.from_report(report)

        assert sc.topic_change_checked == 2
        assert sc.topic_change_passed == 1

    def test_averages_fact_cluster_and_memory_item_scores(self):
        report = _empty_report()
        matched_item = MemoryItem(
            id=1, content="c", item_type="fact",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
        )
        report.turn_results = [
            TurnResult(
                turn_number=1, user_message="a", topic_change_actual="NEW_TOPIC",
                memory_item_matches=[
                    MemoryItemMatch(
                        hint=MemoryItemHint(item_type="fact", content_hint="h"),
                        score=0.8, matched_item=matched_item, type_match=True,
                    ),
                    MemoryItemMatch(
                        hint=MemoryItemHint(item_type="fact", content_hint="h2"),
                        score=0.6, matched_item=None, type_match=False,
                    ),
                ],
            ),
        ]
        report.end_result = EndResult(
            fact_matches=[
                FactMatch(expected=FactExpectation(category="c", key="k", value="v"), score=0.9,
                          matched_fact=Fact(workspace_id="p", category="c", key="k", value="v", source="ai")),
                FactMatch(expected=FactExpectation(category="c", key="k2", value="v2"), score=0.7,
                          matched_fact=None),
            ],
            cluster_matches=[
                ClusterMatch(expected=ClusterHint(label_hint="h"), score=1.0, matched_cluster=None),
            ],
        )

        sc = Scorecard.from_report(report)

        assert sc.avg_fact_similarity == 0.8
        assert sc.avg_cluster_similarity == 1.0
        assert sc.avg_memory_item_similarity == 0.7
        # type_match only counted for matches that actually resolved to an item
        assert sc.memory_item_type_checked == 1
        assert sc.memory_item_type_matched == 1
