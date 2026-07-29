"""Tests for EvalScorer."""

from __future__ import annotations

from core.models import Fact, MemoryItem, TopicChange
from tools.eval.models import ClusterHint, EndExpectation, FactExpectation, MemoryItemHint, TurnExpectation
from tools.eval.scorer import EvalScorer


class FakeEmbedder:
    """Returns a fixed vector per known text, so similarity is predictable."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def embed(self, text: str) -> list[float]:
        return self._vectors[text]


class TestScoreTurnTopicChange:
    def test_pass_when_actual_matches_expected(self):
        scorer = EvalScorer(FakeEmbedder({}))
        result = scorer.score_turn(
            1, "hi", TurnExpectation(topic_change="NEW_TOPIC"),
            TopicChange(type="NEW_TOPIC", label="Bike"), [],
        )
        assert result.topic_change_pass is True

    def test_fail_when_actual_differs(self):
        scorer = EvalScorer(FakeEmbedder({}))
        result = scorer.score_turn(
            1, "hi", TurnExpectation(topic_change="CONTINUE"),
            TopicChange(type="NEW_TOPIC", label="Bike"), [],
        )
        assert result.topic_change_pass is False

    def test_none_when_not_asserted(self):
        scorer = EvalScorer(FakeEmbedder({}))
        result = scorer.score_turn(
            1, "hi", TurnExpectation(topic_change=None),
            TopicChange(type="CONTINUE"), [],
        )
        assert result.topic_change_pass is None


class TestScoreTurnMemoryItems:
    def test_no_actual_items_scores_zero(self):
        scorer = EvalScorer(FakeEmbedder({}))
        expectation = TurnExpectation(
            memory_items=[MemoryItemHint(item_type="essence", content_hint="wants a bike")]
        )

        result = scorer.score_turn(1, "hi", expectation, TopicChange(type="CONTINUE"), [])

        assert len(result.memory_item_matches) == 1
        match = result.memory_item_matches[0]
        assert match.score == 0.0
        assert match.matched_item is None
        assert match.type_match is False

    def test_picks_best_matching_item_and_flags_type(self):
        hint_text = "wants a bike"
        embedder = FakeEmbedder({
            hint_text: [1.0, 0.0],
        })
        essence = MemoryItem(
            id=1, content="user wants a bike for 2000 euro", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            embedding=[1.0, 0.0],  # perfect match
        )
        fact = MemoryItem(
            id=2, content="user has 2000 euro", item_type="fact",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            embedding=[0.0, 1.0],  # orthogonal, worse match
        )
        expectation = TurnExpectation(
            memory_items=[MemoryItemHint(item_type="essence", content_hint=hint_text)]
        )
        scorer = EvalScorer(embedder)

        result = scorer.score_turn(1, "hi", expectation, TopicChange(type="CONTINUE"), [essence, fact])

        match = result.memory_item_matches[0]
        assert match.matched_item is essence
        assert match.score == 1.0
        assert match.type_match is True
        # The fact item wasn't claimed by any hint, so it's unmatched.
        assert result.unmatched_actual_items == [fact]

    def test_type_mismatch_is_flagged_but_still_reported(self):
        hint_text = "wants a bike"
        embedder = FakeEmbedder({hint_text: [1.0, 0.0]})
        fact = MemoryItem(
            id=1, content="user wants a bike", item_type="fact",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            embedding=[1.0, 0.0],
        )
        expectation = TurnExpectation(
            memory_items=[MemoryItemHint(item_type="essence", content_hint=hint_text)]
        )
        scorer = EvalScorer(embedder)

        result = scorer.score_turn(1, "hi", expectation, TopicChange(type="CONTINUE"), [fact])

        match = result.memory_item_matches[0]
        assert match.matched_item is fact
        assert match.type_match is False

    def test_all_actual_items_claimed_leaves_no_unmatched(self):
        hint_text = "wants a bike"
        embedder = FakeEmbedder({hint_text: [1.0, 0.0]})
        item = MemoryItem(
            id=1, content="user wants a bike", item_type="essence",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            embedding=[1.0, 0.0],
        )
        expectation = TurnExpectation(
            memory_items=[MemoryItemHint(item_type="essence", content_hint=hint_text)]
        )
        scorer = EvalScorer(embedder)

        result = scorer.score_turn(1, "hi", expectation, TopicChange(type="CONTINUE"), [item])

        assert result.unmatched_actual_items == []


class TestScoreEnd:
    def test_scores_exact_match_highly(self):
        embedder = FakeEmbedder({"preference: bike_budget = 2000 EUR": [1.0, 0.0]})
        scorer = EvalScorer(embedder)
        expectation = EndExpectation(
            facts=[FactExpectation(category="preference", key="bike_budget", value="2000 EUR")]
        )
        actual = [Fact(workspace_id="p1", category="preference", key="bike_budget", value="2000 EUR", source="user")]

        result = scorer.score_end(expectation, actual, [], [])

        match = result.fact_matches[0]
        assert match.score == 1.0
        assert match.matched_fact is actual[0]

    def test_includes_full_actual_facts_list_for_stray_duplicate_review(self):
        embedder = FakeEmbedder({"preference: bike_budget = 2000 EUR": [1.0, 0.0]})
        scorer = EvalScorer(embedder)
        expectation = EndExpectation(
            facts=[FactExpectation(category="preference", key="bike_budget", value="2000 EUR")]
        )
        actual = [Fact(workspace_id="p1", category="preference", key="bike_budget", value="2000 EUR", source="user")]

        result = scorer.score_end(expectation, actual, [], [])

        assert result.actual_facts == actual

    def test_scores_semantically_similar_but_differently_named_fact_highly(self):
        # Reproduces a real result: the model chose "finance/budget_item"
        # for what the script expected as "preference/bike_budget" — a
        # fine outcome, since Fact is never consumed programmatically,
        # only injected into prompts. Exact category/key match would be
        # the wrong bar here.
        embedder = FakeEmbedder({
            "preference: bike_budget = 2000 EUR": [1.0, 0.0],
            "finance: budget_item = 2000 euros": [0.9, 0.1],
        })
        scorer = EvalScorer(embedder)
        expectation = EndExpectation(
            facts=[FactExpectation(category="preference", key="bike_budget", value="2000 EUR")]
        )
        actual = [Fact(workspace_id="p1", category="finance", key="budget_item", value="2000 euros", source="ai")]

        result = scorer.score_end(expectation, actual, [], [])

        match = result.fact_matches[0]
        assert match.matched_fact is actual[0]
        assert match.score > 0.9

    def test_no_actual_facts_scores_zero(self):
        scorer = EvalScorer(FakeEmbedder({}))
        expectation = EndExpectation(
            facts=[FactExpectation(category="preference", key="bike_budget", value="2000 EUR")]
        )

        result = scorer.score_end(expectation, [], [], [])

        match = result.fact_matches[0]
        assert match.score == 0.0
        assert match.matched_fact is None


class TestScoreEndClusters:
    def test_scores_exact_match_highly(self):
        embedder = FakeEmbedder({
            "bike purchase saga": [1.0, 0.0],
            "Bike saga: consolidated bike story": [1.0, 0.0],
        })
        scorer = EvalScorer(embedder)
        expectation = EndExpectation(clusters=[ClusterHint(label_hint="bike purchase saga")])
        actual = [{"id": 1, "label": "Bike saga", "summary": "consolidated bike story", "topic_count": 2}]

        result = scorer.score_end(expectation, [], [], actual)

        match = result.cluster_matches[0]
        assert match.score == 1.0
        assert match.matched_cluster is actual[0]

    def test_includes_full_actual_clusters_list_for_stray_review(self):
        embedder = FakeEmbedder({"Bike saga: summary": [1.0, 0.0]})
        scorer = EvalScorer(embedder)
        expectation = EndExpectation()
        actual = [{"id": 1, "label": "Bike saga", "summary": "summary", "topic_count": 2}]

        result = scorer.score_end(expectation, [], [], actual)

        assert result.actual_clusters == actual

    def test_no_actual_clusters_scores_zero(self):
        scorer = EvalScorer(FakeEmbedder({}))
        expectation = EndExpectation(clusters=[ClusterHint(label_hint="bike purchase saga")])

        result = scorer.score_end(expectation, [], [], [])

        match = result.cluster_matches[0]
        assert match.score == 0.0
        assert match.matched_cluster is None

    def test_picks_best_scoring_cluster_among_several(self):
        embedder = FakeEmbedder({
            "bike purchase saga": [1.0, 0.0],
            "Bike saga: about a bike": [0.95, 0.05],
            "Weather chat: about the weather": [0.0, 1.0],
        })
        scorer = EvalScorer(embedder)
        expectation = EndExpectation(clusters=[ClusterHint(label_hint="bike purchase saga")])
        actual = [
            {"id": 1, "label": "Weather chat", "summary": "about the weather", "topic_count": 2},
            {"id": 2, "label": "Bike saga", "summary": "about a bike", "topic_count": 3},
        ]

        result = scorer.score_end(expectation, [], [], actual)

        match = result.cluster_matches[0]
        assert match.matched_cluster is actual[1]
