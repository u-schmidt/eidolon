"""Tests for the eval package's cosine_similarity helper."""

from __future__ import annotations

from tools.eval.similarity import cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite_vectors_score_negative_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0

    def test_zero_vector_scores_zero_not_nan(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_mismatched_length_does_not_raise(self):
        # strict=False keeps zip's dot-product truncating to the shorter
        # vector rather than raising — but the norms still use each
        # vector's own full length, so this is NOT the same value as
        # cosine_similarity([1.0, 1.0], [1.0, 1.0]) (which would be 1.0).
        result = cosine_similarity([1.0, 1.0, 1.0], [1.0, 1.0])
        assert result == 2 / (3**0.5 * 2**0.5)
