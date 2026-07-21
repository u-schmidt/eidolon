"""Cosine similarity for eval scoring.

Deliberately a standalone copy of ``core/stores/sqlite_store.py``'s
private ``_cosine_similarity`` rather than an import across module
boundaries — same three-line formula, kept local to this package.
"""

from __future__ import annotations


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
