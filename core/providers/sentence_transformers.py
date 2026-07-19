"""Sentence-transformers embedder for Eidolon.

Local PyTorch-based embedding as a fallback when no external provider is available.
Requires the ``sentence-transformers`` package (optional dependency).
"""

from __future__ import annotations


class SentenceTransformersEmbedder:
    """Generates embeddings locally via sentence-transformers.

    Falls back to this when no external embedding API is available
    (e.g., standalone CLI usage without LM Studio or Ollama).

    Parameters
    ----------
    model_name : str
        Name of the sentence-transformers model to load
        (default: ``all-MiniLM-L6-v2``).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text.

        Parameters
        ----------
        text : str
            The text to embed.

        Returns
        -------
        list[float]
            The embedding vector.
        """
        self._load_model()
        return self._model.encode(text).tolist()  # type: ignore[union-attr]
