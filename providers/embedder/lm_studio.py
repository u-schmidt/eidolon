"""LM Studio embedder provider for Eidolon.

Calls a local LM Studio instance for text embeddings.
Default endpoint: ``http://localhost:1234/v1/embeddings``
"""

from __future__ import annotations

import requests


class LMStudioEmbedder:
    """Generates embeddings via a local LM Studio API.

    Parameters
    ----------
    base_url : str
        Base URL of the LM Studio instance (default: ``http://localhost:1234``).
    model : str
        Model name to use for embeddings (default: ``text-embedding-bge-m3``).
    timeout : int
        Request timeout in seconds (default: 30).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: str = "text-embedding-bge-m3",
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

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

        Raises
        ------
        requests.RequestException
            If the LM Studio API call fails.
        """
        response = requests.post(
            f"{self._base_url}/v1/embeddings",
            json={"input": text, "model": self._model},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
