"""LM Studio provider for Eidolon.

A single class that provides both LLM (chat completion) and Embedder capabilities
via a local LM Studio instance.

LLM endpoint: POST /v1/chat/completions
Embedder endpoint: POST /v1/embeddings
"""

from __future__ import annotations

import requests

# Fallback response cap when neither the caller nor the factory supplies
# one. Matches ModelConfig.max_tokens' default so a bare provider (no
# config) behaves like a configured one, and is generous enough to give
# a reasoning model headroom instead of truncating mid-answer.
_DEFAULT_MAX_TOKENS = 4096


class LMStudio:
    """Provider for LM Studio — supports both chat completion and embeddings.

    Parameters
    ----------
    base_url : str
        Base URL of the LM Studio instance (default: ``http://localhost:1234``).
    llm_model : str
        Model name for chat completions (default: ``deepseek-v4``).
    embedder_model : str
        Model name for embeddings (default: ``text-embedding-bge-m3``).
    api_key : str
        Optional API key (default: ``""``).
    timeout : int
        Request timeout in seconds (default: 30).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        llm_model: str = "deepseek-v4",
        embedder_model: str = "text-embedding-bge-m3",
        api_key: str = "",
        timeout: int = 30,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._llm_model = llm_model
        self._embedder_model = embedder_model
        self._api_key = api_key
        self._timeout = timeout
        self._max_tokens = max_tokens

    # ── LLMProvider ─────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request.

        Parameters
        ----------
        prompt : str
            The user/assistant prompt.
        system_prompt : str | None
            Optional system prompt.
        max_tokens : int | None
            Cap on the response. ``None`` uses this provider's configured
            default (``self._max_tokens``, from ``ModelConfig.max_tokens``).

        Returns
        -------
        str
            The completion text.
        """
        tokens = max_tokens if max_tokens is not None else self._max_tokens
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = requests.post(
            f"{self._base_url}/v1/chat/completions",
            json={
                "model": self._llm_model,
                "messages": messages,
                "max_tokens": tokens,
            },
            headers=headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # ── EmbedderProvider ────────────────────────────────────

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
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        response = requests.post(
            f"{self._base_url}/v1/embeddings",
            json={"input": text, "model": self._embedder_model},
            headers=headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
