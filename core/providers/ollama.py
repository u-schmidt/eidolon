"""Ollama provider for Eidolon.

A single class that provides both LLM (chat completion) and Embedder capabilities
via an Ollama instance or any OpenAI-compatible endpoint.

LLM endpoint: POST /v1/chat/completions (OpenAI-compatible)
Embedder endpoint: POST /api/embed (Ollama native) or /v1/embeddings (OpenAI-compatible)

Uses the OpenAI-compatible format for chat, which works with:
- Modern Ollama (0.5+)
- LM Studio
- Custom OpenAI-compatible endpoints (like Alex's deepseek-v4-flash:cloud)
- OpenAI / Anthropic / etc.
"""

from __future__ import annotations

import requests


class Ollama:
    """Provider for Ollama / OpenAI-compatible endpoints.

    Supports both chat completion and embeddings via OpenAI-compatible API.

    Parameters
    ----------
    base_url : str
        Base URL of the Ollama instance (default: ``http://localhost:11434``).
    llm_model : str
        Model name for chat completions (default: ``deepseek-v4-flash:cloud``).
    embedder_model : str
        Model name for embeddings (default: ``nomic-embed-text``).
    api_key : str
        Optional API key for authenticated endpoints (default: ``""``).
    timeout : int
        Request timeout in seconds (default: 60).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        llm_model: str = "deepseek-v4-flash:cloud",
        embedder_model: str = "nomic-embed-text",
        api_key: str = "",
        timeout: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._llm_model = llm_model
        self._embedder_model = embedder_model
        self._api_key = api_key
        self._timeout = timeout

    # ── LLMProvider ─────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 100,
    ) -> str:
        """Send a chat completion request via OpenAI-compatible /v1/chat/completions.

        Parameters
        ----------
        prompt : str
            The user/assistant prompt.
        system_prompt : str | None
            Optional system prompt.
        max_tokens : int
            Maximum tokens in the response.

        Returns
        -------
        str
            The completion text.
        """
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
                "max_tokens": max_tokens,
                "stream": False,
            },
            headers=headers,
            timeout=self._timeout,
        )

        if not response.ok:
            raise RuntimeError(
                f"Ollama API error ({response.status_code}): {response.text[:200]}"
            )

        data = response.json()

        # OpenAI-compatible format: {"choices": [{"message": {"content": "..."}}]}
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]

        # Error response
        if "error" in data:
            raise RuntimeError(
                f"Ollama model '{self._llm_model}' error: {data['error']}"
            )

        raise RuntimeError(
            f"Unexpected Ollama response format: {str(data)[:200]}"
        )

    # ── EmbedderProvider ────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector via OpenAI-compatible /v1/embeddings.

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
            json={
                "model": self._embedder_model,
                "input": text,
            },
            headers=headers,
            timeout=self._timeout,
        )

        if not response.ok:
            raise RuntimeError(
                f"Ollama embed API error ({response.status_code}): {response.text[:200]}"
            )

        data = response.json()

        # OpenAI-compatible format: {"data": [{"embedding": [...]}]}
        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["embedding"]

        if "error" in data:
            raise RuntimeError(
                f"Ollama embed model '{self._embedder_model}' error: {data['error']}"
            )

        raise RuntimeError(
            f"Unexpected Ollama embed response format: {str(data)[:200]}"
        )
