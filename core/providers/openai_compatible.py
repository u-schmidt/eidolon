"""OpenAI-compatible provider for Eidolon.

A single class that provides both LLM (chat completion) and Embedder
capabilities via any endpoint speaking the OpenAI HTTP dialect:

- LLM:      POST /v1/chat/completions
- Embedder: POST /v1/embeddings

This is the canonical client. It works with hosted APIs (OpenAI, Groq,
Together, OpenRouter, DeepInfra, Fireworks, ...), self-hosted servers
(vLLM, LiteLLM, a private gateway), and local tools that expose the same
API (Ollama 0.5+, LM Studio). The ``ollama`` and ``lm_studio`` providers
are thin presets of this class — see core/providers/ollama.py and
core/providers/lm_studio.py.

Authenticated endpoints: set ``api_key`` and it's sent as an
``Authorization: Bearer`` header.
"""

from __future__ import annotations

import time
from typing import Any

import requests

# Fallback response cap when neither the caller nor the factory supplies
# one. Matches ModelConfig.max_tokens' default so a bare provider (no
# config) behaves like a configured one, and is generous enough to give
# a reasoning model headroom instead of truncating mid-answer.
_DEFAULT_MAX_TOKENS = 4096

# Status codes worth retrying: 429 (rate limit) and 5xx (server-side,
# transient by nature). Anything else (400, 401, 404, ...) is a client
# error a retry can't fix, so it's returned to the caller immediately.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Exponential backoff base — a turn makes 1-3+ of these calls (detection,
# extraction, summarization), so a single transient network blip
# shouldn't silently drop that turn's memory. Two retries (three attempts
# total) at 0.5s/1.0s keeps the added worst-case latency modest.
_RETRY_BACKOFF_SECONDS = 0.5


class OpenAICompatible:
    """Provider for any OpenAI-compatible endpoint — chat completion and
    embeddings via the OpenAI HTTP API.

    Parameters
    ----------
    base_url : str
        Base URL of the endpoint (e.g. ``https://api.openai.com``,
        ``http://localhost:11434``). The ``/v1/...`` paths are appended.
    llm_model : str
        Model name for chat completions.
    embedder_model : str
        Model name for embeddings.
    api_key : str
        Optional API key for authenticated endpoints — sent as
        ``Authorization: Bearer <key>`` when non-empty (default: ``""``).
    timeout : int
        Request timeout in seconds (default: 60).
    max_tokens : int
        Default response cap for chat completions (default: 4096).
    max_retries : int
        Retries on a transient failure (connection error, timeout, 429,
        5xx) before giving up — 2 extra attempts by default, so 3 total
        (default: 2). Client errors (4xx other than 429) are never
        retried since retrying can't change the outcome.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        llm_model: str = "",
        embedder_model: str = "",
        api_key: str = "",
        timeout: int = 60,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        max_retries: int = 2,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._llm_model = llm_model
        self._embedder_model = embedder_model
        self._api_key = api_key
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _post_with_retry(self, url: str, payload: dict[str, Any]) -> requests.Response:
        """POST with retry-on-transient-failure (see ``max_retries`` above).

        A connection error/timeout, or a 429/5xx response, triggers a
        retry with exponential backoff; anything else (including a 4xx
        response) is returned immediately for the caller to handle.
        """
        attempt = 0
        while True:
            try:
                response = requests.post(url, json=payload, headers=self._headers(), timeout=self._timeout)
            except requests.exceptions.RequestException:
                if attempt >= self._max_retries:
                    raise
            else:
                if response.status_code not in _RETRYABLE_STATUS_CODES or attempt >= self._max_retries:
                    return response

            time.sleep(_RETRY_BACKOFF_SECONDS * (2**attempt))
            attempt += 1

    # ── LLMProvider ─────────────────────────────────────────

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request via POST /v1/chat/completions.

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

        response = self._post_with_retry(
            f"{self._base_url}/v1/chat/completions",
            {
                "model": self._llm_model,
                "messages": messages,
                "max_tokens": tokens,
                "stream": False,
            },
        )

        if not response.ok:
            raise RuntimeError(
                f"OpenAI-compatible API error ({response.status_code}): {response.text[:200]}"
            )

        data = response.json()

        # OpenAI-compatible format: {"choices": [{"message": {"content": "..."}}]}
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]

        if "error" in data:
            raise RuntimeError(f"LLM model '{self._llm_model}' error: {data['error']}")

        raise RuntimeError(f"Unexpected chat completion response format: {str(data)[:200]}")

    # ── EmbedderProvider ────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector via POST /v1/embeddings.

        Parameters
        ----------
        text : str
            The text to embed.

        Returns
        -------
        list[float]
            The embedding vector.
        """
        response = self._post_with_retry(
            f"{self._base_url}/v1/embeddings",
            {"model": self._embedder_model, "input": text},
        )

        if not response.ok:
            raise RuntimeError(
                f"OpenAI-compatible embed API error ({response.status_code}): {response.text[:200]}"
            )

        data = response.json()

        # OpenAI-compatible format: {"data": [{"embedding": [...]}]}
        if "data" in data and len(data["data"]) > 0:
            return data["data"][0]["embedding"]

        if "error" in data:
            raise RuntimeError(f"Embed model '{self._embedder_model}' error: {data['error']}")

        raise RuntimeError(f"Unexpected embedding response format: {str(data)[:200]}")
