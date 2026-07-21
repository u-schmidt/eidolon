"""Ollama provider for Eidolon.

A thin preset of ``OpenAICompatible`` (see core/providers/openai_compatible.py)
with Ollama-flavored defaults. Modern Ollama (0.5+) exposes the OpenAI HTTP
API at ``/v1/chat/completions`` and ``/v1/embeddings``, so the shared client
handles it directly — this subclass only supplies sensible local defaults.

Kept as its own provider name (``provider: ollama``) for backward
compatibility and a familiar zero-config local default.
"""

from __future__ import annotations

from core.providers.openai_compatible import _DEFAULT_MAX_TOKENS, OpenAICompatible


class Ollama(OpenAICompatible):
    """OpenAI-compatible client preset for a local Ollama instance."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        llm_model: str = "deepseek-v4-flash:cloud",
        embedder_model: str = "nomic-embed-text",
        api_key: str = "",
        timeout: int = 60,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        super().__init__(base_url, llm_model, embedder_model, api_key, timeout, max_tokens)
