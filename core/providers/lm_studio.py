"""LM Studio provider for Eidolon.

A thin preset of ``OpenAICompatible`` (see core/providers/openai_compatible.py)
with LM Studio-flavored defaults. LM Studio exposes the OpenAI HTTP API at
``/v1/chat/completions`` and ``/v1/embeddings``, so the shared client handles
it directly — this subclass only supplies sensible local defaults.

Kept as its own provider name (``provider: lm_studio``) for backward
compatibility and a familiar zero-config local default.
"""

from __future__ import annotations

from core.providers.openai_compatible import _DEFAULT_MAX_TOKENS, OpenAICompatible


class LMStudio(OpenAICompatible):
    """OpenAI-compatible client preset for a local LM Studio instance."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        llm_model: str = "deepseek-v4",
        embedder_model: str = "text-embedding-bge-m3",
        api_key: str = "",
        timeout: int = 30,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        super().__init__(base_url, llm_model, embedder_model, api_key, timeout, max_tokens)
