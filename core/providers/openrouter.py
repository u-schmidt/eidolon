"""OpenRouter provider for Eidolon.

A thin preset of ``OpenAICompatible`` (see core/providers/openai_compatible.py)
with OpenRouter-flavored defaults. OpenRouter proxies the OpenAI HTTP
dialect for chat completions, so the shared client handles it directly —
this subclass only supplies sensible defaults for OpenRouter's endpoint.

LLM role only: OpenRouter does not proxy an embeddings endpoint, so
``embedder_model`` has no usable default here — configure a different
provider (``openai``, ``sentence_transformers``, ...) for the ``embedder``
role. Using this preset for ``embedder`` will fail at request time with a
404/error from OpenRouter, same as pointing any provider at a
nonexistent endpoint.
"""

from __future__ import annotations

from core.providers.openai_compatible import _DEFAULT_MAX_TOKENS, OpenAICompatible


class OpenRouter(OpenAICompatible):
    """OpenAI-compatible client preset for OpenRouter (LLM role only)."""

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api",
        llm_model: str = "openai/gpt-4o-mini",
        embedder_model: str = "",
        api_key: str = "",
        timeout: int = 60,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        super().__init__(base_url, llm_model, embedder_model, api_key, timeout, max_tokens)
