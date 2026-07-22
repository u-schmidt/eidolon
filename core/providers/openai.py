"""OpenAI provider for Eidolon.

A thin preset of ``OpenAICompatible`` (see core/providers/openai_compatible.py)
with OpenAI-flavored defaults. OpenAI's own API is the OpenAI HTTP dialect,
so the shared client handles it directly — this subclass only supplies
sensible defaults for OpenAI's hosted endpoint.

Functionally identical to pointing ``openai_compatible`` at
``https://api.openai.com`` yourself — this preset exists so the provider
name makes the target obvious in config, and to spare new users from
having to know the base_url/model names up front.
"""

from __future__ import annotations

from core.providers.openai_compatible import _DEFAULT_MAX_TOKENS, OpenAICompatible


class OpenAI(OpenAICompatible):
    """OpenAI-compatible client preset for OpenAI's own hosted API.

    ``api_key`` is required in practice (OpenAI rejects unauthenticated
    requests) but defaults to ``""`` like every other preset, so a missing
    key surfaces as OpenAI's own 401 rather than a config-time guess.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com",
        llm_model: str = "gpt-4o-mini",
        embedder_model: str = "text-embedding-3-small",
        api_key: str = "",
        timeout: int = 60,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        super().__init__(base_url, llm_model, embedder_model, api_key, timeout, max_tokens)
