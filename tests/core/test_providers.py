"""Tests for provider implementations.

Each provider is a single class that fulfills both LLMProvider and
EmbedderProvider protocols. Tests verify the protocol contract is met.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.providers.lm_studio import LMStudio
from core.providers.ollama import Ollama
from core.providers.openai_compatible import OpenAICompatible


class TestLLMWrapperPattern:
    """Tests that a simple inline wrapper satisfies the LLMProvider protocol.

    This is the pattern used by the Hermes adapters — a thin inline class
    that wraps a callable instead of a separate provider class.
    """

    def test_inline_wrapper_satisfies_protocol(self):
        mock_fn = MagicMock(return_value="CONTINUE")

        class _Wrapper:
            def complete(self, prompt, system_prompt=None, max_tokens=100):
                return mock_fn(prompt, system=system_prompt, max_tokens=max_tokens)

        llm = _Wrapper()
        result = llm.complete("prompt", system_prompt="system", max_tokens=50)
        assert result == "CONTINUE"
        mock_fn.assert_called_once_with("prompt", system="system", max_tokens=50)

    def test_inline_wrapper_defaults(self):
        mock_fn = MagicMock(return_value="[]")

        class _Wrapper:
            def complete(self, prompt, system_prompt=None, max_tokens=100):
                return mock_fn(prompt, system=system_prompt, max_tokens=max_tokens)

        llm = _Wrapper()
        result = llm.complete("test")
        assert result == "[]"
        mock_fn.assert_called_once_with("test", system=None, max_tokens=100)


class TestOpenAICompatible:
    """Behavior suite for the canonical OpenAI-compatible client. The
    ollama/lm_studio providers are thin presets of this class, so their
    request/response behavior is covered here once."""

    def test_complete_sends_chat_request(self):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hello!"}}]}

        with patch("core.providers.openai_compatible.requests.post", return_value=mock_response) as mock_post:
            provider = OpenAICompatible(base_url="http://host:1234", llm_model="my-llm")
            result = provider.complete("Hi", system_prompt="Be nice", max_tokens=50)

            assert result == "Hello!"
            assert "/v1/chat/completions" in mock_post.call_args[0][0]
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["model"] == "my-llm"
            assert call_kwargs["json"]["max_tokens"] == 50
            assert len(call_kwargs["json"]["messages"]) == 2
            # No Authorization header when api_key is empty
            assert "Authorization" not in call_kwargs["headers"]

    def test_complete_sends_bearer_header_when_api_key_set(self):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi"}}]}

        with patch("core.providers.openai_compatible.requests.post", return_value=mock_response) as mock_post:
            OpenAICompatible(api_key="sk-secret").complete("Hi")
            assert mock_post.call_args[1]["headers"]["Authorization"] == "Bearer sk-secret"

    def test_default_max_tokens_and_override(self):
        assert OpenAICompatible()._max_tokens == 4096
        assert OpenAICompatible(max_tokens=9000)._max_tokens == 9000

    def test_complete_uses_configured_max_tokens_when_omitted(self):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        with patch("core.providers.openai_compatible.requests.post", return_value=mock_response) as mock_post:
            OpenAICompatible(max_tokens=7777).complete("Hi")
            assert mock_post.call_args[1]["json"]["max_tokens"] == 7777

    def test_complete_raises_on_http_error(self):
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        with (
            patch("core.providers.openai_compatible.requests.post", return_value=mock_response),
            pytest.raises(RuntimeError, match="OpenAI-compatible API error"),
        ):
            OpenAICompatible().complete("Hi")

    def test_embed_sends_embedding_request(self):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"data": [{"embedding": [0.5, 0.6, 0.7]}]}

        with patch("core.providers.openai_compatible.requests.post", return_value=mock_response) as mock_post:
            provider = OpenAICompatible(base_url="http://host:1234", embedder_model="my-embed")
            result = provider.embed("test text")

            assert result == [0.5, 0.6, 0.7]
            assert "/v1/embeddings" in mock_post.call_args[0][0]
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["model"] == "my-embed"
            assert call_kwargs["json"]["input"] == "test text"

    def test_embed_raises_on_http_error(self):
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with (
            patch("core.providers.openai_compatible.requests.post", return_value=mock_response),
            pytest.raises(RuntimeError, match="OpenAI-compatible embed API error"),
        ):
            OpenAICompatible().embed("test")


class TestPresetDefaults:
    """The ollama/lm_studio providers only differ from OpenAICompatible in
    their default base_url/model/timeout — behavior is inherited and tested
    in TestOpenAICompatible."""

    def test_lm_studio_defaults(self):
        provider = LMStudio()
        assert isinstance(provider, OpenAICompatible)
        assert provider._base_url == "http://localhost:1234"
        assert provider._llm_model == "deepseek-v4"
        assert provider._embedder_model == "text-embedding-bge-m3"
        assert provider._timeout == 30

    def test_ollama_defaults(self):
        provider = Ollama()
        assert isinstance(provider, OpenAICompatible)
        assert provider._base_url == "http://localhost:11434"
        assert provider._llm_model == "deepseek-v4-flash:cloud"
        assert provider._embedder_model == "nomic-embed-text"
        assert provider._timeout == 60

    def test_presets_accept_overrides_and_api_key(self):
        provider = Ollama(base_url="http://10.0.0.5:11434", llm_model="mistral", api_key="sk-test")
        assert provider._base_url == "http://10.0.0.5:11434"
        assert provider._llm_model == "mistral"
        assert provider._api_key == "sk-test"


class TestSentenceTransformersEmbedder:
    def test_constructor(self):
        from core.providers.sentence_transformers import SentenceTransformersEmbedder
        embedder = SentenceTransformersEmbedder()
        assert embedder._model_name == "all-MiniLM-L6-v2"
        assert embedder._model is None  # lazy-loaded

    def test_constructor_custom_model(self):
        from core.providers.sentence_transformers import SentenceTransformersEmbedder
        embedder = SentenceTransformersEmbedder(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        assert embedder._model_name == "paraphrase-multilingual-MiniLM-L12-v2"
