"""Tests for provider implementations.

Each provider is a single class that fulfills both LLMProvider and
EmbedderProvider protocols. Tests verify the protocol contract is met.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.providers.lm_studio import LMStudio
from core.providers.ollama import Ollama


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


class TestLMStudio:
    """Tests that LMStudio fulfills both LLMProvider and EmbedderProvider."""

    def test_constructor(self):
        provider = LMStudio()
        assert provider._base_url == "http://localhost:1234"
        assert provider._llm_model == "deepseek-v4"
        assert provider._embedder_model == "text-embedding-bge-m3"

    def test_constructor_custom(self):
        provider = LMStudio(
            base_url="http://10.0.0.5:8080",
            llm_model="custom-llm",
            embedder_model="custom-embed",
            api_key="sk-test",
            timeout=60,
        )
        assert provider._base_url == "http://10.0.0.5:8080"
        assert provider._llm_model == "custom-llm"
        assert provider._embedder_model == "custom-embed"
        assert provider._api_key == "sk-test"
        assert provider._timeout == 60

    def test_complete_sends_chat_request(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        mock_response.raise_for_status.return_value = None

        with patch("core.providers.lm_studio.requests.post", return_value=mock_response) as mock_post:
            provider = LMStudio()
            result = provider.complete("Hi", system_prompt="Be nice", max_tokens=50)

            assert result == "Hello!"
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["model"] == "deepseek-v4"
            assert call_kwargs["json"]["max_tokens"] == 50
            assert len(call_kwargs["json"]["messages"]) == 2

    def test_default_max_tokens(self):
        assert LMStudio()._max_tokens == 4096
        assert LMStudio(max_tokens=9000)._max_tokens == 9000

    def test_complete_uses_configured_max_tokens_when_omitted(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_response.raise_for_status.return_value = None

        with patch("core.providers.lm_studio.requests.post", return_value=mock_response) as mock_post:
            LMStudio(max_tokens=7777).complete("Hi")
            assert mock_post.call_args[1]["json"]["max_tokens"] == 7777

    def test_embed_sends_embedding_request(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_response.raise_for_status.return_value = None

        with patch("core.providers.lm_studio.requests.post", return_value=mock_response) as mock_post:
            provider = LMStudio()
            result = provider.embed("test text")

            assert result == [0.1, 0.2, 0.3]
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["model"] == "text-embedding-bge-m3"
            assert call_kwargs["json"]["input"] == "test text"

    def test_embed_raises_on_http_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500")

        with patch("core.providers.lm_studio.requests.post", return_value=mock_response):
            provider = LMStudio()
            with pytest.raises(Exception, match="HTTP 500"):
                provider.embed("test")


class TestOllama:
    """Tests that Ollama fulfills both LLMProvider and EmbedderProvider."""

    def test_constructor(self):
        provider = Ollama()
        assert provider._base_url == "http://localhost:11434"
        assert provider._llm_model == "deepseek-v4-flash:cloud"
        assert provider._embedder_model == "nomic-embed-text"

    def test_constructor_custom(self):
        provider = Ollama(
            base_url="http://10.0.0.5:11434",
            llm_model="mistral",
            embedder_model="all-minilm",
            api_key="sk-test",
            timeout=120,
        )
        assert provider._base_url == "http://10.0.0.5:11434"
        assert provider._llm_model == "mistral"
        assert provider._embedder_model == "all-minilm"
        assert provider._api_key == "sk-test"
        assert provider._timeout == 120

    def test_complete_sends_chat_request(self):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from Ollama!"}}]
        }

        with patch("core.providers.ollama.requests.post", return_value=mock_response) as mock_post:
            provider = Ollama()
            result = provider.complete("Hi", system_prompt="Be nice", max_tokens=50)

            assert result == "Hello from Ollama!"
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["model"] == "deepseek-v4-flash:cloud"
            assert call_kwargs["json"]["max_tokens"] == 50
            assert len(call_kwargs["json"]["messages"]) == 2
            # No Authorization header when api_key is empty
            assert "Authorization" not in call_kwargs.get("headers", {})

    def test_complete_with_api_key(self):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }

        with patch("core.providers.ollama.requests.post", return_value=mock_response) as mock_post:
            provider = Ollama(api_key="sk-secret")
            provider.complete("Hi")

            headers = mock_post.call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer sk-secret"

    def test_default_max_tokens(self):
        assert Ollama()._max_tokens == 4096
        assert Ollama(max_tokens=9000)._max_tokens == 9000

    def test_complete_uses_configured_max_tokens_when_omitted(self):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        with patch("core.providers.ollama.requests.post", return_value=mock_response) as mock_post:
            Ollama(max_tokens=7777).complete("Hi")
            assert mock_post.call_args[1]["json"]["max_tokens"] == 7777

    def test_embed_sends_embedding_request(self):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "data": [{"embedding": [0.5, 0.6, 0.7]}]
        }

        with patch("core.providers.ollama.requests.post", return_value=mock_response) as mock_post:
            provider = Ollama()
            result = provider.embed("test text")

            assert result == [0.5, 0.6, 0.7]
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["model"] == "nomic-embed-text"
            assert call_kwargs["json"]["input"] == "test text"
            assert "/v1/embeddings" in mock_post.call_args[0][0]

    def test_embed_raises_on_http_error(self):
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("core.providers.ollama.requests.post", return_value=mock_response):
            provider = Ollama()
            with pytest.raises(RuntimeError, match="Ollama embed API error"):
                provider.embed("test")


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
