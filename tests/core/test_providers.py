"""Tests for provider implementations.

Each provider is a single class that fulfills both LLMProvider and
EmbedderProvider protocols. Tests verify the protocol contract is met.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from core.providers.lm_studio import LMStudio
from core.providers.ollama import Ollama
from core.providers.openai import OpenAI
from core.providers.openai_compatible import OpenAICompatible
from core.providers.openrouter import OpenRouter


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
            patch("core.providers.openai_compatible.time.sleep"),
            pytest.raises(RuntimeError, match="OpenAI-compatible embed API error"),
        ):
            OpenAICompatible().embed("test")


class TestOpenAICompatibleRetry:
    """Retry/backoff behavior on transient failures — see
    OpenAICompatible._post_with_retry. ``time.sleep`` is patched
    throughout so these tests don't actually wait out the backoff."""

    def test_retries_connection_error_then_succeeds(self):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}

        with (
            patch(
                "core.providers.openai_compatible.requests.post",
                side_effect=[requests.exceptions.ConnectionError("boom"), mock_response],
            ) as mock_post,
            patch("core.providers.openai_compatible.time.sleep") as mock_sleep,
        ):
            result = OpenAICompatible().complete("Hi")

            assert result == "ok"
            assert mock_post.call_count == 2
            mock_sleep.assert_called_once()

    def test_retries_5xx_then_succeeds(self):
        failure = MagicMock(ok=False, status_code=503, text="unavailable")
        success = MagicMock(ok=True)
        success.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}

        with (
            patch("core.providers.openai_compatible.requests.post", side_effect=[failure, success]) as mock_post,
            patch("core.providers.openai_compatible.time.sleep"),
        ):
            result = OpenAICompatible().embed("text")

            assert result == [0.1, 0.2]
            assert mock_post.call_count == 2

    def test_gives_up_after_max_retries_on_persistent_failure(self):
        with (
            patch(
                "core.providers.openai_compatible.requests.post",
                side_effect=requests.exceptions.Timeout("slow"),
            ) as mock_post,
            patch("core.providers.openai_compatible.time.sleep") as mock_sleep,
            pytest.raises(requests.exceptions.Timeout),
        ):
            OpenAICompatible(max_retries=2).complete("Hi")

        # 1 initial attempt + 2 retries = 3 calls, 2 sleeps in between
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2

    def test_does_not_retry_client_error(self):
        mock_response = MagicMock(ok=False, status_code=401, text="Unauthorized")

        with (
            patch("core.providers.openai_compatible.requests.post", return_value=mock_response) as mock_post,
            patch("core.providers.openai_compatible.time.sleep") as mock_sleep,
            pytest.raises(RuntimeError, match="OpenAI-compatible API error"),
        ):
            OpenAICompatible().complete("Hi")

        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()

    def test_max_retries_zero_disables_retry(self):
        with (
            patch(
                "core.providers.openai_compatible.requests.post",
                side_effect=requests.exceptions.ConnectionError("boom"),
            ) as mock_post,
            patch("core.providers.openai_compatible.time.sleep") as mock_sleep,
            pytest.raises(requests.exceptions.ConnectionError),
        ):
            OpenAICompatible(max_retries=0).complete("Hi")

        assert mock_post.call_count == 1
        mock_sleep.assert_not_called()


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
        assert provider._llm_model == "deepseek-v4.1-flash:cloud"
        assert provider._embedder_model == "nomic-embed-text"
        assert provider._timeout == 60

    def test_presets_accept_overrides_and_api_key(self):
        provider = Ollama(base_url="http://10.0.0.5:11434", llm_model="mistral", api_key="sk-test")
        assert provider._base_url == "http://10.0.0.5:11434"
        assert provider._llm_model == "mistral"
        assert provider._api_key == "sk-test"

    def test_openai_defaults(self):
        provider = OpenAI()
        assert isinstance(provider, OpenAICompatible)
        assert provider._base_url == "https://api.openai.com"
        assert provider._llm_model == "gpt-4o-mini"
        assert provider._embedder_model == "text-embedding-3-small"
        assert provider._timeout == 60

    def test_openrouter_defaults(self):
        provider = OpenRouter()
        assert isinstance(provider, OpenAICompatible)
        assert provider._base_url == "https://openrouter.ai/api"
        assert provider._llm_model == "openai/gpt-4o-mini"
        assert provider._embedder_model == ""
        assert provider._timeout == 60


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
