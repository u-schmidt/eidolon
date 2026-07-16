"""Tests for provider implementations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from providers.embedder.lm_studio import LMStudioEmbedder
from providers.llm.hermes_llm import HermesLLM


class TestHermesLLM:
    def test_complete_calls_wrapped_function(self):
        mock_fn = MagicMock(return_value="CONTINUE")
        llm = HermesLLM(mock_fn)

        result = llm.complete("prompt", system_prompt="system", max_tokens=50)

        assert result == "CONTINUE"
        mock_fn.assert_called_once_with("prompt", system="system", max_tokens=50)

    def test_complete_defaults(self):
        mock_fn = MagicMock(return_value="[]")
        llm = HermesLLM(mock_fn)

        result = llm.complete("test")

        assert result == "[]"
        mock_fn.assert_called_once_with("test", system=None, max_tokens=100)


class TestLMStudioEmbedder:
    def test_embed_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_response.raise_for_status.return_value = None

        with patch("providers.embedder.lm_studio.requests.post", return_value=mock_response) as mock_post:
            embedder = LMStudioEmbedder()
            result = embedder.embed("test text")

            assert result == [0.1, 0.2, 0.3]
            mock_post.assert_called_once_with(
                "http://localhost:1234/v1/embeddings",
                json={"input": "test text", "model": "text-embedding-bge-m3"},
                timeout=30,
            )

    def test_embed_custom_url_and_model(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.5]}]
        }
        mock_response.raise_for_status.return_value = None

        with patch("providers.embedder.lm_studio.requests.post", return_value=mock_response) as mock_post:
            embedder = LMStudioEmbedder(base_url="http://10.0.0.5:8080", model="custom-model")
            embedder.embed("hello")

            mock_post.assert_called_once_with(
                "http://10.0.0.5:8080/v1/embeddings",
                json={"input": "hello", "model": "custom-model"},
                timeout=30,
            )

    def test_embed_raises_on_http_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500")

        with patch("providers.embedder.lm_studio.requests.post", return_value=mock_response):
            embedder = LMStudioEmbedder()
            with pytest.raises(Exception, match="HTTP 500"):
                embedder.embed("test")


class TestSentenceTransformersEmbedder:
    def test_constructor(self):
        from providers.embedder.sentence_transformers import SentenceTransformersEmbedder

        embedder = SentenceTransformersEmbedder()
        assert embedder._model_name == "all-MiniLM-L6-v2"
        assert embedder._model is None  # lazy-loaded

    def test_constructor_custom_model(self):
        from providers.embedder.sentence_transformers import SentenceTransformersEmbedder

        embedder = SentenceTransformersEmbedder(model_name="paraphrase-multilingual-MiniLM-L12-v2")
        assert embedder._model_name == "paraphrase-multilingual-MiniLM-L12-v2"
