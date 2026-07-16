"""Tests for configuration loading and provider factory."""

from __future__ import annotations

import os
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest

from core.config import (
    EidolonConfig,
    EmbedderConfig,
    LLMConfig,
    ProviderEndpoint,
    RetrievalConfig,
    StorageConfig,
)
from core.config_loader import ConfigLoader
from core.provider_factory import ProviderFactory


class TestEidolonConfig:
    def test_defaults(self):
        config = EidolonConfig()
        assert config.storage.provider == "sqlite"
        assert config.llm.provider == "simulated"
        assert config.embedder.provider == "simulated"
        assert config.retrieval.semantic == 0.4
        assert config.language == "en"

    def test_default_providers(self):
        config = EidolonConfig()
        assert "simulated" in config.providers
        assert config.providers["simulated"].base_url == "http://localhost:1234"

    def test_get_provider_found(self):
        config = EidolonConfig()
        config.providers["lm_studio"] = ProviderEndpoint(
            name="lm_studio", base_url="http://10.0.0.5:1234"
        )
        ep = config.get_provider("lm_studio")
        assert ep.base_url == "http://10.0.0.5:1234"

    def test_get_provider_fallback(self):
        config = EidolonConfig()
        ep = config.get_provider("nonexistent")
        assert ep.name == "nonexistent"  # auto-created
        assert "nonexistent" in config.providers

    def test_custom_values(self):
        config = EidolonConfig(
            storage=StorageConfig(provider="sqlite", path="/tmp/test.db"),
            providers={
                "ollama": ProviderEndpoint(name="ollama", base_url="http://10.0.0.5:11434"),
                "lm_studio": ProviderEndpoint(name="lm_studio", base_url="http://10.0.0.5:1234"),
            },
            llm=LLMConfig(provider="ollama", model="llama3.2"),
            embedder=EmbedderConfig(provider="lm_studio", model="bge-m3"),
            retrieval=RetrievalConfig(semantic=0.5, tag_overlap=0.3, person=0.1, temporal=0.1),
            language="de",
        )
        assert config.storage.path == "/tmp/test.db"
        assert config.llm.provider == "ollama"
        assert config.llm.model == "llama3.2"
        assert config.embedder.provider == "lm_studio"
        assert config.embedder.model == "bge-m3"
        assert config.retrieval.semantic == 0.5
        assert config.language == "de"
        assert config.providers["ollama"].base_url == "http://10.0.0.5:11434"
        assert config.providers["lm_studio"].base_url == "http://10.0.0.5:1234"


class TestConfigLoader:
    def test_defaults_without_sources(self):
        loader = ConfigLoader()
        config = loader.load()
        assert config.storage.provider == "sqlite"
        assert config.llm.provider == "simulated"

    def test_yaml_loading(self):
        yaml_content = """
storage:
  provider: sqlite
  path: /tmp/test.db
providers:
  ollama:
    base_url: http://10.0.0.5:11434
  lm_studio:
    base_url: http://10.0.0.5:1234
llm:
  provider: ollama
  model: llama3.2
embedder:
  provider: lm_studio
  model: bge-m3
retrieval:
  semantic: 0.5
  tag_overlap: 0.3
language: de
"""
        tmp = tempfile.mkdtemp()
        yaml_path = str(Path(tmp) / "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        loader = ConfigLoader(config_path=yaml_path)
        config = loader.load()

        assert config.storage.path == "/tmp/test.db"
        assert config.llm.provider == "ollama"
        assert config.llm.model == "llama3.2"
        assert config.embedder.provider == "lm_studio"
        assert config.embedder.model == "bge-m3"
        assert config.providers["ollama"].base_url == "http://10.0.0.5:11434"
        assert config.providers["lm_studio"].base_url == "http://10.0.0.5:1234"
        assert config.retrieval.semantic == 0.5
        assert config.language == "de"

    def test_yaml_file_not_found(self):
        loader = ConfigLoader(config_path="/nonexistent/config.yaml")
        config = loader.load()
        assert config.storage.provider == "sqlite"

    def test_env_vars_override_yaml(self):
        yaml_content = """
llm:
  provider: simulated
  model: default-model
"""
        tmp = tempfile.mkdtemp()
        yaml_path = str(Path(tmp) / "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        os.environ["EIDOLON_LLM_PROVIDER"] = "lm_studio"
        os.environ["EIDOLON_LLM_MODEL"] = "deepseek-v4"

        loader = ConfigLoader(config_path=yaml_path)
        config = loader.load()

        assert config.llm.provider == "lm_studio"
        assert config.llm.model == "deepseek-v4"

        del os.environ["EIDOLON_LLM_PROVIDER"]
        del os.environ["EIDOLON_LLM_MODEL"]

    def test_cli_args_override_env(self):
        os.environ["EIDOLON_LLM_PROVIDER"] = "simulated"
        os.environ["EIDOLON_LLM_MODEL"] = "default-model"

        args = Namespace(
            llm_provider="lm_studio",
            llm_model="cli-model",
            embedder_provider="ollama",
            embedder_model="nomic-embed-text",
            language="de",
            db_path="/cli/test.db",
            storage_provider=None,
            storage_path=None,
        )

        loader = ConfigLoader(cli_args=args)
        config = loader.load()

        assert config.llm.provider == "lm_studio"
        assert config.llm.model == "cli-model"
        assert config.embedder.provider == "ollama"
        assert config.embedder.model == "nomic-embed-text"
        assert config.language == "de"
        assert config.db_path == "/cli/test.db"
        assert config.storage.provider == "sqlite"

        del os.environ["EIDOLON_LLM_PROVIDER"]
        del os.environ["EIDOLON_LLM_MODEL"]

    def test_env_int_conversion(self):
        os.environ["EIDOLON_LLM_MAX_TOKENS"] = "2048"
        os.environ["EIDOLON_RETRIEVAL_SEMANTIC"] = "0.6"

        loader = ConfigLoader()
        config = loader.load()

        assert config.llm.max_tokens == 2048
        assert config.retrieval.semantic == 0.6

        del os.environ["EIDOLON_LLM_MAX_TOKENS"]
        del os.environ["EIDOLON_RETRIEVAL_SEMANTIC"]


class TestProviderFactory:
    def test_create_sqlite_storage(self):
        config = EidolonConfig(
            storage=StorageConfig(provider="sqlite", path="/tmp/test_factory.db"),
        )
        factory = ProviderFactory(config)
        store = factory.create_storage()
        assert store is not None
        store.close()

    def test_create_simulated_llm(self):
        config = EidolonConfig()
        factory = ProviderFactory(config)
        llm = factory.create_llm()
        result = llm.complete("test")
        assert isinstance(result, str)

    def test_create_simulated_embedder(self):
        config = EidolonConfig()
        factory = ProviderFactory(config)
        embedder = factory.create_embedder()
        vec = embedder.embed("test")
        assert isinstance(vec, list)
        assert len(vec) == 768

    def test_create_lm_studio_embedder(self):
        config = EidolonConfig(
            providers={
                "lm_studio": ProviderEndpoint(name="lm_studio", base_url="http://localhost:1234"),
            },
            embedder=EmbedderConfig(provider="lm_studio"),
        )
        factory = ProviderFactory(config)
        embedder = factory.create_embedder()
        assert embedder is not None

    def test_unknown_storage_raises(self):
        config = EidolonConfig(storage=StorageConfig(provider="mongodb"))
        factory = ProviderFactory(config)
        with pytest.raises(ValueError, match="Unknown storage provider: mongodb"):
            factory.create_storage()

    def test_unknown_llm_raises(self):
        config = EidolonConfig(llm=LLMConfig(provider="nonexistent"))
        factory = ProviderFactory(config)
        with pytest.raises(ValueError, match="Unknown LLM provider: nonexistent"):
            factory.create_llm()

    def test_unknown_embedder_raises(self):
        config = EidolonConfig(embedder=EmbedderConfig(provider="nonexistent"))
        factory = ProviderFactory(config)
        with pytest.raises(ValueError, match="Unknown embedder provider: nonexistent"):
            factory.create_embedder()
