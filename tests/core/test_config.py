"""Tests for configuration loading and provider factory."""

from __future__ import annotations

import os
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest

from core.config import (
    ConfigError,
    DebugConfig,
    EidolonConfig,
    FactsConfig,
    ModelConfig,
    RetrievalConfig,
    StorageConfig,
)
from core.config_loader import ConfigLoader
from core.provider_factory import ProviderFactory
from core.trace.null_tracer import NullTracer
from core.trace.safe_tracer import SafeTracer
from core.trace.sqlite_tracer import SqliteTracer


class TestModelConfig:
    def test_is_complete_true(self):
        cfg = ModelConfig(provider="ollama", base_url="http://localhost:11434", model="llama3.2")
        assert cfg.is_complete() is True

    def test_is_complete_false_missing_base_url(self):
        cfg = ModelConfig(provider="ollama", model="llama3.2")
        assert cfg.is_complete() is False

    def test_is_complete_false_empty(self):
        assert ModelConfig().is_complete() is False


class TestEidolonConfig:
    def test_defaults(self):
        config = EidolonConfig()
        assert config.storage.provider == "sqlite"
        assert config.models["llm"].provider == "simulated"
        assert config.models["embedder"].provider == "simulated"
        assert config.retrieval.semantic == 0.4
        assert config.retrieval.related_topics_new == 6
        assert config.retrieval.related_topics_continue == 3
        assert config.retrieval.memory_items_new == 4
        assert config.retrieval.memory_items_continue == 2
        assert config.retrieval.recent_topics == 2
        assert config.facts.consolidation_interval == 20
        assert config.facts.target_max == 100
        assert config.language == "en"
        assert config.timezone == "UTC"
        assert config.debug.enabled is False

    def test_zero_config_models_are_complete(self):
        """`python tools/cli.py` with no config file must keep working —
        the zero-config simulated defaults must be usable, not just present."""
        config = EidolonConfig()
        assert config.models["llm"].is_complete()
        assert config.models["embedder"].is_complete()

    def test_debug_config_defaults(self):
        debug = DebugConfig()
        assert debug.enabled is False
        assert debug.trace_db_path.endswith("trace.db")

    def test_custom_values(self):
        config = EidolonConfig(
            storage=StorageConfig(provider="sqlite", path="/tmp/test.db"),
            models={
                "llm": ModelConfig(provider="ollama", base_url="http://10.0.0.5:11434", model="llama3.2"),
                "embedder": ModelConfig(provider="lm_studio", base_url="http://10.0.0.5:1234", model="bge-m3"),
            },
            retrieval=RetrievalConfig(
                semantic=0.5, tag_overlap=0.3, person=0.1, temporal=0.1,
                related_topics_new=9, related_topics_continue=4,
                memory_items_new=5, memory_items_continue=3,
            ),
            language="de",
            timezone="Europe/Berlin",
        )
        assert config.storage.path == "/tmp/test.db"
        assert config.models["llm"].provider == "ollama"
        assert config.models["llm"].base_url == "http://10.0.0.5:11434"
        assert config.models["llm"].model == "llama3.2"
        assert config.models["embedder"].provider == "lm_studio"
        assert config.models["embedder"].model == "bge-m3"
        assert config.retrieval.semantic == 0.5
        assert config.retrieval.related_topics_new == 9
        assert config.retrieval.related_topics_continue == 4
        assert config.retrieval.memory_items_new == 5
        assert config.retrieval.memory_items_continue == 3
        assert config.language == "de"
        assert config.timezone == "Europe/Berlin"

    def test_require_model_returns_complete_config(self):
        config = EidolonConfig(
            models={"llm": ModelConfig(provider="ollama", base_url="http://x", model="y")}
        )
        cfg = config.require_model("llm")
        assert cfg.provider == "ollama"

    def test_require_model_raises_when_missing(self):
        config = EidolonConfig(models={})
        with pytest.raises(ConfigError, match="models.llm"):
            config.require_model("llm")

    def test_require_model_raises_when_incomplete(self):
        """The exact bug that started this redesign: a provider name set
        but no base_url — must fail loudly, not silently default elsewhere."""
        config = EidolonConfig(models={"llm": ModelConfig(provider="ollama")})
        with pytest.raises(ConfigError):
            config.require_model("llm")

    def test_require_model_error_mentions_cli_flags(self):
        config = EidolonConfig(models={})
        with pytest.raises(ConfigError, match="--judge-provider"):
            config.require_model("judge")


class TestConfigLoader:
    def test_defaults_without_sources(self):
        loader = ConfigLoader()
        config = loader.load()
        assert config.storage.provider == "sqlite"
        assert config.models["llm"].provider == "simulated"

    def test_yaml_loading(self):
        yaml_content = """
storage:
  provider: sqlite
  path: /tmp/test.db
models:
  llm:
    provider: ollama
    base_url: http://10.0.0.5:11434
    model: llama3.2
  embedder:
    provider: lm_studio
    base_url: http://10.0.0.5:1234
    model: bge-m3
retrieval:
  semantic: 0.5
  tag_overlap: 0.3
  related_topics_new: 9
  related_topics_continue: 4
language: de
timezone: Europe/Berlin
"""
        tmp = tempfile.mkdtemp()
        yaml_path = str(Path(tmp) / "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        loader = ConfigLoader(config_path=yaml_path)
        config = loader.load()

        assert config.storage.path == "/tmp/test.db"
        assert config.models["llm"].provider == "ollama"
        assert config.models["llm"].base_url == "http://10.0.0.5:11434"
        assert config.models["llm"].model == "llama3.2"
        assert config.models["embedder"].provider == "lm_studio"
        assert config.models["embedder"].base_url == "http://10.0.0.5:1234"
        assert config.models["embedder"].model == "bge-m3"
        assert config.retrieval.semantic == 0.5
        assert config.retrieval.related_topics_new == 9
        assert config.retrieval.related_topics_continue == 4
        assert config.language == "de"
        assert config.timezone == "Europe/Berlin"

    def test_yaml_models_block_replaces_zero_config_default(self):
        """A YAML llm: block must fully replace the zero-config `simulated`
        default, not merge into it and leave stale fields behind."""
        yaml_content = """
models:
  llm:
    provider: ollama
    base_url: http://localhost:11434
    model: deepseek-v4
"""
        tmp = tempfile.mkdtemp()
        yaml_path = str(Path(tmp) / "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = ConfigLoader(config_path=yaml_path).load()
        assert config.models["llm"].provider == "ollama"
        assert config.models["llm"].base_url == "http://localhost:11434"
        # embedder untouched by this YAML — stays at the zero-config default
        assert config.models["embedder"].provider == "simulated"

    def test_yaml_additional_model_role(self):
        """A role beyond llm/embedder (e.g. judge) just works — no code
        changes needed to add a new role."""
        yaml_content = """
models:
  judge:
    provider: ollama
    base_url: http://localhost:11434
    model: llama3.2:3b
"""
        tmp = tempfile.mkdtemp()
        yaml_path = str(Path(tmp) / "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = ConfigLoader(config_path=yaml_path).load()
        assert config.models["judge"].provider == "ollama"
        assert config.require_model("judge").model == "llama3.2:3b"

    def test_yaml_file_not_found(self):
        loader = ConfigLoader(config_path="/nonexistent/config.yaml")
        config = loader.load()
        assert config.storage.provider == "sqlite"

    def test_env_vars_override_yaml(self):
        yaml_content = """
models:
  llm:
    provider: simulated
    base_url: local
    model: default-model
"""
        tmp = tempfile.mkdtemp()
        yaml_path = str(Path(tmp) / "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        os.environ["EIDOLON_LLM_PROVIDER"] = "lm_studio"
        os.environ["EIDOLON_LLM_BASE_URL"] = "http://localhost:1234"
        os.environ["EIDOLON_LLM_MODEL"] = "deepseek-v4"

        loader = ConfigLoader(config_path=yaml_path)
        config = loader.load()

        assert config.models["llm"].provider == "lm_studio"
        assert config.models["llm"].base_url == "http://localhost:1234"
        assert config.models["llm"].model == "deepseek-v4"

        del os.environ["EIDOLON_LLM_PROVIDER"]
        del os.environ["EIDOLON_LLM_BASE_URL"]
        del os.environ["EIDOLON_LLM_MODEL"]

    def test_env_provider_switch_resets_stale_fields(self):
        """Switching provider via env var must not leak base_url/model from
        the previous provider (here: the zero-config `simulated` default)."""
        os.environ["EIDOLON_LLM_PROVIDER"] = "ollama"

        loader = ConfigLoader()
        config = loader.load()

        assert config.models["llm"].provider == "ollama"
        assert config.models["llm"].base_url == ""  # not "local" (the simulated default)

        del os.environ["EIDOLON_LLM_PROVIDER"]

    def test_cli_args_override_env(self):
        os.environ["EIDOLON_LLM_PROVIDER"] = "simulated"
        os.environ["EIDOLON_LLM_MODEL"] = "default-model"

        args = Namespace(
            llm_provider="lm_studio",
            llm_base_url="http://localhost:1234",
            llm_model="cli-model",
            embedder_provider="ollama",
            embedder_base_url="http://localhost:11434",
            embedder_model="nomic-embed-text",
            language="de",
            timezone="Europe/Berlin",
            storage_provider=None,
            storage_path="/cli/test.db",
        )

        loader = ConfigLoader(cli_args=args)
        config = loader.load()

        assert config.models["llm"].provider == "lm_studio"
        assert config.models["llm"].base_url == "http://localhost:1234"
        assert config.models["llm"].model == "cli-model"
        assert config.models["embedder"].provider == "ollama"
        assert config.models["embedder"].model == "nomic-embed-text"
        assert config.language == "de"
        assert config.timezone == "Europe/Berlin"
        assert config.storage.path == "/cli/test.db"
        assert config.storage.provider == "sqlite"

        del os.environ["EIDOLON_LLM_PROVIDER"]
        del os.environ["EIDOLON_LLM_MODEL"]

    def test_cli_provider_switch_resets_stale_fields(self):
        """The actual bug that started this redesign: --llm-provider ollama
        with no --llm-base-url must not silently keep the simulated
        default's base_url="local" — the resulting ModelConfig must be
        incomplete (and therefore fail loudly via require_model) rather
        than usable with the wrong URL."""
        args = Namespace(llm_provider="ollama", storage_provider=None, storage_path=None)

        config = ConfigLoader(cli_args=args).load()

        assert config.models["llm"].provider == "ollama"
        assert config.models["llm"].base_url == ""
        assert config.models["llm"].model == ""
        assert not config.models["llm"].is_complete()
        with pytest.raises(ConfigError):
            config.require_model("llm")

    def test_env_int_conversion(self):
        os.environ["EIDOLON_LLM_MAX_TOKENS"] = "2048"
        os.environ["EIDOLON_RETRIEVAL_SEMANTIC"] = "0.6"

        loader = ConfigLoader()
        config = loader.load()

        assert config.models["llm"].max_tokens == 2048
        assert config.retrieval.semantic == 0.6

        del os.environ["EIDOLON_LLM_MAX_TOKENS"]
        del os.environ["EIDOLON_RETRIEVAL_SEMANTIC"]

    def test_env_retrieval_limits(self):
        os.environ["EIDOLON_RETRIEVAL_RELATED_TOPICS_NEW"] = "9"
        os.environ["EIDOLON_RETRIEVAL_RELATED_TOPICS_CONTINUE"] = "4"
        os.environ["EIDOLON_RETRIEVAL_MEMORY_ITEMS_NEW"] = "5"
        os.environ["EIDOLON_RETRIEVAL_MEMORY_ITEMS_CONTINUE"] = "3"
        os.environ["EIDOLON_RETRIEVAL_RECENT_TOPICS"] = "1"

        config = ConfigLoader().load()

        assert config.retrieval.related_topics_new == 9
        assert config.retrieval.related_topics_continue == 4
        assert config.retrieval.memory_items_new == 5
        assert config.retrieval.memory_items_continue == 3
        assert config.retrieval.recent_topics == 1

        del os.environ["EIDOLON_RETRIEVAL_RELATED_TOPICS_NEW"]
        del os.environ["EIDOLON_RETRIEVAL_RELATED_TOPICS_CONTINUE"]
        del os.environ["EIDOLON_RETRIEVAL_MEMORY_ITEMS_NEW"]
        del os.environ["EIDOLON_RETRIEVAL_MEMORY_ITEMS_CONTINUE"]
        del os.environ["EIDOLON_RETRIEVAL_RECENT_TOPICS"]

    def test_cli_retrieval_limits(self):
        args = Namespace(
            retrieval_related_topics_new=9, retrieval_related_topics_continue=4,
            retrieval_memory_items_new=5, retrieval_memory_items_continue=3,
            retrieval_recent_topics=1,
            storage_provider=None, storage_path=None,
        )
        config = ConfigLoader(cli_args=args).load()
        assert config.retrieval.related_topics_new == 9
        assert config.retrieval.related_topics_continue == 4
        assert config.retrieval.memory_items_new == 5
        assert config.retrieval.memory_items_continue == 3
        assert config.retrieval.recent_topics == 1

    def test_yaml_facts_section(self):
        yaml_content = """
facts:
  consolidation_interval: 15
  target_max: 75
"""
        tmp = tempfile.mkdtemp()
        yaml_path = str(Path(tmp) / "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = ConfigLoader(config_path=yaml_path).load()
        assert config.facts.consolidation_interval == 15
        assert config.facts.target_max == 75

    def test_env_facts_vars(self):
        os.environ["EIDOLON_FACTS_CONSOLIDATION_INTERVAL"] = "10"
        os.environ["EIDOLON_FACTS_TARGET_MAX"] = "50"

        config = ConfigLoader().load()

        assert config.facts.consolidation_interval == 10
        assert config.facts.target_max == 50

        del os.environ["EIDOLON_FACTS_CONSOLIDATION_INTERVAL"]
        del os.environ["EIDOLON_FACTS_TARGET_MAX"]

    def test_cli_facts_args(self):
        args = Namespace(
            facts_consolidation_interval=5, facts_target_max=30,
            storage_provider=None, storage_path=None,
        )
        config = ConfigLoader(cli_args=args).load()
        assert config.facts.consolidation_interval == 5
        assert config.facts.target_max == 30

    def test_env_timezone(self):
        os.environ["EIDOLON_TIMEZONE"] = "Europe/Berlin"

        config = ConfigLoader().load()
        assert config.timezone == "Europe/Berlin"

        del os.environ["EIDOLON_TIMEZONE"]

    def test_cli_timezone(self):
        args = Namespace(timezone="Europe/Berlin", storage_provider=None, storage_path=None)
        config = ConfigLoader(cli_args=args).load()
        assert config.timezone == "Europe/Berlin"

    def test_yaml_debug_section(self):
        yaml_content = """
debug:
  enabled: true
  trace_db_path: /tmp/custom-trace.db
"""
        tmp = tempfile.mkdtemp()
        yaml_path = str(Path(tmp) / "config.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        loader = ConfigLoader(config_path=yaml_path)
        config = loader.load()

        assert config.debug.enabled is True
        assert config.debug.trace_db_path == "/tmp/custom-trace.db"

    def test_env_debug_vars(self):
        os.environ["EIDOLON_DEBUG_ENABLED"] = "true"
        os.environ["EIDOLON_DEBUG_TRACE_DB_PATH"] = "/tmp/env-trace.db"

        loader = ConfigLoader()
        config = loader.load()

        assert config.debug.enabled is True
        assert config.debug.trace_db_path == "/tmp/env-trace.db"

        del os.environ["EIDOLON_DEBUG_ENABLED"]
        del os.environ["EIDOLON_DEBUG_TRACE_DB_PATH"]

    def test_cli_debug_args(self):
        args = Namespace(debug_enabled=True, debug_trace_db_path="/cli/trace.db")

        loader = ConfigLoader(cli_args=args)
        config = loader.load()

        assert config.debug.enabled is True
        assert config.debug.trace_db_path == "/cli/trace.db"


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
            models={
                "embedder": ModelConfig(provider="lm_studio", base_url="http://localhost:1234", model="bge-m3"),
            },
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
        config = EidolonConfig(
            models={"llm": ModelConfig(provider="nonexistent", base_url="http://x", model="y")}
        )
        factory = ProviderFactory(config)
        with pytest.raises(ValueError, match="Unknown LLM provider: nonexistent"):
            factory.create_llm()

    def test_unknown_embedder_raises(self):
        config = EidolonConfig(
            models={"embedder": ModelConfig(provider="nonexistent", base_url="http://x", model="y")}
        )
        factory = ProviderFactory(config)
        with pytest.raises(ValueError, match="Unknown embedder provider: nonexistent"):
            factory.create_embedder()

    def test_create_llm_raises_config_error_when_incomplete(self):
        """This is the actual fix: an incomplete role must never silently
        build a provider pointed at a guessed endpoint."""
        config = EidolonConfig(models={"llm": ModelConfig(provider="ollama")})
        factory = ProviderFactory(config)
        with pytest.raises(ConfigError):
            factory.create_llm()

    def test_create_embedder_raises_config_error_when_missing(self):
        config = EidolonConfig(models={})
        factory = ProviderFactory(config)
        with pytest.raises(ConfigError):
            factory.create_embedder()

    def test_create_llm_wires_configured_max_tokens(self):
        """The configured ModelConfig.max_tokens must reach the provider —
        it used to be dead config, ignored in favor of a hardcoded default."""
        config = EidolonConfig(models={
            "llm": ModelConfig(
                provider="ollama", base_url="http://localhost:11434",
                model="deepseek", max_tokens=12345,
            ),
        })
        factory = ProviderFactory(config)
        llm = factory.create_llm()
        assert llm._max_tokens == 12345

    def test_create_llm_for_additional_role(self):
        config = EidolonConfig(models={
            "llm": ModelConfig(provider="simulated", base_url="local", model="simulated"),
            "judge": ModelConfig(provider="simulated", base_url="local", model="simulated"),
        })
        factory = ProviderFactory(config)
        judge = factory.create_llm("judge")
        assert judge.complete("test") is not None

    def test_create_tracer_null_when_disabled(self):
        config = EidolonConfig(debug=DebugConfig(enabled=False))
        factory = ProviderFactory(config)
        tracer = factory.create_tracer()
        assert isinstance(tracer, NullTracer)

    def test_create_tracer_sqlite_when_enabled(self):
        config = EidolonConfig(
            debug=DebugConfig(enabled=True, trace_db_path="/tmp/test_factory_trace.db"),
        )
        factory = ProviderFactory(config)
        tracer = factory.create_tracer()
        # Wrapped in SafeTracer so a broken trace backend can never take
        # down real extraction/storage — see core/trace/safe_tracer.py.
        assert isinstance(tracer, SafeTracer)
        assert isinstance(tracer._wrapped, SqliteTracer)
        tracer.close()
