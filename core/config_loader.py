"""Configuration loader for Eidolon.

Loads configuration from three sources (in order of precedence):
1. CLI arguments (highest)
2. Environment variables
3. YAML config file (lowest)

Each source overrides the previous one, so CLI args always win.
"""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from typing import Any

from core.config import EidolonConfig, ProviderEndpoint

# Environment variable prefix
_ENV_PREFIX = "EIDOLON_"


def _env_key(*parts: str) -> str:
    return _ENV_PREFIX + "_".join(parts).upper()


def _str_to_bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "y", "on")


class ConfigLoader:
    """Loads Eidolon configuration from YAML, env vars, and CLI args.

    Parameters
    ----------
    config_path : str | None
        Path to a YAML config file. If None, only env vars and defaults are used.
    cli_args : Namespace | None
        Parsed CLI arguments from argparse. If None, only env and YAML are used.
    """

    def __init__(
        self,
        config_path: str | None = None,
        cli_args: Namespace | None = None,
    ) -> None:
        self._config_path = config_path
        self._cli_args = cli_args

    def load(self) -> EidolonConfig:
        config = EidolonConfig()

        # 1. Start with defaults
        # 2. Override with YAML file
        if self._config_path:
            self._merge_yaml(config, self._config_path)

        # 3. Override with environment variables
        self._merge_env(config)

        # 4. Override with CLI arguments
        if self._cli_args is not None:
            self._merge_cli(config, self._cli_args)

        return config

    # ── YAML ───────────────────────────────────────────────

    def _merge_yaml(self, config: EidolonConfig, path: str) -> None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            return

        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install it with: pip install pyyaml"
            )

        with open(resolved) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        self._merge_dict(config, data)

    # ── Environment Variables ──────────────────────────────

    def _merge_env(self, config: EidolonConfig) -> None:
        _map: dict[str, tuple[str, ...]] = {
            "storage_provider": ("storage", "provider"),
            "storage_path": ("storage", "path"),
            "llm_provider": ("llm", "provider"),
            "llm_model": ("llm", "model"),
            "llm_max_tokens": ("llm", "max_tokens"),
            "llm_timeout": ("llm", "timeout"),
            "embedder_provider": ("embedder", "provider"),
            "embedder_model": ("embedder", "model"),
            "embedder_timeout": ("embedder", "timeout"),
            "embedder_dimension": ("embedder", "dimension"),
            "retrieval_semantic": ("retrieval", "semantic"),
            "retrieval_tag_overlap": ("retrieval", "tag_overlap"),
            "retrieval_person": ("retrieval", "person"),
            "retrieval_temporal": ("retrieval", "temporal"),
            "debug_enabled": ("debug", "enabled"),
            "debug_trace_db_path": ("debug", "trace_db_path"),
            "language": ("language",),
        }

        for env_suffix, attr_path in _map.items():
            key = _env_key(*env_suffix.split("_"))
            value = os.environ.get(key)
            if value is None:
                continue

            obj: Any = config
            for part in attr_path[:-1]:
                obj = getattr(obj, part)
            field_name = attr_path[-1]

            field_type = type(getattr(obj, field_name))
            if field_type is bool:
                converted: Any = _str_to_bool(value)
            elif field_type is int:
                converted = int(value)
            elif field_type is float:
                converted = float(value)
            else:
                converted = value

            setattr(obj, field_name, converted)

    # ── CLI Arguments ──────────────────────────────────────

    def _merge_cli(self, config: EidolonConfig, args: Namespace) -> None:
        cli_map: dict[str, tuple[str, ...]] = {
            "storage_provider": ("storage", "provider"),
            "storage_path": ("storage", "path"),
            "llm_provider": ("llm", "provider"),
            "llm_model": ("llm", "model"),
            "llm_max_tokens": ("llm", "max_tokens"),
            "llm_timeout": ("llm", "timeout"),
            "embedder_provider": ("embedder", "provider"),
            "embedder_model": ("embedder", "model"),
            "embedder_timeout": ("embedder", "timeout"),
            "embedder_dimension": ("embedder", "dimension"),
            "retrieval_semantic": ("retrieval", "semantic"),
            "retrieval_tag_overlap": ("retrieval", "tag_overlap"),
            "retrieval_person": ("retrieval", "person"),
            "retrieval_temporal": ("retrieval", "temporal"),
            "debug_enabled": ("debug", "enabled"),
            "debug_trace_db_path": ("debug", "trace_db_path"),
            "language": ("language",),
        }

        for cli_attr, attr_path in cli_map.items():
            if not hasattr(args, cli_attr):
                continue
            value = getattr(args, cli_attr)
            if value is None:
                continue
            obj: Any = config
            for part in attr_path[:-1]:
                obj = getattr(obj, part)
            setattr(obj, attr_path[-1], value)

    # ── Dict Merge Helper ─────────────────────────────────

    def _merge_dict(self, config: EidolonConfig, data: dict[str, Any]) -> None:
        """Recursively merge a nested dict into the config object."""
        for key, value in data.items():
            if key == "storage" and isinstance(value, dict):
                self._merge_into(config.storage, value)
            elif key == "providers" and isinstance(value, dict):
                self._merge_providers(config, value)
            elif key == "llm" and isinstance(value, dict):
                self._merge_into(config.llm, value)
            elif key == "embedder" and isinstance(value, dict):
                self._merge_into(config.embedder, value)
            elif key == "retrieval" and isinstance(value, dict):
                self._merge_into(config.retrieval, value)
            elif key == "debug" and isinstance(value, dict):
                self._merge_into(config.debug, value)
            elif hasattr(config, key):
                setattr(config, key, value)

    def _merge_providers(
        self, config: EidolonConfig, data: dict[str, Any]
    ) -> None:
        """Merge provider endpoint dicts into the providers map."""
        for name, endpoint_data in data.items():
            if not isinstance(endpoint_data, dict):
                continue
            if name in config.providers:
                ep = config.providers[name]
            else:
                ep = ProviderEndpoint(name=name)
                config.providers[name] = ep
            self._merge_into(ep, endpoint_data)

    @staticmethod
    def _merge_into(obj: Any, data: dict[str, Any]) -> None:
        """Merge dict values into a dataclass instance."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
