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

from core.config import EidolonConfig, ModelConfig

# Environment variable prefix
_ENV_PREFIX = "EIDOLON_"

# Fields a `models.<role>` block (YAML) or CLI/env override may set.
_MODEL_ROLE_FIELDS = ("provider", "base_url", "model", "api_key", "timeout", "max_tokens", "dimension")


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
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install it with: pip install pyyaml"
            ) from exc

        with open(resolved) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        self._merge_dict(config, data)

    # ── Environment Variables ──────────────────────────────

    def _merge_env(self, config: EidolonConfig) -> None:
        _map: dict[str, tuple[str, ...]] = {
            "storage_provider": ("storage", "provider"),
            "storage_path": ("storage", "path"),
            "retrieval_semantic": ("retrieval", "semantic"),
            "retrieval_tag_overlap": ("retrieval", "tag_overlap"),
            "retrieval_person": ("retrieval", "person"),
            "retrieval_temporal": ("retrieval", "temporal"),
            "retrieval_related_topics_new": ("retrieval", "related_topics_new"),
            "retrieval_related_topics_continue": ("retrieval", "related_topics_continue"),
            "retrieval_memory_items_new": ("retrieval", "memory_items_new"),
            "retrieval_memory_items_continue": ("retrieval", "memory_items_continue"),
            "retrieval_recent_topics": ("retrieval", "recent_topics"),
            "facts_consolidation_interval": ("facts", "consolidation_interval"),
            "facts_target_max": ("facts", "target_max"),
            "debug_enabled": ("debug", "enabled"),
            "debug_trace_db_path": ("debug", "trace_db_path"),
            "language": ("language",),
            "timezone": ("timezone",),
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

        for role in ("llm", "embedder"):
            self._merge_model_role(config, role, {
                "provider": os.environ.get(_env_key(role, "provider")),
                "base_url": os.environ.get(_env_key(role, "base_url")),
                "model": os.environ.get(_env_key(role, "model")),
                "api_key": os.environ.get(_env_key(role, "api_key")),
                "timeout": self._coerce_int(os.environ.get(_env_key(role, "timeout"))),
                "max_tokens": self._coerce_int(os.environ.get(_env_key(role, "max_tokens"))),
                "dimension": self._coerce_int(os.environ.get(_env_key(role, "dimension"))),
            })

    # ── CLI Arguments ──────────────────────────────────────

    def _merge_cli(self, config: EidolonConfig, args: Namespace) -> None:
        cli_map: dict[str, tuple[str, ...]] = {
            "storage_provider": ("storage", "provider"),
            "storage_path": ("storage", "path"),
            "retrieval_semantic": ("retrieval", "semantic"),
            "retrieval_tag_overlap": ("retrieval", "tag_overlap"),
            "retrieval_person": ("retrieval", "person"),
            "retrieval_temporal": ("retrieval", "temporal"),
            "retrieval_related_topics_new": ("retrieval", "related_topics_new"),
            "retrieval_related_topics_continue": ("retrieval", "related_topics_continue"),
            "retrieval_memory_items_new": ("retrieval", "memory_items_new"),
            "retrieval_memory_items_continue": ("retrieval", "memory_items_continue"),
            "retrieval_recent_topics": ("retrieval", "recent_topics"),
            "facts_consolidation_interval": ("facts", "consolidation_interval"),
            "facts_target_max": ("facts", "target_max"),
            "debug_enabled": ("debug", "enabled"),
            "debug_trace_db_path": ("debug", "trace_db_path"),
            "language": ("language",),
            "timezone": ("timezone",),
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

        for role in ("llm", "embedder"):
            self._merge_model_role(config, role, {
                "provider": getattr(args, f"{role}_provider", None),
                "base_url": getattr(args, f"{role}_base_url", None),
                "model": getattr(args, f"{role}_model", None),
                "timeout": getattr(args, f"{role}_timeout", None),
                "max_tokens": getattr(args, f"{role}_max_tokens", None),
                "dimension": getattr(args, f"{role}_dimension", None),
            })

    # ── Model Role Merge (shared by CLI + env) ─────────────

    def _merge_model_role(self, config: EidolonConfig, role: str, updates: dict[str, Any]) -> None:
        """Merge CLI/env overrides for one model role (llm, embedder).

        If ``provider`` is being changed to a different provider than what's
        currently set for this role, the whole ``ModelConfig`` is replaced
        fresh instead of mutating just the ``provider`` field — otherwise
        switching ``--llm-provider ollama`` would silently keep the
        zero-config ``simulated`` default's ``base_url="local"``,
        reproducing the exact silent-misroute bug this design fixes.
        """
        updates = {k: v for k, v in updates.items() if v is not None}
        if not updates:
            return

        current = config.models.get(role)
        new_provider = updates.get("provider")
        if new_provider and (current is None or current.provider != new_provider):
            current = ModelConfig(provider=new_provider)
        elif current is None:
            current = ModelConfig()

        for field_name, value in updates.items():
            if field_name in _MODEL_ROLE_FIELDS:
                setattr(current, field_name, value)

        config.models[role] = current

    @staticmethod
    def _coerce_int(value: str | None) -> int | None:
        return int(value) if value is not None else None

    # ── Dict Merge Helper ─────────────────────────────────

    def _merge_dict(self, config: EidolonConfig, data: dict[str, Any]) -> None:
        """Recursively merge a nested dict into the config object."""
        for key, value in data.items():
            if key == "storage" and isinstance(value, dict):
                self._merge_into(config.storage, value)
            elif key == "models" and isinstance(value, dict):
                self._merge_models_yaml(config, value)
            elif key == "retrieval" and isinstance(value, dict):
                self._merge_into(config.retrieval, value)
            elif key == "facts" and isinstance(value, dict):
                self._merge_into(config.facts, value)
            elif key == "debug" and isinstance(value, dict):
                self._merge_into(config.debug, value)
            elif hasattr(config, key):
                setattr(config, key, value)

    def _merge_models_yaml(self, config: EidolonConfig, data: dict[str, Any]) -> None:
        """Merge the ``models:`` YAML block.

        Each role's block constructs a *fresh* ``ModelConfig`` — full
        replacement, not an in-place merge — so a role always reflects
        exactly what's specified in YAML, with no stale fields surviving
        from the zero-config ``simulated`` default.
        """
        for role, role_data in data.items():
            if not isinstance(role_data, dict):
                continue
            fields = {k: v for k, v in role_data.items() if k in _MODEL_ROLE_FIELDS}
            config.models[role] = ModelConfig(**fields)

    @staticmethod
    def _merge_into(obj: Any, data: dict[str, Any]) -> None:
        """Merge dict values into a dataclass instance."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
