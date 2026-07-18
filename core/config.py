"""Eidolon configuration model.

Pure dataclasses — no loading logic, no provider instantiation.
Defines the *shape* of configuration for storage, providers, LLM, embedder, and retrieval.

Multiple providers can be defined (e.g. ollama + lm_studio).
LLM and Embedder each reference one provider by name.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def default_data_dir() -> str:
    """Return the OS-appropriate default data directory for Eidolon.

    - Windows: ``%APPDATA%/eidolon``
    - macOS/Linux: ``~/.eidolon``
    """
    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA", os.path.expanduser("~"))
        return str(Path(appdata) / "eidolon")
    return str(Path.home() / ".eidolon")


def default_db_path() -> str:
    """Return the OS-appropriate default database path."""
    return str(Path(default_data_dir()) / "memory.db")


def default_providers() -> dict[str, ProviderEndpoint]:
    """Return default provider endpoints."""
    return {
        "simulated": ProviderEndpoint(
            name="simulated",
            base_url="http://localhost:1234",
        ),
    }


@dataclass
class StorageConfig:
    """Storage backend configuration."""

    provider: str = "sqlite"
    path: str = field(default_factory=default_db_path)


@dataclass
class ProviderEndpoint:
    """A single provider service endpoint.

    Defines the connection details for one provider (LM Studio, Ollama, etc.).
    Multiple endpoints can be defined in ``EidolonConfig.providers``.
    """

    name: str = "simulated"
    base_url: str = "http://localhost:1234"
    api_key: str = ""
    timeout: int = 60

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """LLM configuration.

    References a provider by name from ``EidolonConfig.providers``.
    """

    provider: str = "simulated"
    model: str = "deepseek-v4"
    max_tokens: int = 4096
    timeout: int = 60

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbedderConfig:
    """Embedder configuration.

    References a provider by name from ``EidolonConfig.providers``.
    """

    provider: str = "simulated"
    model: str = "text-embedding-bge-m3"
    timeout: int = 30
    dimension: int = 768

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    """Retrieval weight configuration."""

    semantic: float = 0.4
    tag_overlap: float = 0.3
    person: float = 0.2
    temporal: float = 0.1


def default_trace_db_path() -> str:
    """Return the OS-appropriate default trace database path."""
    return str(Path(default_data_dir()) / "trace.db")


@dataclass
class DebugConfig:
    """Debug-tracing configuration.

    Disabled by default — enabling writes detailed per-step trace events
    (prompts, responses, retrieval score breakdowns, durations, failures)
    to a separate SQLite file for inspection via ``tools/trace.py``.
    """

    enabled: bool = False
    trace_db_path: str = field(default_factory=default_trace_db_path)


@dataclass
class EidolonConfig:
    """Complete Eidolon configuration.

    This is the single source of truth for all configurable parameters.
    Loaded via ``ConfigLoader`` from YAML + CLI + env, then used by
    ``ProviderFactory`` to create provider instances.
    """

    storage: StorageConfig = field(default_factory=StorageConfig)
    providers: dict[str, ProviderEndpoint] = field(default_factory=default_providers)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    language: str = "en"

    def get_provider(self, name: str) -> ProviderEndpoint:
        """Get a provider endpoint by name, creating a default if missing."""
        if name in self.providers:
            return self.providers[name]
        ep = ProviderEndpoint(name=name)
        self.providers[name] = ep
        return ep

    def to_dict(self) -> dict[str, Any]:
        """Export config as a flat dict for logging/debugging."""
        return {
            "storage": {
                "provider": self.storage.provider,
                "path": self.storage.path,
            },
            "providers": {
                name: {
                    "base_url": ep.base_url,
                }
                for name, ep in self.providers.items()
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "max_tokens": self.llm.max_tokens,
            },
            "embedder": {
                "provider": self.embedder.provider,
                "model": self.embedder.model,
                "dimension": self.embedder.dimension,
            },
            "retrieval": {
                "semantic": self.retrieval.semantic,
                "tag_overlap": self.retrieval.tag_overlap,
                "person": self.retrieval.person,
                "temporal": self.retrieval.temporal,
            },
            "debug": {
                "enabled": self.debug.enabled,
                "trace_db_path": self.debug.trace_db_path,
            },
            "language": self.language,
        }
