"""Eidolon configuration model.

Pure dataclasses — no loading logic, no provider instantiation.
Defines the *shape* of configuration for storage, models, and retrieval.

Each model role (llm, embedder, judge, ...) is a fully self-contained
ModelConfig — provider, connection, and model name together. There is no
shared "providers" registry and no cross-referencing by name: a role
that isn't fully specified raises ConfigError at the point of use rather
than silently falling back to a guessed endpoint.
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


def default_trace_db_path() -> str:
    """Return the OS-appropriate default trace database path."""
    return str(Path(default_data_dir()) / "trace.db")


class ConfigError(Exception):
    """Raised for missing or incomplete configuration.

    Fails loudly at the point of use instead of silently falling back to
    a guessed default endpoint.
    """


@dataclass
class StorageConfig:
    """Storage backend configuration.

    `path` is used by `provider: sqlite`; `host`/`port`/`dbname`/`user`/
    `password` by `provider: postgres` — see core/stores/postgres_store.py.
    Discrete fields rather than a single DSN string, matching this
    project's "self-contained, explicit fields, no opaque connection
    blob" philosophy (see ModelConfig below).
    """

    provider: str = "sqlite"
    path: str = field(default_factory=default_db_path)
    host: str = "127.0.0.1"
    port: int = 5432
    dbname: str = "eidolon"
    user: str = "eidolon"
    password: str = ""


@dataclass
class ModelConfig:
    """A single, self-contained model endpoint.

    Bundles provider, connection details, and model name together — no
    shared registry, no cross-references by name. Each model role (llm,
    embedder, judge, ...) gets its own independent ModelConfig.
    """

    provider: str = ""
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    timeout: int = 60
    max_tokens: int = 4096  # meaningful for LLM-role configs only
    dimension: int = 768  # meaningful for embedder-role configs only

    def is_complete(self) -> bool:
        """Whether provider, base_url, and model are all set."""
        return bool(self.provider and self.base_url and self.model)


def default_models() -> dict[str, ModelConfig]:
    """Default model roles.

    ``simulated`` stays as the zero-config default for ``llm``/``embedder``
    only, so ``python tools/cli.py`` keeps working with no YAML file. Real
    providers (ollama, lm_studio, ...) must be fully specified — see
    ``EidolonConfig.require_model``. This is a documented, intentional
    testing default, not a silent fallback for real providers.
    """
    return {
        "llm": ModelConfig(provider="simulated", base_url="local", model="simulated"),
        "embedder": ModelConfig(provider="simulated", base_url="local", model="simulated"),
    }


@dataclass
class RetrievalConfig:
    """Retrieval weight and result-count configuration."""

    semantic: float = 0.4
    tag_overlap: float = 0.3
    person: float = 0.2
    temporal: float = 0.1
    related_topics_new: int = 6       # was hardcoded 5
    related_topics_continue: int = 3  # was hardcoded 2
    memory_items_new: int = 4         # was hardcoded 3
    memory_items_continue: int = 2    # was hardcoded 1
    recent_topics: int = 5            # most-recently-closed topics, regardless of relevance —
                                       # NOT filtered by min_topic_messages below (see that field's
                                       # docstring): this surface is unconditional recency by design.
    cluster_summaries: int = 3        # recurring-topic clusters competing for injection per turn
    cluster_recent_topics: int = 2    # member topics shown in a cluster's "newest: ..." tail
    min_topic_messages: int = 3       # soft-hide topics at/below this message count from
                                       # search_topics only ("mini-topics" — one-off 1-2 exchange
                                       # asides). Deliberately NOT applied to get_recent_topics —
                                       # that surface is unconditional recency, not relevance, so a
                                       # size filter there would contradict its whole purpose.
                                       # Nothing is deleted; a lower/higher value instantly
                                       # reclassifies every topic, old and new, since this is a
                                       # query-time filter, not a stored flag.


@dataclass
class FactsConfig:
    """Durable-facts profile consolidation configuration.

    See ``core/consolidation/fact_consolidator.py``.
    """

    consolidation_interval: int = 20  # reconcile every N user turns
    target_max: int = 100  # soft guidance to the LLM, not hard-enforced
    # Closed category list offered to the LLM during consolidation — not
    # hard-enforced (a fact under an unlisted category is still stored,
    # just logged), but keeps categories from drifting across passes.
    categories: list[str] = field(default_factory=lambda: [
        "identity", "relationship", "work", "health", "routine",
        "technology", "preference", "possession", "instruction",
    ])


@dataclass
class ClusteringConfig:
    """Recurring-topic clustering configuration.

    See ``core/consolidation/cluster_consolidator.py``.
    """

    enabled: bool = True
    shortlist_size: int = 5  # candidates shown to the LLM per topic close


@dataclass
class RelabelConfig:
    """Mid-conversation topic relabeling — see
    core/summarization/topic_relabeler.py.

    A topic's label starts empty ("") at creation instead of the raw
    first-message guess, and gets finalized one of two ways: this
    mechanism, once a still-pending topic accumulates
    ``message_threshold`` messages, or (always on, no config here)
    TopicSummarizer's existing close-time pass, whichever comes first.

    On by default: Mechanism A (close-time) alone leaves a still-open,
    growing conversation showing its wrong first-message-guess label
    for the topic's entire live duration — this is what actually fixes
    that. Cost is bounded to at most one extra LLM call per topic, ever.
    """

    enabled: bool = True
    message_threshold: int = 4  # ~2 exchanges — low enough to catch a
    # drifting topic quickly (the whole point), high enough that a
    # topic about to close anyway isn't relabeled for near-zero gain.


@dataclass
class DeploymentConfig:
    """Which core-as-a-service mode a caller (e.g. the Hermes adapter)
    runs in — see ROADMAP.md's "Core-as-a-service deployment mode".

    ``mode: "embedded"`` (the default) builds a real in-process
    ``core.Eidolon`` from the rest of this config, exactly as before —
    ``server_url``/``token``/``timeout`` are unused in that mode.
    ``mode: "server"`` instead builds a ``client.http_client.
    EidolonHttpClient`` pointed at ``server_url``, and the rest of this
    config (models/storage/facts/...) becomes irrelevant to the caller,
    since it's the separately-run server process's own config, not this
    one's — see tools/serve.py.
    """

    mode: str = "embedded"  # "embedded" | "server"
    server_url: str = "http://127.0.0.1:8420"
    token: str = ""  # via EIDOLON_DEPLOYMENT_TOKEN — excluded from to_dict()
    timeout: float = 30.0


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
class MaintenanceConfig:
    """Activity-gated stale-topic cleanup — see ROADMAP.md's "Topics
    never close on their own" entry.

    Disabled by default: enabling this changes behavior for
    already-stuck-open topics on upgrade (each closed topic runs a real
    LLM summarization call), so it's opt-in rather than on by default
    like ``ClusteringConfig``, which only adds behavior with no side
    effect on existing data.

    Deliberately *not* timer-driven — a topic is only ever closed as a
    side effect of some other real turn happening (any peer, any
    session), checked no more often than ``interval_hours``. Closing a
    topic independently of activity risks fragmenting a conversation a
    peer would have naturally continued: once closed, topic detection
    has nothing "active" left to compare against, so a returning peer's
    continuation gets forced into a new topic instead of resuming the
    old one.
    """

    enabled: bool = False
    interval_hours: float = 1.0
    stale_after_hours: float = 24.0  # mirrors tools/cli.py's --stale-older-than-hours default
    peer_id: str | None = None  # restrict the sweep to one peer; None = every peer


@dataclass
class EidolonConfig:
    """Complete Eidolon configuration.

    This is the single source of truth for all configurable parameters.
    Loaded via ``ConfigLoader`` from YAML + CLI + env, then used by
    ``ProviderFactory`` to create provider instances.
    """

    storage: StorageConfig = field(default_factory=StorageConfig)
    models: dict[str, ModelConfig] = field(default_factory=default_models)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    facts: FactsConfig = field(default_factory=FactsConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    relabel: RelabelConfig = field(default_factory=RelabelConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)
    language: str = "en"
    timezone: str = "UTC"  # IANA name (e.g. "Europe/Berlin"); used to display env time

    def require_model(self, role: str) -> ModelConfig:
        """Return the fully-specified ``ModelConfig`` for ``role``.

        Raises ``ConfigError`` if the role is missing or incomplete,
        instead of silently falling back to a guessed endpoint. Called
        lazily by ``ProviderFactory`` at the point a role is actually
        needed — an incomplete ``judge`` block sitting unused in YAML
        doesn't break anything until something actually asks for it.
        """
        cfg = self.models.get(role)
        if cfg is None or not cfg.is_complete():
            raise ConfigError(
                f"Model role '{role}' is not fully configured (need provider, "
                f"base_url, model) — check the 'models.{role}' section of your "
                f"eidolon.yaml, or pass --{role}-provider/--{role}-model/--{role}-base-url."
            )
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Export config as a flat dict for logging/debugging."""
        return {
            "storage": {
                "provider": self.storage.provider,
                "path": self.storage.path,
                "host": self.storage.host,
                "port": self.storage.port,
                "dbname": self.storage.dbname,
                "user": self.storage.user,
                # password intentionally excluded — this dict is for
                # logging/debugging, matching how ModelConfig's api_key
                # is excluded below.
            },
            "models": {
                role: {
                    "provider": m.provider,
                    "base_url": m.base_url,
                    "model": m.model,
                }
                for role, m in self.models.items()
            },
            "retrieval": {
                "semantic": self.retrieval.semantic,
                "tag_overlap": self.retrieval.tag_overlap,
                "person": self.retrieval.person,
                "temporal": self.retrieval.temporal,
                "related_topics_new": self.retrieval.related_topics_new,
                "related_topics_continue": self.retrieval.related_topics_continue,
                "memory_items_new": self.retrieval.memory_items_new,
                "memory_items_continue": self.retrieval.memory_items_continue,
                "recent_topics": self.retrieval.recent_topics,
                "cluster_summaries": self.retrieval.cluster_summaries,
                "cluster_recent_topics": self.retrieval.cluster_recent_topics,
            },
            "facts": {
                "consolidation_interval": self.facts.consolidation_interval,
                "target_max": self.facts.target_max,
            },
            "clustering": {
                "enabled": self.clustering.enabled,
                "shortlist_size": self.clustering.shortlist_size,
            },
            "relabel": {
                "enabled": self.relabel.enabled,
                "message_threshold": self.relabel.message_threshold,
            },
            "debug": {
                "enabled": self.debug.enabled,
                "trace_db_path": self.debug.trace_db_path,
            },
            "deployment": {
                "mode": self.deployment.mode,
                "server_url": self.deployment.server_url,
                "timeout": self.deployment.timeout,
                # token intentionally excluded — same reason as
                # storage.password/models.*.api_key above.
            },
            "maintenance": {
                "enabled": self.maintenance.enabled,
                "interval_hours": self.maintenance.interval_hours,
                "stale_after_hours": self.maintenance.stale_after_hours,
                "peer_id": self.maintenance.peer_id,
            },
            "language": self.language,
            "timezone": self.timezone,
        }
