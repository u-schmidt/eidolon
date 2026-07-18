"""Provider factory for Eidolon.

Creates provider instances (LLM, Embedder, Storage) from configuration.
Each model role (``llm``, ``embedder``, and future roles like ``judge``)
is a fully self-contained ``ModelConfig`` — no shared endpoint registry,
no cross-referencing by name. A role that isn't fully configured raises
``ConfigError`` (via ``EidolonConfig.require_model``) instead of silently
falling back to a guessed endpoint.
"""

from __future__ import annotations

from core.config import EidolonConfig, ModelConfig
from core.interfaces import EmbedderProvider, LLMProvider, StorageProvider, Tracer
from core.stores.sqlite_store import SqliteStore
from core.trace.null_tracer import NullTracer
from core.trace.sqlite_tracer import SqliteTracer


class ProviderFactory:
    """Creates provider instances from an ``EidolonConfig``.

    Usage::

        config = ConfigLoader("config.yaml").load()
        factory = ProviderFactory(config)
        store = factory.create_storage()
        llm = factory.create_llm()          # role="llm"
        embedder = factory.create_embedder()  # role="embedder"
        judge = factory.create_llm("judge")   # any additional configured role
    """

    def __init__(self, config: EidolonConfig) -> None:
        self._config = config

    # ── Storage ─────────────────────────────────────────────

    def create_storage(self) -> StorageProvider:
        provider = self._config.storage.provider
        if provider == "sqlite":
            return self._create_sqlite()
        raise ValueError(f"Unknown storage provider: {provider}")

    def _create_sqlite(self) -> SqliteStore:
        store = SqliteStore(self._config.storage.path)
        store.connect()
        return store

    # ── Tracer ─────────────────────────────────────────────

    def create_tracer(self) -> Tracer:
        """Create the debug tracer — a connected ``SqliteTracer`` when
        ``config.debug.enabled``, otherwise a no-op ``NullTracer``."""
        if not self._config.debug.enabled:
            return NullTracer()
        tracer = SqliteTracer(self._config.debug.trace_db_path)
        tracer.connect()
        return tracer

    # ── LLM ────────────────────────────────────────────────

    def create_llm(self, role: str = "llm") -> LLMProvider:
        cfg = self._config.require_model(role)
        return self._build_llm(cfg)

    def _build_llm(self, cfg: ModelConfig) -> LLMProvider:
        if cfg.provider == "simulated":
            from providers.simulated import SimulatedLLM
            return SimulatedLLM()

        if cfg.provider == "lm_studio":
            from providers.lm_studio import LMStudio
            return LMStudio(
                base_url=cfg.base_url,
                llm_model=cfg.model,
                api_key=cfg.api_key,
                timeout=cfg.timeout,
            )

        if cfg.provider == "ollama":
            from providers.ollama import Ollama
            return Ollama(
                base_url=cfg.base_url,
                llm_model=cfg.model,
                timeout=cfg.timeout,
            )

        raise ValueError(f"Unknown LLM provider: {cfg.provider}")

    # ── Embedder ───────────────────────────────────────────

    def create_embedder(self, role: str = "embedder") -> EmbedderProvider:
        cfg = self._config.require_model(role)
        return self._build_embedder(cfg)

    def _build_embedder(self, cfg: ModelConfig) -> EmbedderProvider:
        if cfg.provider == "simulated":
            from providers.simulated import SimulatedEmbedder
            return SimulatedEmbedder(dimension=cfg.dimension)

        if cfg.provider == "lm_studio":
            from providers.lm_studio import LMStudio
            return LMStudio(
                base_url=cfg.base_url,
                embedder_model=cfg.model,
                api_key=cfg.api_key,
                timeout=cfg.timeout,
            )

        if cfg.provider == "ollama":
            from providers.ollama import Ollama
            return Ollama(
                base_url=cfg.base_url,
                embedder_model=cfg.model,
                timeout=cfg.timeout,
            )

        if cfg.provider == "sentence_transformers":
            from providers.sentence_transformers import SentenceTransformersEmbedder
            return SentenceTransformersEmbedder(model_name=cfg.model)

        raise ValueError(f"Unknown embedder provider: {cfg.provider}")
