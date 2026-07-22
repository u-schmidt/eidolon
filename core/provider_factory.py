"""Provider factory for Eidolon.

Creates provider instances (LLM, Embedder, Storage) from configuration.
Each model role (``llm``, ``embedder``, and future roles like ``judge``)
is a fully self-contained ``ModelConfig`` — no shared endpoint registry,
no cross-referencing by name. A role that isn't fully configured raises
``ConfigError`` (via ``EidolonConfig.require_model``) instead of silently
falling back to a guessed endpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.config import EidolonConfig, ModelConfig
from core.interfaces import EmbedderProvider, LLMProvider, StorageProvider, Tracer
from core.stores.sqlite_store import SqliteStore
from core.trace.null_tracer import NullTracer
from core.trace.safe_tracer import SafeTracer
from core.trace.sqlite_tracer import SqliteTracer

if TYPE_CHECKING:
    from core.providers.openai_compatible import OpenAICompatible


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
        """Create the debug tracer — a connected ``SqliteTracer`` wrapped
        in ``SafeTracer`` when ``config.debug.enabled``, otherwise a
        no-op ``NullTracer``. ``SafeTracer`` ensures a broken trace
        backend degrades to "no tracing" instead of taking down real
        extraction/storage."""
        if not self._config.debug.enabled:
            return NullTracer()
        tracer = SafeTracer(SqliteTracer(self._config.debug.trace_db_path))
        tracer.connect()
        return tracer

    # ── LLM ────────────────────────────────────────────────

    def create_llm(self, role: str = "llm") -> LLMProvider:
        cfg = self._config.require_model(role)
        return self._build_llm(cfg)

    def _build_llm(self, cfg: ModelConfig) -> LLMProvider:
        if cfg.provider == "simulated":
            from core.providers.simulated import SimulatedLLM
            return SimulatedLLM()

        cls = self._openai_compatible_class(cfg.provider)
        if cls is not None:
            return cls(
                base_url=cfg.base_url,
                llm_model=cfg.model,
                api_key=cfg.api_key,
                timeout=cfg.timeout,
                max_tokens=cfg.max_tokens,
            )

        raise ValueError(f"Unknown LLM provider: {cfg.provider}")

    # ── Embedder ───────────────────────────────────────────

    def create_embedder(self, role: str = "embedder") -> EmbedderProvider:
        cfg = self._config.require_model(role)
        return self._build_embedder(cfg)

    def _build_embedder(self, cfg: ModelConfig) -> EmbedderProvider:
        if cfg.provider == "simulated":
            from core.providers.simulated import SimulatedEmbedder
            return SimulatedEmbedder(dimension=cfg.dimension)

        if cfg.provider == "sentence_transformers":
            from core.providers.sentence_transformers import SentenceTransformersEmbedder
            return SentenceTransformersEmbedder(model_name=cfg.model)

        cls = self._openai_compatible_class(cfg.provider)
        if cls is not None:
            return cls(
                base_url=cfg.base_url,
                embedder_model=cfg.model,
                api_key=cfg.api_key,
                timeout=cfg.timeout,
            )

        raise ValueError(f"Unknown embedder provider: {cfg.provider}")

    @staticmethod
    def _openai_compatible_class(provider: str) -> type[OpenAICompatible] | None:
        """Map an OpenAI-compatible provider name to its class, or None.

        ``ollama``/``lm_studio``/``openai``/``openrouter`` are thin presets
        of ``OpenAICompatible`` (different default base_url/model/timeout);
        ``openai_compatible`` is the generic name for any other hosted or
        self-hosted OpenAI-API endpoint (Groq, Together, vLLM, a private
        gateway). All five take the same constructor arguments, so one
        mapped branch builds any of them — and every one now receives
        ``api_key``."""
        if provider == "openai_compatible":
            from core.providers.openai_compatible import OpenAICompatible
            return OpenAICompatible
        if provider == "ollama":
            from core.providers.ollama import Ollama
            return Ollama
        if provider == "lm_studio":
            from core.providers.lm_studio import LMStudio
            return LMStudio
        if provider == "openai":
            from core.providers.openai import OpenAI
            return OpenAI
        if provider == "openrouter":
            from core.providers.openrouter import OpenRouter
            return OpenRouter
        return None
