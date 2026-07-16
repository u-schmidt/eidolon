"""Provider factory for Eidolon.

Creates provider instances (LLM, Embedder, Storage) from configuration.
``config.providers`` is a map of named endpoints (e.g. ollama, lm_studio).
``config.llm.provider`` and ``config.embedder.provider`` reference one by name.
"""

from __future__ import annotations

from core.config import EidolonConfig, ProviderEndpoint
from core.interfaces import EmbedderProvider, LLMProvider, StorageProvider
from core.stores.sqlite_store import SqliteStore


class ProviderFactory:
    """Creates provider instances from an ``EidolonConfig``.

    Usage::

        config = ConfigLoader("config.yaml").load()
        factory = ProviderFactory(config)
        store = factory.create_storage()
        llm = factory.create_llm()
        embedder = factory.create_embedder()
    """

    def __init__(self, config: EidolonConfig) -> None:
        self._config = config

    def _get_endpoint(self, name: str) -> ProviderEndpoint:
        """Get a provider endpoint by name, creating a default if missing."""
        if name in self._config.providers:
            return self._config.providers[name]
        # Auto-create a default endpoint for convenience (e.g. --llm-provider lm_studio)
        ep = ProviderEndpoint(name=name)
        self._config.providers[name] = ep
        return ep

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

    # ── LLM ────────────────────────────────────────────────

    def create_llm(self) -> LLMProvider:
        provider_name = self._config.llm.provider
        ep = self._get_endpoint(provider_name)

        if provider_name == "simulated":
            from tests.mocks import SimulatedLLM
            return SimulatedLLM()

        if provider_name == "lm_studio":
            from providers.lm_studio import LMStudio
            return LMStudio(
                base_url=ep.base_url,
                llm_model=self._config.llm.model,
                api_key=ep.api_key,
                timeout=self._config.llm.timeout or ep.timeout,
            )

        if provider_name == "ollama":
            from providers.ollama import Ollama
            return Ollama(
                base_url=ep.base_url,
                llm_model=self._config.llm.model,
                timeout=self._config.llm.timeout or ep.timeout,
            )

        raise ValueError(f"Unknown LLM provider: {provider_name}")

    # ── Embedder ───────────────────────────────────────────

    def create_embedder(self) -> EmbedderProvider:
        provider_name = self._config.embedder.provider
        ep = self._get_endpoint(provider_name)

        if provider_name == "simulated":
            from tests.mocks import SimulatedEmbedder
            return SimulatedEmbedder(dimension=self._config.embedder.dimension)

        if provider_name == "lm_studio":
            from providers.lm_studio import LMStudio
            return LMStudio(
                base_url=ep.base_url,
                embedder_model=self._config.embedder.model,
                api_key=ep.api_key,
                timeout=self._config.embedder.timeout or ep.timeout,
            )

        if provider_name == "ollama":
            from providers.ollama import Ollama
            return Ollama(
                base_url=ep.base_url,
                embedder_model=self._config.embedder.model,
                timeout=self._config.embedder.timeout or ep.timeout,
            )

        if provider_name == "sentence_transformers":
            from providers.sentence_transformers import SentenceTransformersEmbedder
            return SentenceTransformersEmbedder(model_name=self._config.embedder.model)

        raise ValueError(f"Unknown embedder provider: {provider_name}")
