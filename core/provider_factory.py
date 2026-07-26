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

from core.config import ConfigError, EidolonConfig, ModelConfig, StorageConfig
from core.interfaces import (
    EmbedderProvider,
    LLMProvider,
    MetricsReader,
    MetricsRecorder,
    StorageProvider,
    Tracer,
    TraceReader,
)
from core.metrics.safe_metrics_recorder import SafeMetricsRecorder
from core.metrics.sqlite_metrics_reader import SqliteMetricsReader
from core.metrics.sqlite_metrics_recorder import SqliteMetricsRecorder
from core.stores.sqlite_store import SqliteStore
from core.trace.null_tracer import NullTracer
from core.trace.safe_tracer import SafeTracer
from core.trace.sqlite_trace_reader import SqliteTraceReader
from core.trace.sqlite_tracer import SqliteTracer

if TYPE_CHECKING:
    from core.providers.openai_compatible import OpenAICompatible
    from core.stores.postgres_store import PostgresStore


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
        if provider == "postgres":
            return self._create_postgres()
        raise ValueError(f"Unknown storage provider: {provider}")

    def _create_sqlite(self) -> SqliteStore:
        store = SqliteStore(self._config.storage.path)
        store.connect()
        return store

    @staticmethod
    def _require_postgres_password(cfg: StorageConfig) -> None:
        if not cfg.password:
            raise ConfigError(
                "storage.password is required for provider='postgres' — set it in "
                "eidolon.yaml, EIDOLON_STORAGE_PASSWORD, or --storage-password."
            )

    def _create_postgres(self) -> PostgresStore:
        from core.stores.postgres_store import PostgresStore

        cfg = self._config.storage
        self._require_postgres_password(cfg)
        # PostgresStore needs the embedder's dimension up front to size its
        # VECTOR(n) columns at schema-creation time — pgvector's vector
        # width is fixed per column, unlike SQLite's dimension-agnostic
        # BLOB storage. A narrow, unavoidable coupling to the embedder
        # config, not a general pattern other providers need.
        embedder_cfg = self._config.require_model("embedder")
        store = PostgresStore(
            host=cfg.host, port=cfg.port, dbname=cfg.dbname,
            user=cfg.user, password=cfg.password,
            embedding_dim=embedder_cfg.dimension,
        )
        store.connect()
        return store

    # ── Tracer ─────────────────────────────────────────────

    def create_tracer(self) -> Tracer:
        """Create the debug tracer — a connected backend wrapped in
        ``SafeTracer`` when ``config.debug.enabled``, otherwise a no-op
        ``NullTracer``. ``SafeTracer`` ensures a broken trace backend
        degrades to "no tracing" instead of taking down real
        extraction/storage.

        Backend follows ``storage.provider``, mirroring ``create_storage()``
        — a Postgres-backed tracer writes into the same instance as
        production data instead of a separate SQLite file, avoiding the
        mounted-volume reliability problem ``PostgresStore`` itself exists
        to solve (see its docstring). Still fully gated by
        ``debug.enabled`` either way — this is the rich, verbose trace,
        not the always-on operational metrics (see ``create_metrics_recorder``).
        """
        if not self._config.debug.enabled:
            return NullTracer()
        if self._config.storage.provider == "postgres":
            from core.trace.postgres_tracer import PostgresTracer

            cfg = self._config.storage
            self._require_postgres_password(cfg)
            tracer = SafeTracer(PostgresTracer(
                host=cfg.host, port=cfg.port, dbname=cfg.dbname,
                user=cfg.user, password=cfg.password,
            ))
        else:
            tracer = SafeTracer(SqliteTracer(self._config.debug.trace_db_path))
        tracer.connect()
        return tracer

    # ── Metrics ────────────────────────────────────────────

    def create_metrics_recorder(self) -> MetricsRecorder:
        """Create the always-on operational-metrics recorder.

        Unlike ``create_tracer()``, this has **no** ``debug.enabled``
        gate — every deployment always gets error/duration visibility,
        regardless of whether verbose debug tracing is turned on. Backend
        follows ``storage.provider``, same branch as ``create_tracer()``,
        writing into the same database instance as production storage —
        both tables are created unconditionally by that database's own
        migrations (see ``core/stores/migrations.py``/
        ``postgres_migrations.py``), so this recorder runs no migration
        of its own, it just connects. Wrapped in ``SafeMetricsRecorder``
        — more important here than for ``SafeTracer``, since this path
        always runs.
        """
        if self._config.storage.provider == "postgres":
            from core.metrics.postgres_metrics_recorder import PostgresMetricsRecorder

            cfg = self._config.storage
            self._require_postgres_password(cfg)
            recorder = SafeMetricsRecorder(PostgresMetricsRecorder(
                host=cfg.host, port=cfg.port, dbname=cfg.dbname,
                user=cfg.user, password=cfg.password,
            ))
        else:
            recorder = SafeMetricsRecorder(SqliteMetricsRecorder(self._config.storage.path))
        recorder.connect()
        return recorder

    # ── Readers (turn-detail dashboard, tools/dashboard/) ───

    def create_metrics_reader(self) -> MetricsReader:
        """Create a read-only handle onto the operational-metrics
        tables. Unconditional, like ``create_metrics_recorder()`` — no
        ``debug.enabled`` gate, since these tables are always populated.
        Not wrapped in a ``Safe*`` decorator: a failed read should
        surface as a visible error on the dashboard page, not be
        silently swallowed the way a failed *write* from a hot
        extraction/detection path should be.
        """
        if self._config.storage.provider == "postgres":
            from core.metrics.postgres_metrics_reader import PostgresMetricsReader

            cfg = self._config.storage
            self._require_postgres_password(cfg)
            reader: MetricsReader = PostgresMetricsReader(
                host=cfg.host, port=cfg.port, dbname=cfg.dbname,
                user=cfg.user, password=cfg.password,
            )
        else:
            reader = SqliteMetricsReader(self._config.storage.path)
        reader.connect()
        return reader

    def create_trace_reader(self) -> TraceReader:
        """Create a read-only handle onto the debug trace backend.
        Unconditional, unlike ``create_tracer()`` — historical trace
        rows may exist from a period when ``debug.enabled`` was on even
        if it's off now, and reading is always safe. Backend follows
        ``storage.provider``, same branch as ``create_tracer()``.
        """
        if self._config.storage.provider == "postgres":
            from core.trace.postgres_trace_reader import PostgresTraceReader

            cfg = self._config.storage
            self._require_postgres_password(cfg)
            trace_reader: TraceReader = PostgresTraceReader(
                host=cfg.host, port=cfg.port, dbname=cfg.dbname,
                user=cfg.user, password=cfg.password,
            )
        else:
            trace_reader = SqliteTraceReader(self._config.debug.trace_db_path)
        trace_reader.connect()
        return trace_reader

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
