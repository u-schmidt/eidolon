"""Builds an EidolonClient (and, for embedded mode, its storage handle)
from an already-loaded EidolonConfig — the construction logic factored
out of EidolonMemoryProvider.initialize().
"""

from __future__ import annotations

from dataclasses import dataclass

from client.http_client import EidolonHttpClient
from core import Eidolon
from core.config import EidolonConfig
from core.eidolon_client import EidolonClient
from core.interfaces import StorageProvider, Tracer
from core.provider_factory import ProviderFactory


@dataclass
class BuiltEidolonClient:
    """The live client plus (embedded mode only) the storage handle the
    adapter must close itself in shutdown() — None in server mode, where
    there's nothing local to close."""

    client: EidolonClient
    store: StorageProvider | None


class EidolonClientBuilder:
    """Builds the EidolonClient, branching on ``config.deployment.mode``.

    Two separate static methods rather than one branching method:
    ``build_embedded`` takes an already-built ``factory``/``tracer``
    rather than building them itself, because the caller
    (EidolonMemoryProvider.initialize()) needs the tracer in hand
    *before* this can fail, to record the failure and keep the tracer
    reachable for shutdown() even though initialize() didn't complete —
    see its call site.
    """

    @staticmethod
    def build_server(config: EidolonConfig) -> EidolonClient:
        """Server mode: core runs as its own process (tools/serve.py) —
        this adapter is just a thin HTTP client, with no storage/llm/
        embedder/tracer to build locally at all. No tracer is wired here
        either: there's nothing on this side to attach one to, and
        building one would just open a trace.db file for no purpose.
        Deliberate v1 limitation — see ROADMAP.md's "Core-as-a-service
        deployment mode" entry: a transport-layer failure (timeout,
        connection refused) is only visible via the broad except/
        logger.warning already in EidolonMemoryProvider's prefetch()/
        sync_turn(), not via tools/trace.py.
        """
        return EidolonHttpClient(
            base_url=config.deployment.server_url,
            token=config.deployment.token,
            timeout=config.deployment.timeout,
        )

    @staticmethod
    def build_embedded(factory: ProviderFactory, tracer: Tracer, config: EidolonConfig) -> BuiltEidolonClient:
        """Embedded mode (default): build storage/llm/embedder and run
        core in-process, wired into a real core.Eidolon instance.

        Metrics after storage: it opens its own connection into tables
        storage's own connect() just migrated into existence (see
        core/stores/migrations.py) — unlike tracer (already built by the
        caller), there's no failure-recording reason to build it first,
        and it couldn't record anything yet if there were.
        """
        store = factory.create_storage()
        llm = factory.create_llm()
        embedder = factory.create_embedder()
        metrics = factory.create_metrics_recorder()

        client = Eidolon(
            storage=store,
            llm=llm,
            embedder=embedder,
            language=config.language,
            retrieval_weights={
                "semantic": config.retrieval.semantic,
                "tag_overlap": config.retrieval.tag_overlap,
                "person": config.retrieval.person,
                "temporal": config.retrieval.temporal,
            },
            retrieval_limits={
                "related_topics": config.retrieval.related_topics,
                "memory_items_fact": config.retrieval.memory_items_fact,
                "memory_items_surprise": config.retrieval.memory_items_surprise,
                "memory_items_feeling": config.retrieval.memory_items_feeling,
                "memory_items_free": config.retrieval.memory_items_free,
                "recent_topics": config.retrieval.recent_topics,
                "cluster_summaries": config.retrieval.cluster_summaries,
                "cluster_recent_topics": config.retrieval.cluster_recent_topics,
                "min_topic_messages": config.retrieval.min_topic_messages,
            },
            facts_consolidation_interval=config.facts.consolidation_interval,
            facts_target_max=config.facts.target_max,
            facts_categories=config.facts.categories,
            entity_profiles_enabled=config.entity_profiles.enabled,
            entity_materialization_threshold=config.entity_profiles.materialization_threshold,
            entity_consolidation_interval=config.entity_profiles.consolidation_interval,
            entity_target_max=config.entity_profiles.target_max_per_entity,
            clustering_enabled=config.clustering.enabled,
            clustering_shortlist_size=config.clustering.shortlist_size,
            clustering_target_max_milestones=config.clustering.target_max_milestones,
            clustering_target_max_tags=config.clustering.target_max_tags,
            relabel_enabled=config.relabel.enabled,
            relabel_message_threshold=config.relabel.message_threshold,
            tracer=tracer,
            metrics=metrics,
        )
        return BuiltEidolonClient(client=client, store=store)
