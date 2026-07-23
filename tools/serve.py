#!/usr/bin/env python3
"""Run Eidolon's core as a standalone HTTP server.

The "server + pgvector" deployment preset: instead of a caller (the
Hermes adapter, or any future adapter) importing core.Eidolon directly
and running it in-process, core runs as its own process here and
callers become thin HTTP clients (see client/http_client.py). See
ROADMAP.md's "Core-as-a-service deployment mode" entry for the full
rationale.

Reuses the exact same ConfigLoader/ProviderFactory path tools/cli.py
uses for LLM/storage/facts config — this process's own eidolon.yaml is
the full config (the same shape embedded mode uses), not a new config
type. Only --host/--port and EIDOLON_DEPLOYMENT_TOKEN (auth) are new,
server-process-specific settings that don't belong on EidolonConfig.

Usage:
    EIDOLON_DEPLOYMENT_TOKEN=... uv run python tools/serve.py --config eidolon.yaml --host 0.0.0.0 --port 8420
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path for direct execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import uvicorn

from core import Eidolon
from core.config import ConfigError
from core.config_loader import ConfigLoader
from core.provider_factory import ProviderFactory
from server.app import create_app


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all available options."""
    parser = argparse.ArgumentParser(description="Run Eidolon's core as a standalone HTTP server.")
    parser.add_argument("--config", default=None, help="Path to YAML config file (LLM/storage/facts/...)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8420, help="Bind port (default: 8420)")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not os.environ.get("EIDOLON_DEPLOYMENT_TOKEN"):
        print(
            "\n  ✗ EIDOLON_DEPLOYMENT_TOKEN must be set — the server refuses to start "
            "without an auth token configured.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load configuration (YAML → env)
    loader = ConfigLoader(config_path=args.config)
    config = loader.load()

    print(f"  Config: {args.config or '(defaults + env)'}")
    if config.storage.provider == "postgres":
        print(f"  Storage: postgres ({config.storage.host}:{config.storage.port}/{config.storage.dbname})")
    else:
        print(f"  Storage: {config.storage.provider} ({config.storage.path})")
    for role, m in config.models.items():
        print(f"  {role.capitalize()}: {m.provider} / {m.model} ({m.base_url or '(no base_url)'})")
    print()

    # Create providers via factory
    factory = ProviderFactory(config)
    try:
        store = factory.create_storage()
        llm = factory.create_llm()
        embedder = factory.create_embedder()
    except ConfigError as exc:
        print(f"\n  ✗ {exc}\n", file=sys.stderr)
        sys.exit(1)
    tracer = factory.create_tracer()

    # Initialize Eidolon core — identical construction to tools/cli.py's,
    # just wrapped in an HTTP server instead of a REPL.
    eidolon = Eidolon(
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
            "related_topics_new": config.retrieval.related_topics_new,
            "related_topics_continue": config.retrieval.related_topics_continue,
            "memory_items_new": config.retrieval.memory_items_new,
            "memory_items_continue": config.retrieval.memory_items_continue,
            "recent_topics": config.retrieval.recent_topics,
            "cluster_summaries": config.retrieval.cluster_summaries,
            "cluster_recent_topics": config.retrieval.cluster_recent_topics,
        },
        facts_consolidation_interval=config.facts.consolidation_interval,
        facts_target_max=config.facts.target_max,
        facts_categories=config.facts.categories,
        clustering_enabled=config.clustering.enabled,
        clustering_shortlist_size=config.clustering.shortlist_size,
        tracer=tracer,
    )

    app = create_app(eidolon)
    print(f"  Starting Eidolon server on {args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
