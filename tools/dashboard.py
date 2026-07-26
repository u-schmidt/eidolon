#!/usr/bin/env python3
"""Run Eidolon's turn-detail dashboard — a read-only web view over one
turn (request_id) at a time: which topic was active, the user/AI
message, and a step-by-step account of what fired during it.

Independent of ``deployment.mode`` — always connects directly to
storage (embedded or server deployment alike), since it needs raw
historical/analytical queries no ``EidolonClient`` RPC method offers.
Works with or without ``debug.enabled``: the always-on operational
metrics give a "skeleton" view (steps, timing, success) regardless;
rich per-step payloads (prompts, responses, score breakdowns) additionally
show up for any turn that ran while ``debug.enabled: true``.

Usage:
    uv run python tools/dashboard.py --config eidolon.yaml --host 127.0.0.1 --port 8421
    EIDOLON_DASHBOARD_PASSWORD=... uv run python tools/dashboard.py  # optional HTTP Basic auth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path for direct execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import uvicorn

from core.config import ConfigError
from core.config_loader import ConfigLoader
from core.provider_factory import ProviderFactory
from tools.dashboard.app import create_dashboard_app
from tools.dashboard.turn_reader import TurnReader


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all available options."""
    parser = argparse.ArgumentParser(description="Run Eidolon's turn-detail dashboard.")
    parser.add_argument("--config", default=None, help="Path to YAML config file (storage/debug/...)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8421, help="Bind port (default: 8421)")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    loader = ConfigLoader(config_path=args.config)
    config = loader.load()

    print(f"  Config: {args.config or '(defaults + env)'}")
    if config.storage.provider == "postgres":
        print(f"  Storage: postgres ({config.storage.host}:{config.storage.port}/{config.storage.dbname})")
    else:
        print(f"  Storage: {config.storage.provider} ({config.storage.path})")
    print()

    factory = ProviderFactory(config)
    try:
        store = factory.create_storage()
    except ConfigError as exc:
        print(f"\n  ✗ {exc}\n", file=sys.stderr)
        sys.exit(1)
    metrics_reader = factory.create_metrics_reader()
    trace_reader = factory.create_trace_reader()

    turn_reader = TurnReader(store, metrics_reader, trace_reader)
    app = create_dashboard_app(turn_reader)
    print(f"  Starting Eidolon dashboard on {args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
