"""CLI entry point for running an Eidolon eval script.

Uses ConfigLoader + ProviderFactory for the LLM/embedder, exactly like
tools/cli.py — but storage is always a fresh, throwaway SQLite DB, never
the configured (dev or live) store, so eval runs can never write into
real peer data. The DB path is printed at the end for drill-down with
tools/cli.py or tools/trace_tui.py.

Usage:
    uv run python tools/run_eval.py tools/eval/scripts/<script>.yaml
    uv run python tools/run_eval.py <script>.yaml --config my-eval-config.yaml

See tools/eval/eval_config.example.yaml for a config template — copy it
and point it at your real LLM/embedder. It bakes in two eval-specific
settings worth keeping: a low facts.consolidation_interval (so short
scripts actually trigger consolidation) and debug.enabled: true (so
every run leaves a trace DB for drill-down).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Ensure the project root is on sys.path for direct execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core import Eidolon
from core.config import ConfigError
from core.config_loader import ConfigLoader
from core.provider_factory import ProviderFactory
from core.stores.sqlite_store import SqliteStore
from tools.eval.report_formatter import EvalReportFormatter
from tools.eval.runner import EvalRunner
from tools.eval.scorer import EvalScorer
from tools.eval.script_loader import EvalScriptLoader


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an Eidolon eval script against real configured providers.")
    parser.add_argument("script", help="Path to the eval script YAML file")
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML config file for the LLM/embedder (storage is always a fresh throwaway DB)",
    )
    parser.add_argument(
        "--reports-dir", default="tools/eval/reports", help="Directory to write the report files to",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    loader = ConfigLoader(config_path=args.config)
    config = loader.load()

    print(f"  Config: {args.config or '(defaults + env + CLI)'}")
    for role, m in config.models.items():
        print(f"  {role.capitalize()}: {m.provider} / {m.model} ({m.base_url or '(no base_url)'})")
    print(f"  Facts consolidation interval: every {config.facts.consolidation_interval} user turns")
    if config.debug.enabled:
        print(f"  Debug tracing: ON ({config.debug.trace_db_path})")
    print()

    factory = ProviderFactory(config)
    try:
        llm = factory.create_llm()
        embedder = factory.create_embedder()
    except ConfigError as exc:
        print(f"\n  ✗ {exc}\n", file=sys.stderr)
        sys.exit(1)
    tracer = factory.create_tracer()

    tmp_dir = tempfile.mkdtemp(prefix="eidolon-eval-")
    db_path = str(Path(tmp_dir) / "eval.db")
    storage = SqliteStore(db_path)
    storage.connect()

    eidolon = Eidolon(
        storage=storage,
        llm=llm,
        embedder=embedder,
        language=config.language,
        facts_consolidation_interval=config.facts.consolidation_interval,
        facts_target_max=config.facts.target_max,
        tracer=tracer,
    )

    script = EvalScriptLoader().load(args.script)
    runner = EvalRunner(eidolon, storage, EvalScorer(embedder))

    llm_cfg = config.models["llm"]
    embedder_cfg = config.models["embedder"]
    report = runner.run(
        script,
        model_identity=f"{llm_cfg.provider}/{llm_cfg.model}",
        embedder_identity=f"{embedder_cfg.provider}/{embedder_cfg.model}",
        db_path=db_path,
    )

    formatter = EvalReportFormatter()
    print(formatter.to_markdown(report))
    written = formatter.write(report, args.reports_dir)
    print(f"\nReport written to {written}")
    print(f"Eval DB (for drill-down with tools/cli.py or tools/trace_tui.py): {db_path}")
    if config.debug.enabled:
        print(f"Trace DB (for drill-down with tools/trace.py or tools/trace_tui.py): {config.debug.trace_db_path}")

    storage.close()


if __name__ == "__main__":
    main()
