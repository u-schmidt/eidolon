"""CLI entry point for running an Eidolon eval script.

Uses ConfigLoader + ProviderFactory for the LLM/embedder, exactly like
tools/cli.py — but storage is always a fresh, isolated store, never the
configured (dev or live) one, so eval runs can never write into real
workspace data. Default is a throwaway temp SQLite file; --storage-provider
postgres points it at a Postgres/pgvector target instead (its tables are
truncated before the run for the same fresh-isolation guarantee) — this
is what makes a SQLite-vs-Postgres retrieval-quality parity comparison
possible: run the same script against both, `--tag sqlite-baseline` /
`--tag postgres-candidate`, then compare via tools/analyze_reports.py.
Per ROADMAP.md's own documented caveat, a single run each way isn't a
statistically reliable comparison given cloud-model run-to-run variance
— repeat several times per backend before trusting a parity claim.
The storage descriptor (file path or connection string) is printed at
the end for drill-down with tools/cli.py or tools/trace_tui.py.

Usage:
    uv run python tools/run_eval.py tools/eval/scripts/<script>.yaml
    uv run python tools/run_eval.py <script>.yaml --config my-eval-config.yaml
    uv run python tools/run_eval.py <script>.yaml --tag sqlite-baseline
    uv run python tools/run_eval.py <script>.yaml --tag postgres-candidate \\
        --storage-provider postgres --storage-host 127.0.0.1 --storage-dbname eidolon_eval \\
        --storage-user eidolon --storage-password secret

See tools/eval/eval_config.example.yaml for a config template — copy it
and point it at your real LLM/embedder. It bakes in two eval-specific
settings worth keeping: a low facts.consolidation_interval (so short
scripts actually trigger consolidation) and debug.enabled: true (so
every run leaves a trace DB for drill-down).
"""

from __future__ import annotations

import argparse
import os
import subprocess
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
from core.interfaces import StorageProvider
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
        help="Path to YAML config file for the LLM/embedder (storage is always a fresh, isolated store)",
    )
    parser.add_argument(
        "--reports-dir", default="tools/eval/reports", help="Directory to write the report files to",
    )
    parser.add_argument(
        "--tag", default=None,
        help="Free-form label for this run (e.g. 'prompt-17', 'sqlite-baseline'), recorded in the "
             "report for later comparison",
    )
    parser.add_argument(
        "--storage-provider", default="sqlite", choices=["sqlite", "postgres"],
        help="Eval storage backend (default: sqlite, a throwaway temp file)",
    )
    parser.add_argument("--storage-host", default=None, help="Postgres host (postgres only)")
    parser.add_argument("--storage-port", type=int, default=5432, help="Postgres port (postgres only)")
    parser.add_argument("--storage-dbname", default=None, help="Postgres database name (postgres only)")
    parser.add_argument("--storage-user", default=None, help="Postgres user (postgres only)")
    parser.add_argument(
        "--storage-password", default=None,
        help="Postgres password (postgres only) — or set EIDOLON_STORAGE_PASSWORD",
    )
    return parser


def _build_storage(args: argparse.Namespace, embedding_dim: int) -> tuple[StorageProvider, str]:
    """Build the eval run's storage backend plus a display string for the
    report's db_path field. Always fresh and isolated regardless of
    backend: SQLite gets a new throwaway temp file (as before); Postgres
    has its tables truncated before the run — an eval run must never
    accumulate state across runs or touch real workspace data either way.
    """
    if args.storage_provider == "postgres":
        from core.stores.postgres_store import PostgresStore

        password = args.storage_password or os.environ.get("EIDOLON_STORAGE_PASSWORD", "")
        if not (args.storage_host and args.storage_dbname and args.storage_user and password):
            print(
                "\n  ✗ --storage-provider postgres requires --storage-host, --storage-dbname, "
                "--storage-user, and --storage-password (or EIDOLON_STORAGE_PASSWORD)\n",
                file=sys.stderr,
            )
            sys.exit(1)

        pg_storage = PostgresStore(
            host=args.storage_host, port=args.storage_port, dbname=args.storage_dbname,
            user=args.storage_user, password=password, embedding_dim=embedding_dim,
        )
        pg_storage.connect()
        with pg_storage._conn.cursor() as cur:
            cur.execute(
                "TRUNCATE topics, memory_items, messages, facts, "
                "topic_clusters, fact_consolidation_state RESTART IDENTITY CASCADE"
            )
        pg_storage._conn.commit()
        descriptor = f"postgres://{args.storage_host}:{args.storage_port}/{args.storage_dbname}"
        print(f"  Storage: postgres ({descriptor}) — truncated for a fresh run")
        return pg_storage, descriptor

    tmp_dir = tempfile.mkdtemp(prefix="eidolon-eval-")
    db_path = str(Path(tmp_dir) / "eval.db")
    storage = SqliteStore(db_path)
    storage.connect()
    return storage, db_path


def _current_git_commit() -> str:
    """Short commit hash of HEAD, suffixed '-dirty' if the working tree has
    uncommitted changes — the closest thing this project has to a version
    number, recorded so eval reports can be compared against what actually
    ran, not what was last committed."""
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    )
    return f"{commit}-dirty" if dirty else commit


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

    llm_cfg = config.models["llm"]
    embedder_cfg = config.models["embedder"]
    storage, db_path = _build_storage(args, embedder_cfg.dimension)

    eidolon = Eidolon(
        storage=storage,
        llm=llm,
        embedder=embedder,
        language=config.language,
        facts_consolidation_interval=config.facts.consolidation_interval,
        facts_target_max=config.facts.target_max,
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
    )

    script = EvalScriptLoader().load(args.script)
    runner = EvalRunner(eidolon, storage, EvalScorer(embedder))

    report = runner.run(
        script,
        model_identity=f"{llm_cfg.provider}/{llm_cfg.model}",
        embedder_identity=f"{embedder_cfg.provider}/{embedder_cfg.model}",
        db_path=db_path,
        git_commit=_current_git_commit(),
        tag=args.tag,
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
