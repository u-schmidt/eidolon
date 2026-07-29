#!/usr/bin/env python3
"""Eidolon CLI — standalone test tool for the core library.

Run without Hermes to test topic detection, context retrieval, and memory storage.
Uses ConfigLoader + ProviderFactory for flexible provider selection.

Usage:
    # Quick test with simulated providers (no external dependencies):
    python tools/cli.py

    # With a YAML config file:
    python tools/cli.py --config ~/.eidolon/config.yaml

    # With specific LLM and embedder providers:
    python tools/cli.py --llm-provider ollama --llm-model deepseek-v4 \\
                       --embedder-provider lm_studio --embedder-model text-embedding-bge-m3

    # With environment variables:
    EIDOLON_LLM_PROVIDER=ollama EIDOLON_LLM_MODEL=deepseek-v4 python tools/cli.py

    # Cleanup: gracefully close topics left 'active' too long (e.g. a
    # crashed session), then exit without entering the REPL:
    python tools/cli.py --close-stale-topics --stale-older-than-hours 12 --dry-run
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Ensure the project root is on sys.path for direct execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core import Eidolon
from core.config import ConfigError
from core.config_loader import ConfigLoader
from core.context_formatter import ContextFormatter
from core.interfaces import StorageProvider, Tracer
from core.models import EnvironmentSnapshot, MemoryItemBudget
from core.provider_factory import ProviderFactory
from core.stores.sqlite_store import SqliteStore
from core.trace.in_memory_tracer import InMemoryTracer
from tools.trace import TracePrinter


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all available options."""
    parser = argparse.ArgumentParser(
        description="Eidolon CLI — test the core library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python tools/cli.py
  python tools/cli.py --config ~/.eidolon/config.yaml
  python tools/cli.py --llm-provider ollama --llm-base-url http://localhost:11434 --llm-model deepseek-v4
  EIDOLON_LLM_PROVIDER=ollama EIDOLON_LLM_BASE_URL=http://localhost:11434 python tools/cli.py
        """,
    )

    # Config file
    parser.add_argument("--config", default=None, help="Path to YAML config file")

    # Storage
    parser.add_argument("--storage-provider", default=None, choices=["sqlite", "postgres"], help="Storage provider")
    parser.add_argument("--storage-path", default=None, help="Database path (sqlite only)")
    parser.add_argument("--storage-host", default=None, help="Postgres host (postgres only)")
    parser.add_argument("--storage-port", type=int, default=None, help="Postgres port (postgres only)")
    parser.add_argument("--storage-dbname", default=None, help="Postgres database name (postgres only)")
    parser.add_argument("--storage-user", default=None, help="Postgres user (postgres only)")
    parser.add_argument("--storage-password", default=None, help="Postgres password (postgres only)")

    # LLM
    parser.add_argument("--llm-provider", default=None,
                        choices=["simulated", "openai_compatible", "lm_studio", "ollama", "openai", "openrouter"],
                        help="LLM provider name")
    parser.add_argument("--llm-base-url", default=None, help="LLM endpoint URL")
    parser.add_argument("--llm-model", default=None, help="LLM model name")
    parser.add_argument("--llm-max-tokens", type=int, default=None, help="LLM max tokens")
    parser.add_argument("--llm-timeout", type=int, default=None, help="LLM timeout")

    # Embedder
    parser.add_argument("--embedder-provider", default=None,
                        choices=["simulated", "openai_compatible", "lm_studio", "ollama", "openai",
                                 "sentence_transformers"],
                        help="Embedder provider name")
    parser.add_argument("--embedder-base-url", default=None, help="Embedder endpoint URL")
    parser.add_argument("--embedder-model", default=None, help="Embedder model name")
    parser.add_argument("--embedder-timeout", type=int, default=None, help="Embedder timeout")
    parser.add_argument("--embedder-dimension", type=int, default=None, help="Embedder vector dimension")

    # Retrieval weights
    parser.add_argument("--retrieval-semantic", type=float, default=None, help="Semantic weight (0-1)")
    parser.add_argument("--retrieval-tag-overlap", type=float, default=None, help="Tag overlap weight (0-1)")
    parser.add_argument("--retrieval-person", type=float, default=None, help="Person weight (0-1)")
    parser.add_argument("--retrieval-temporal", type=float, default=None, help="Temporal weight (0-1)")

    # Retrieval result-count limits
    parser.add_argument("--related-topics", type=int, default=None, dest="retrieval_related_topics",
                        help="Related topics to retrieve per turn")
    parser.add_argument("--memory-items-fact", type=int, default=None, dest="retrieval_memory_items_fact",
                        help="Reserved memory-item slots for item_type=fact")
    parser.add_argument("--memory-items-surprise", type=int, default=None, dest="retrieval_memory_items_surprise",
                        help="Reserved memory-item slots for item_type=surprise")
    parser.add_argument("--memory-items-feeling", type=int, default=None, dest="retrieval_memory_items_feeling",
                        help="Reserved memory-item slots for item_type=feeling")
    parser.add_argument("--memory-items-free", type=int, default=None, dest="retrieval_memory_items_free",
                        help="Free memory-item slots open to any item_type")
    parser.add_argument("--recent-topics", type=int, default=None, dest="retrieval_recent_topics",
                        help="Most-recently-discussed topics to inject, regardless of relevance")
    parser.add_argument("--cluster-summaries", type=int, default=None, dest="retrieval_cluster_summaries",
                        help="Recurring-topic clusters to inject per turn")
    parser.add_argument("--cluster-recent-topics", type=int, default=None, dest="retrieval_cluster_recent_topics",
                        help="Member topics shown in a cluster's 'newest: ...' tail")
    parser.add_argument("--retrieval-min-topic-messages", type=int, default=None,
                        help="Soft-hide topics at/below this message count from search/recent results")

    # Facts consolidation
    parser.add_argument("--facts-consolidation-interval", type=int, default=None,
                        help="Reconcile the durable facts profile every N user turns")
    parser.add_argument("--facts-target-max", type=int, default=None,
                        help="Soft target size for the durable facts profile")

    # Recurring-topic clustering
    parser.add_argument("--clustering-enabled", dest="clustering_enabled", action="store_true", default=None,
                        help="Enable recurring-topic clustering (on by default)")
    parser.add_argument("--no-clustering", dest="clustering_enabled", action="store_false",
                        help="Disable recurring-topic clustering")
    parser.add_argument("--clustering-shortlist-size", type=int, default=None,
                        help="Candidates shown to the LLM per topic close for cluster matching")

    # General
    parser.add_argument("--language", default=None, help="Language code (default: en)")
    parser.add_argument("--timezone", default=None,
                        help="IANA zone name for displaying injected time, e.g. Europe/Berlin (default: UTC)")
    parser.add_argument("--db", "--db-path", dest="storage_path", default=None, help="Database path (alias for --storage-path)")

    # Debug tracing
    parser.add_argument("--debug", dest="debug_enabled", action="store_true", default=None,
                        help="Enable debug tracing (records per-step trace events to a separate DB)")
    parser.add_argument("--trace-db", dest="debug_trace_db_path", default=None,
                        help="Trace database path (only used when --debug is set)")

    # Maintenance: stale-topic cleanup (runs once and exits, no REPL)
    parser.add_argument("--close-stale-topics", action="store_true",
                        help="Gracefully close 'active' topics idle past --stale-older-than-hours, then exit")
    parser.add_argument("--stale-older-than-hours", type=float, default=24.0,
                        help="Idle threshold in hours for --close-stale-topics (default: 24)")
    parser.add_argument("--stale-workspace-id", default=None,
                        help="Restrict --close-stale-topics to one workspace (default: every workspace)")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --close-stale-topics, list candidates without closing them")

    return parser


def print_banner() -> None:
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║         Eidolon CLI  v0.1.0          ║")
    print("  ║   Topic Memory System — Test Tool    ║")
    print("  ╚══════════════════════════════════════╝")
    print()


def print_help() -> None:
    print("  Commands:")
    print("    <message>     Send a message (topic detection + context retrieval)")
    print("    /topic        Show the current active topic")
    print("    /facts        Show stored facts")
    print("    /consolidate  Manually reconcile the durable facts profile now")
    print("    /topics       List all topics")
    print("    /items        List stored memory items")
    print("    /score <text> Show the full live scored ranking for a query (not just top-K trace summary)")
    print("    /new          Start a new session (reset active topic)")
    print("    /help         Show this help")
    print("    /quit         Exit")
    print()


def _run_stale_topic_cleanup(
    eidolon: Eidolon, store: StorageProvider, tracer: Tracer, args: argparse.Namespace
) -> None:
    """One-shot maintenance mode for --close-stale-topics: list 'active'
    topics idle past the cutoff and, unless --dry-run, close them
    gracefully. Runs once and returns — never enters the REPL."""
    cutoff = datetime.now(UTC) - timedelta(hours=args.stale_older_than_hours)
    candidates = store.get_stale_active_topics(cutoff, workspace_id=args.stale_workspace_id)

    if not candidates:
        print(f"  No stale open topics (idle > {args.stale_older_than_hours}h).")
    else:
        print(f"  {len(candidates)} stale open topic(s) (idle > {args.stale_older_than_hours}h):")
        for t in candidates:
            print(f"    #{t['id']} {t['label']!r} (workspace={t['workspace_id']}, last activity: {t['last_activity']})")

        if args.dry_run:
            print("  Dry run — nothing closed.")
        else:
            closed_ids = eidolon.close_stale_topics(cutoff, workspace_id=args.stale_workspace_id)
            print(f"  Closed {len(closed_ids)} topic(s): {closed_ids}")

    store.close()
    tracer.close()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load configuration (YAML → env → CLI)
    loader = ConfigLoader(config_path=args.config, cli_args=args)
    config = loader.load()

    # Print configuration summary
    print(f"  Config: {args.config or '(defaults + env + CLI)'}")
    if config.storage.provider == "postgres":
        print(f"  Storage: postgres ({config.storage.host}:{config.storage.port}/{config.storage.dbname})")
    else:
        print(f"  Storage: {config.storage.provider} ({config.storage.path})")
    for role, m in config.models.items():
        print(f"  {role.capitalize()}: {m.provider} / {m.model} ({m.base_url or '(no base_url)'})")
    print(f"  Language: {config.language}")
    print(f"  Timezone: {config.timezone}")
    print(
        f"  Clustering: {'ON' if config.clustering.enabled else 'OFF'} "
        f"(shortlist size {config.clustering.shortlist_size})"
    )
    if config.debug.enabled:
        print(f"  Debug tracing: ON ({config.debug.trace_db_path})")

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
    metrics = factory.create_metrics_recorder()

    # Initialize Eidolon core
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
        clustering_enabled=config.clustering.enabled,
        clustering_shortlist_size=config.clustering.shortlist_size,
        relabel_enabled=config.relabel.enabled,
        relabel_message_threshold=config.relabel.message_threshold,
        tracer=tracer,
        metrics=metrics,
    )

    if args.close_stale_topics:
        _run_stale_topic_cleanup(eidolon, store, tracer, args)
        return

    # Session state (managed by the CLI, not by the Core)
    active_topic_id: int | None = None
    turn_count = 0
    chat_history: list[dict[str, str]] = []  # rolling history of last 5 turns

    print_banner()
    print_help()

    while True:
        try:
            text = input("  You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue

        # Handle commands
        if text.startswith("/"):
            cmd = text[1:].lower()

            if cmd in ("quit", "exit", "q"):
                break

            if cmd == "help":
                print_help()
                continue

            if cmd == "new":
                active_topic_id = None
                turn_count = 0
                print("  ═══ New session started ═══")
                continue

            if cmd == "topic":
                if active_topic_id is None:
                    print("  No active topic.")
                else:
                    topic = store.get_topic(active_topic_id)
                    if topic:
                        print(f"  Active topic: {topic['label']} (id={topic['id']}, status={topic['status']})")
                        print(f"  Summary: {topic['summary'][:100] if topic['summary'] else '(none)'}")
                    else:
                        print(f"  Topic {active_topic_id} not found.")
                continue

            if cmd == "facts":
                facts = store.get_facts("user")
                if not facts:
                    print("  No facts stored.")
                else:
                    print(f"  Facts ({len(facts)}):")
                    for f in facts:
                        print(f"    [{f.category}] {f.key} = {f.value} (source: {f.source})")
                continue

            if cmd == "consolidate":
                before = len(store.get_facts("user"))
                eidolon.consolidate_facts("user", request_id=uuid.uuid4().hex[:8])
                after = len(store.get_facts("user"))
                print(f"  Consolidated: {before} → {after} facts.")
                continue

            if cmd == "topics":
                topics = []
                for tid in range(1, 100):
                    t = store.get_topic(tid)
                    if t is None:
                        break
                    topics.append(t)
                if not topics:
                    print("  No topics.")
                else:
                    print(f"  Topics ({len(topics)}):")
                    for t in topics:
                        marker = " ▶" if t["id"] == active_topic_id else "  "
                        print(f"    {marker} #{t['id']} {t['label']} [{t['status']}]")
                continue

            if cmd == "items":
                # Debug-only raw read; the /items dump is sqlite-specific.
                if not isinstance(store, SqliteStore):
                    print("  /items requires the sqlite storage backend.")
                    continue
                conn = store._conn
                rows = conn.execute(
                    "SELECT id, content, item_type, source_peer, subject_peer, "
                    "attribution_type, topic_id, created_at "
                    "FROM memory_items ORDER BY created_at DESC LIMIT 20"
                ).fetchall()
                if not rows:
                    print("  No memory items.")
                else:
                    print(f"  Memory items ({len(rows)}):")
                    for row in rows:
                        print(f"    [#{row[0]}] [{row[2]}] {row[1][:80]}")
                continue

            if cmd.startswith("score "):
                # Recomputes the full scored ranking live against current
                # DB state — the on-demand counterpart to the top-K summary
                # that's actually persisted to the trace db per turn.
                query_text = text[len("/score "):].strip()
                if not query_text:
                    print("  Usage: /score <query text>")
                    continue
                query_embedding = embedder.embed(query_text)
                weights = {
                    "semantic": config.retrieval.semantic,
                    "tag_overlap": config.retrieval.tag_overlap,
                    "person": config.retrieval.person,
                    "temporal": config.retrieval.temporal,
                }
                spy = InMemoryTracer()
                request_id = uuid.uuid4().hex[:8]
                store.search_topics(
                    query_embedding=query_embedding, workspace_id="user", limit=100000,
                    weights=weights, tracer=spy, request_id=request_id,
                )
                store.search_memory_items(
                    query_embedding=query_embedding, workspace_id="user",
                    budget=MemoryItemBudget(fact=100_000, surprise=100_000, feeling=100_000, free=100_000),
                    weights=weights, tracer=spy, request_id=request_id,
                )
                printer = TracePrinter()
                events = spy.events_of_type("retrieval_scoring_summary")
                if not events:
                    print("  No scored candidates.")
                for event in events:
                    print(printer.format_detail(event))
                continue

            print(f"  Unknown command: {cmd}")
            continue

        # Process a message
        turn_count += 1
        # Captured once and reused for both the injected "current time" and
        # ContextFormatter's relative-age calculations below, so they can't
        # drift apart within a single turn. Explicitly UTC so it's correct
        # regardless of the host's OS timezone — ContextFormatter converts
        # it to config.timezone only for display.
        turn_now = datetime.now(UTC)
        env = EnvironmentSnapshot(
            time=turn_now,
            day_of_week=turn_now.strftime("%A"),
            situation="cli_test",
        )

        print(f"  ─── Turn {turn_count} ───")

        # One id for the whole turn, so tools/trace.py shows detect_topic,
        # retrieve_context, and store_turn as a single grouped request
        # instead of three unrelated ones.
        turn_request_id = uuid.uuid4().hex[:8]

        # Step 1: Detect topic
        topic_change = eidolon.detect_topic(text, active_topic_id, request_id=turn_request_id)
        print(f"  Topic: {topic_change.type}", end="")
        if topic_change.label:
            print(f" → '{topic_change.label}'", end="")
        print()

        # Step 2: Retrieve context
        is_new = topic_change.type == "NEW_TOPIC"
        bundle = eidolon.retrieve_context(
            message=text,
            workspace_id="user",
            environment=env,
            topic_label=topic_change.label,
            is_new_topic=is_new,
            active_topic_id=active_topic_id,
            request_id=turn_request_id,
        )

        if bundle.facts:
            print(f"  Facts: {len(bundle.facts)}")
        if bundle.recent_topics:
            print(f"  Recent topics: {len(bundle.recent_topics)}")
            for rt in bundle.recent_topics:
                print(f"    - #{rt.topic_id} {rt.summary[:60] if rt.summary else '(no summary)'}")
        if bundle.related_topics:
            print(f"  Related topics: {len(bundle.related_topics)}")
            for rt in bundle.related_topics:
                print(f"    - #{rt.topic_id} {rt.summary[:60] if rt.summary else '(no summary)'}")
        if bundle.memory_items:
            print(f"  Memory items: {len(bundle.memory_items)}")

        # Step 3: Generate AI response using the LLM provider
        context_prompt = ContextFormatter.format(bundle, now=turn_now, timezone=config.timezone)

        # Build chat history into the prompt (last 5 turns)
        history_text = ""
        if chat_history:
            history_text = "Recent conversation:\n"
            for turn in chat_history:
                history_text += f"User: {turn['user']}\nAI: {turn['ai']}\n"
            history_text += "\n"

        system_prompt = (
            "You are Eidolon CLI, a test interface for the Eidolon topic memory system. "
            "Respond naturally to the user's message. Use the provided context if relevant, "
            "but don't mention the memory system unless asked. Keep responses concise."
        )

        full_prompt = f"{context_prompt}\n{history_text}User: {text}\nAI:"
        # No max_tokens override — use the provider's configured cap
        # (ModelConfig.max_tokens). Reasoning models bill their internal
        # reasoning against the same completion budget as the visible
        # reply, so the cap must be generous; that's now a single
        # configurable knob rather than a per-call-site guess.
        ai_response = llm.complete(
            prompt=full_prompt,
            system_prompt=system_prompt,
        )

        # Update rolling chat history (keep last 5 turns)
        chat_history.append({"user": text, "ai": ai_response})
        if len(chat_history) > 5:
            chat_history.pop(0)

        # Step 4: Store the turn — store_turn returns the new topic ID
        # Always pass active_topic_id so messages are linked to the current topic.
        # store_turn only closes the topic when new_topic_label is also set.
        old_topic_id = active_topic_id
        new_topic_label = topic_change.label if is_new else None

        new_topic_id = eidolon.store_turn(
            user_message=text,
            ai_response=ai_response,
            workspace_id="user",
            environment=env,
            turn_id=turn_count,
            old_topic_id=old_topic_id,
            new_topic_label=new_topic_label,
            request_id=turn_request_id,
        )

        # Update session state with the returned topic ID
        if new_topic_id is not None:
            active_topic_id = new_topic_id

        print(f"  AI  > {ai_response}")
        print()

    store.close()
    tracer.close()
    print("  Bye!")


if __name__ == "__main__":
    main()
