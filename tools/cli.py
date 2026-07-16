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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path for direct execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from core import Eidolon
from core.config_loader import ConfigLoader
from core.models import EnvironmentSnapshot
from core.provider_factory import ProviderFactory


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all available options."""
    parser = argparse.ArgumentParser(
        description="Eidolon CLI — test the core library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python tools/cli.py
  python tools/cli.py --config ~/.eidolon/config.yaml
  python tools/cli.py --llm-provider ollama --llm-model deepseek-v4 --embedder-provider lm_studio
  EIDOLON_LLM_PROVIDER=ollama python tools/cli.py
        """,
    )

    # Config file
    parser.add_argument("--config", default=None, help="Path to YAML config file")

    # Storage
    parser.add_argument("--storage-provider", default=None, choices=["sqlite"], help="Storage provider")
    parser.add_argument("--storage-path", default=None, help="Database path")

    # LLM
    parser.add_argument("--llm-provider", default=None,
                        choices=["simulated", "lm_studio", "ollama"],
                        help="LLM provider name (must match a key in providers config)")
    parser.add_argument("--llm-model", default=None, help="LLM model name")
    parser.add_argument("--llm-max-tokens", type=int, default=None, help="LLM max tokens")
    parser.add_argument("--llm-timeout", type=int, default=None, help="LLM timeout")

    # Embedder
    parser.add_argument("--embedder-provider", default=None,
                        choices=["simulated", "lm_studio", "ollama", "sentence_transformers"],
                        help="Embedder provider name (must match a key in providers config)")
    parser.add_argument("--embedder-model", default=None, help="Embedder model name")
    parser.add_argument("--embedder-timeout", type=int, default=None, help="Embedder timeout")
    parser.add_argument("--embedder-dimension", type=int, default=None, help="Embedder vector dimension")

    # Retrieval weights
    parser.add_argument("--retrieval-semantic", type=float, default=None, help="Semantic weight (0-1)")
    parser.add_argument("--retrieval-tag-overlap", type=float, default=None, help="Tag overlap weight (0-1)")
    parser.add_argument("--retrieval-person", type=float, default=None, help="Person weight (0-1)")
    parser.add_argument("--retrieval-temporal", type=float, default=None, help="Temporal weight (0-1)")

    # General
    parser.add_argument("--language", default=None, help="Language code (default: en)")
    parser.add_argument("--db", "--db-path", dest="db_path", default=None, help="Database path (overrides storage.path)")

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
    print("    /topics       List all topics")
    print("    /items        List stored memory items")
    print("    /new          Start a new session (reset active topic)")
    print("    /help         Show this help")
    print("    /quit         Exit")
    print()


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Load configuration (YAML → env → CLI)
    loader = ConfigLoader(config_path=args.config, cli_args=args)
    config = loader.load()

    # Print configuration summary
    print(f"  Config: {args.config or '(defaults + env + CLI)'}")
    print(f"  Storage: {config.storage.provider} ({config.storage.path})")
    print(f"  LLM: {config.llm.provider} / {config.llm.model}")
    print(f"  Embedder: {config.embedder.provider} / {config.embedder.model}")
    print(f"  Language: {config.language}")
    for name, ep in config.providers.items():
        print(f"  Provider '{name}': {ep.base_url}")

    # Create providers via factory
    factory = ProviderFactory(config)
    store = factory.create_storage()
    llm = factory.create_llm()
    embedder = factory.create_embedder()

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
    )

    # Session state (managed by the CLI, not by the Core)
    active_topic_id: int | None = None
    turn_count = 0

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

            print(f"  Unknown command: {cmd}")
            continue

        # Process a message
        turn_count += 1
        env = EnvironmentSnapshot(
            time=None,
            day_of_week=None,
            situation="cli_test",
        )

        print(f"  ─── Turn {turn_count} ───")

        # Step 1: Detect topic
        topic_change = eidolon.detect_topic(text, active_topic_id)
        print(f"  Topic: {topic_change.type}", end="")
        if topic_change.label:
            print(f" → '{topic_change.label}'", end="")
        print()

        # Step 2: Retrieve context
        is_new = topic_change.type == "NEW_TOPIC"
        bundle = eidolon.retrieve_context(
            message=text,
            peer_id="user",
            environment=env,
            topic_label=topic_change.label,
            is_new_topic=is_new,
        )

        if bundle.facts:
            print(f"  Facts: {len(bundle.facts)}")
        if bundle.related_topics:
            print(f"  Related topics: {len(bundle.related_topics)}")
            for rt in bundle.related_topics:
                print(f"    - #{rt.topic_id} {rt.summary[:60] if rt.summary else '(no summary)'}")
        if bundle.memory_items:
            print(f"  Memory items: {len(bundle.memory_items)}")

        # Step 3: Generate AI response using the LLM provider
        context_prompt = ""
        if bundle.current_topic_label:
            context_prompt += f"Current topic: {bundle.current_topic_label}\n"
        if bundle.facts:
            context_prompt += "Facts:\n"
            for f in bundle.facts:
                context_prompt += f"- {f.key}: {f.value}\n"
        if bundle.related_topics:
            context_prompt += "Related topics (from memory):\n"
            for rt in bundle.related_topics:
                context_prompt += f"- {rt.summary[:100]}\n"
        if bundle.memory_items:
            context_prompt += "Memory items:\n"
            for mi in bundle.memory_items:
                context_prompt += f"- {mi.content[:100]}\n"

        system_prompt = (
            "You are Eidolon CLI, a test interface for the Eidolon topic memory system. "
            "Respond naturally to the user's message. Use the provided context if relevant, "
            "but don't mention the memory system unless asked. Keep responses concise."
        )

        full_prompt = f"{context_prompt}\nUser: {text}\nAI:"
        ai_response = llm.complete(
            prompt=full_prompt,
            system_prompt=system_prompt,
            max_tokens=200,
        )

        # Step 4: Store the turn — store_turn returns the new topic ID
        old_topic_id = active_topic_id if is_new else None
        new_topic_label = topic_change.label if is_new else None

        new_topic_id = eidolon.store_turn(
            user_message=text,
            ai_response=ai_response,
            peer_id="user",
            environment=env,
            turn_id=turn_count,
            old_topic_id=old_topic_id,
            new_topic_label=new_topic_label,
        )

        # Update session state with the returned topic ID
        if new_topic_id is not None:
            active_topic_id = new_topic_id

        print(f"  AI  > {ai_response}")
        print()

    store.close()
    print("  Bye!")


if __name__ == "__main__":
    main()
