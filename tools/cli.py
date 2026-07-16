#!/usr/bin/env python3
"""Eidolon CLI — standalone test tool for the core library.

Run without Hermes to test topic detection, context retrieval, and memory storage.
Uses SQLite for storage and sentence-transformers for embeddings (no external API needed).

Usage:
    python -m tools.cli
    python -m tools.cli --db /tmp/eidolon_test.db
    python -m tools.cli --language de
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
from core.models import EnvironmentSnapshot
from core.stores.sqlite_store import SqliteStore


class CliLLM:
    """A mock LLM that reads responses from a simple text file or uses defaults.

    For testing without a real LLM, this simulates topic detection and extraction.
    """

    def __init__(self, mode: str = "interactive") -> None:
        self._mode = mode
        self._turn_count = 0

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 100,
    ) -> str:
        self._turn_count += 1

        # Simulate topic detection
        if system_prompt and "topic detection" in system_prompt.lower():
            return self._simulate_detection(prompt)

        # Simulate extraction
        if system_prompt and "memory extraction" in system_prompt.lower():
            return self._simulate_extraction(prompt)

        return ""

    def _simulate_detection(self, prompt: str) -> str:
        """Simulate topic detection based on keywords in the message."""
        # Extract the user message from the prompt
        lines = prompt.strip().split("\n")
        user_line = next((l for l in lines if l.startswith("User:")), "")
        message = user_line[5:].strip().lower() if user_line else ""

        # Check for topic change signals
        change_signals = [
            "lass uns", "übrigens", "ach ja", "was mich noch interessiert",
            "neues thema", "wechseln", "sprechen wir über", "erzähl mir von",
            "let's", "by the way", "speaking of", "what about",
            "change topic", "new topic", "talk about",
        ]
        for signal in change_signals:
            if signal in message:
                # Extract a label from the message
                words = message.split()
                # Find the first meaningful word after the signal
                for i, w in enumerate(words):
                    if w in ("über", "von", "about", "of") and i + 1 < len(words):
                        label = " ".join(words[i+1:i+4]).strip(".,!?")
                        if label:
                            return f"NEW_TOPIC:{label.title()}"
                # Fallback: use the last 2-3 words
                label = " ".join(words[-3:]).strip(".,!?")
                if label:
                    return f"NEW_TOPIC:{label.title()}"
                return "NEW_TOPIC:New Topic"

        # Default: CONTINUE for everything else
        return "CONTINUE"

    def _simulate_extraction(self, prompt: str) -> str:
        """Simulate fragmentary extraction."""
        lines = prompt.strip().split("\n")
        user_line = next((l for l in lines if l.startswith("User:")), "")
        message = user_line[5:].strip() if user_line else ""

        if not message or len(message) < 5:
            return "[]"

        # Create a simple essence extraction
        return (
            '[{"content": "User said: ' + message[:80] + '", '
            '"item_type": "essence", '
            '"source_peer": "user", '
            '"subject_peer": "user", '
            '"attribution_type": "self_statement"}]'
        )


class CliEmbedder:
    """A simple mock embedder that returns a fixed vector.

    Good enough for testing the CLI without loading a real model.
    """

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Return a deterministic vector based on text length."""
        # Simple hash-based vector for reproducibility
        h = hash(text) % 1000
        return [float((h + i) % 10) / 10.0 for i in range(self._dimension)]


def print_banner() -> None:
    """Print the Eidolon CLI banner."""
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║         Eidolon CLI  v0.1.0          ║")
    print("  ║   Topic Memory System — Test Tool    ║")
    print("  ╚══════════════════════════════════════╝")
    print()


def print_help() -> None:
    """Print available commands."""
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
    parser = argparse.ArgumentParser(description="Eidolon CLI — test the core library")
    parser.add_argument("--db", default="", help="Path to SQLite database (default: temp file)")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    args = parser.parse_args()

    # Database
    db_path = args.db or str(Path(tempfile.mkdtemp()) / "eidolon_cli.db")
    print(f"  Database: {db_path}")

    # Initialize components
    store = SqliteStore(db_path)
    store.connect()

    llm = CliLLM()
    embedder = CliEmbedder()

    eidolon = Eidolon(
        storage=store,
        llm=llm,
        embedder=embedder,
        language=args.language,
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
                # List all topics
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
                # Get all items directly from the store
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

        # Step 3: Simulate an AI response (we don't have a real LLM here)
        ai_response = f"[AI response to: {text[:60]}]"

        # Step 4: Store the turn
        old_topic_id = active_topic_id if is_new else None
        new_topic_label = topic_change.label if is_new else None

        eidolon.store_turn(
            user_message=text,
            ai_response=ai_response,
            peer_id="user",
            environment=env,
            old_topic_id=old_topic_id,
            new_topic_label=new_topic_label,
        )

        # Update session state
        if is_new and topic_change.label:
            # Find the newly created topic (it's the last one)
            for tid in range(1, 100):
                t = store.get_topic(tid)
                if t is None:
                    break
                if t["label"] == topic_change.label and t["status"] == "active":
                    active_topic_id = t["id"]
                    break

        print(f"  AI  > {ai_response}")
        print()

    store.close()
    print("  Bye!")


if __name__ == "__main__":
    main()
