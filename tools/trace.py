#!/usr/bin/env python3
"""Eidolon trace printer — human-readable view of debug trace events.

Reads the separate trace database written by SqliteTracer (see
``core/trace/``) when a profile runs with ``debug.enabled: true``, and
prints the most recent request(s) grouped and formatted per event type.

Usage:
    python tools/trace.py
    python tools/trace.py --db ~/.eidolon/trace.db
    python tools/trace.py --session sess_1
    python tools/trace.py --last 5

Deliberately minimal — no query language beyond --session/--last. Extend
once real usage shows what's actually needed.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_MAX_FIELD_CHARS = 500


def _truncate(value: object) -> str:
    text = str(value)
    if len(text) > _MAX_FIELD_CHARS:
        return text[:_MAX_FIELD_CHARS] + f"... ({len(text) - _MAX_FIELD_CHARS} more chars)"
    return text


class TraceReader:
    """Read-only access to a trace database.

    Parameters
    ----------
    db_path : str
        Filesystem path to the trace database file.
    """

    def __init__(self, db_path: str) -> None:
        self._path = Path(db_path).expanduser().resolve()
        if not self._path.exists():
            raise FileNotFoundError(
                f"Trace database not found at {self._path}. "
                f"Run with debug.enabled: true in eidolon.yaml first."
            )
        self._conn = sqlite3.connect(str(self._path))

    def close(self) -> None:
        self._conn.close()

    def recent_request_ids(self, session_id: str | None, limit: int) -> list[str]:
        """Return up to `limit` request_ids, most recent first."""
        if session_id:
            rows = self._conn.execute(
                "SELECT request_id, MAX(id) AS last_id FROM events "
                "WHERE request_id IS NOT NULL AND session_id = ? "
                "GROUP BY request_id ORDER BY last_id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT request_id, MAX(id) AS last_id FROM events "
                "WHERE request_id IS NOT NULL "
                "GROUP BY request_id ORDER BY last_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [r[0] for r in rows]

    def events_for_request(self, request_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT event_type, payload, duration_ms, session_id, created_at "
            "FROM events WHERE request_id = ? ORDER BY id ASC",
            (request_id,),
        ).fetchall()
        return [
            {
                "event_type": r[0],
                "payload": json.loads(r[1]),
                "duration_ms": r[2],
                "session_id": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]


class TracePrinter:
    """Formats and prints one request's trace events to the terminal."""

    def print_request(self, request_id: str, events: list[dict]) -> None:
        session_id = next((e["session_id"] for e in events if e["session_id"]), None)
        total_ms = sum(e["duration_ms"] for e in events if e["duration_ms"])
        print()
        print(f"═══ request {request_id}" + (f"  (session {session_id})" if session_id else "") + " ═══")
        print(f"  {len(events)} events, ~{total_ms:.1f}ms total")
        for event in events:
            self._print_event(event)

    def _print_event(self, event: dict) -> None:
        event_type = event["event_type"]
        duration = f"{event['duration_ms']:.1f}ms" if event["duration_ms"] is not None else "-"
        print(f"\n  ── {event_type} ({duration}) [{event['created_at']}]")

        printer = getattr(self, f"_print_{event_type}", None)
        if printer is not None:
            printer(event["payload"])
        else:
            print(f"    {_truncate(json.dumps(event['payload'], ensure_ascii=False))}")

    def _print_topic_detection(self, p: dict) -> None:
        print(f"    message: {_truncate(p.get('message'))}")
        print(f"    current_topic_summary: {_truncate(p.get('current_topic_summary'))}")
        print(f"    result: {p.get('result_type')} {p.get('result_label') or ''}")
        print(f"    response: {_truncate(p.get('response'))}")

    def _print_extraction(self, p: dict) -> None:
        print(f"    user_message: {_truncate(p.get('user_message'))}")
        items = p.get("items", [])
        print(f"    items ({len(items)}):")
        for item in items:
            print(f"      - [{item.get('item_type')}] {_truncate(item.get('content'))}")
        if not items:
            print(f"    raw response: {_truncate(p.get('response'))}")

    def _print_fact_extraction(self, p: dict) -> None:
        facts = p.get("stored_facts", [])
        print(f"    stored_facts ({len(facts)}):")
        for f in facts:
            print(f"      - {f.get('category')}.{f.get('key')} = {f.get('value')}")
        if not facts:
            print(f"    raw response: {_truncate(p.get('response'))}")

    def _print_topic_summarization(self, p: dict) -> None:
        print(f"    topic_id: {p.get('topic_id')}  message_count: {p.get('message_count')}")
        print(f"    summary: {_truncate(p.get('summary'))}")
        print(f"    tags: {p.get('tags')}")

    def _print_embedding(self, p: dict) -> None:
        print(f"    purpose: {p.get('purpose')}  dimension: {p.get('dimension')}")
        print(f"    text: {_truncate(p.get('text'))}")

    def _print_context_retrieval(self, p: dict) -> None:
        facts = p.get("facts", [])
        related_topics = p.get("related_topics", [])
        memory_items = p.get("memory_items", [])
        print(f"    message: {_truncate(p.get('message'))}")
        print(f"    topic_label: {p.get('topic_label')}  is_new_topic: {p.get('is_new_topic')}")
        print(
            f"    facts: {len(facts)}  "
            f"related_topics: {len(related_topics)}  "
            f"memory_items: {len(memory_items)}"
        )
        for f in facts:
            print(f"      - fact: {f.get('category')}.{f.get('key')} = {f.get('value')}")
        for rt in related_topics:
            print(f"      - topic #{rt.get('id')}: {_truncate(rt.get('summary'))}")
        for mi in memory_items:
            print(
                f"      - item [{mi.get('item_type')}] "
                f"({mi.get('source_peer')} about {mi.get('subject_peer')}): "
                f"{_truncate(mi.get('content'))}"
            )

    def _print_retrieval_candidate_scored(self, p: dict) -> None:
        mark = "✓" if p.get("included") else " "
        print(
            f"    [{mark}] {p.get('candidate_type')} #{p.get('candidate_id')}  "
            f"final={p.get('final_score', 0):.3f}  "
            f"semantic={p.get('semantic_raw', 0):.3f}→{p.get('semantic_weighted', 0):.3f}  "
            f"tag_overlap={p.get('tag_overlap_raw', 0):.3f}→{p.get('tag_overlap_weighted', 0):.3f}  "
            f"person={p.get('person_weighted', 0):.3f}"
        )

    def _print_turn_complete(self, p: dict) -> None:
        print(f"    user_message: {_truncate(p.get('user_message'))}")
        print(f"    ai_response: {_truncate(p.get('ai_response'))}")
        print(
            f"    topic: {p.get('old_topic_id')} → {p.get('new_topic_id')} "
            f"({p.get('new_topic_label') or 'no change'})"
        )
        print(f"    items_extracted: {p.get('items_extracted')}  fact_items: {p.get('fact_items')}")

    def _print_error(self, p: dict) -> None:
        print(f"    step: {p.get('step')}")
        print(f"    exception: {_truncate(p.get('exception'))}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Eidolon trace printer — human-readable debug trace view",
    )
    parser.add_argument("--db", default="~/.eidolon/trace.db", help="Trace database path")
    parser.add_argument("--session", default=None, help="Filter to one session_id")
    parser.add_argument("--last", type=int, default=1, help="Number of most recent requests to show")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        reader = TraceReader(args.db)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    request_ids = reader.recent_request_ids(session_id=args.session, limit=args.last)
    if not request_ids:
        print("No trace events found" + (f" for session {args.session}" if args.session else "") + ".")
        reader.close()
        return

    printer = TracePrinter()
    for request_id in reversed(request_ids):  # oldest of the selected batch first
        printer.print_request(request_id, reader.events_for_request(request_id))

    reader.close()


if __name__ == "__main__":
    main()
