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

``TraceReader``/``TracePrinter`` are also reused by ``tools/tui`` (the
interactive drill-down explorer, see ``tools/trace_tui.py``) — keep their
public surface (``recent_turns``, ``events_for_request``,
``format_summary``, ``format_detail``) stable for that reason.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

_MAX_FIELD_CHARS = 500


def _truncate(value: object) -> str:
    text = str(value)
    if len(text) > _MAX_FIELD_CHARS:
        return text[:_MAX_FIELD_CHARS] + f"... ({len(text) - _MAX_FIELD_CHARS} more chars)"
    return text


@dataclass
class TurnSummary:
    """One row in the recent-turns list — a request_id plus enough to
    recognize it without opening it (see ``TraceReader.recent_turns``)."""

    request_id: str
    created_at: str
    session_id: str | None
    message_preview: str


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

    def recent_turns(self, limit: int = 20, session_id: str | None = None) -> list[TurnSummary]:
        """Return up to `limit` turns (one per request_id), most recent
        first, each with a short message preview for a picker UI.

        A "turn" is one request_id's worth of events (see
        ``core/__init__.py``'s request_id parameter — the Hermes adapter
        and CLI both mint one id per turn and reuse it across
        prefetch/sync_turn). The preview is pulled from whichever event
        carries the user's message first (``message`` or ``user_message``
        payload key, depending on event type).
        """
        summaries: list[TurnSummary] = []
        for request_id in self.recent_request_ids(session_id, limit):
            row = self._conn.execute(
                "SELECT payload, session_id, created_at FROM events "
                "WHERE request_id = ? ORDER BY id ASC LIMIT 1",
                (request_id,),
            ).fetchone()
            if row is None:
                continue
            payload = json.loads(row[0])
            message = payload.get("message") or payload.get("user_message") or ""
            summaries.append(
                TurnSummary(
                    request_id=request_id,
                    created_at=row[2],
                    session_id=row[1],
                    message_preview=_truncate(message) if len(str(message)) <= 80 else str(message)[:80] + "…",
                )
            )
        return summaries


class TracePrinter:
    """Formats trace events for display — one-line summaries for list
    views and full multi-line detail for a single event, plus the
    original terminal `print_request` output built on top of both.
    """

    def print_request(self, request_id: str, events: list[dict]) -> None:
        session_id = next((e["session_id"] for e in events if e["session_id"]), None)
        total_ms = sum(e["duration_ms"] for e in events if e["duration_ms"])
        print()
        print(f"═══ request {request_id}" + (f"  (session {session_id})" if session_id else "") + " ═══")
        print(f"  {len(events)} events, ~{total_ms:.1f}ms total")
        for event in events:
            print(self.format_detail(event))

    def format_detail(self, event: dict) -> str:
        """Multi-line detail view of one event: header line + full body."""
        event_type = event["event_type"]
        duration = f"{event['duration_ms']:.1f}ms" if event["duration_ms"] is not None else "-"
        header = f"\n  ── {event_type} ({duration}) [{event['created_at']}]"
        formatter = getattr(self, f"_format_{event_type}", None)
        if formatter is not None:
            body = formatter(event["payload"])
        else:
            body = f"    {_truncate(json.dumps(event['payload'], ensure_ascii=False))}"
        return f"{header}\n{body}"

    def format_summary(self, event: dict) -> str:
        """One-line description of one event, for a step-list row."""
        summarizer = getattr(self, f"_summary_{event['event_type']}", None)
        if summarizer is not None:
            return summarizer(event["payload"])
        return event["event_type"]

    # ── Detail formatters (used by format_detail) ──────────────────

    def _format_topic_detection(self, p: dict) -> str:
        return "\n".join([
            f"    message: {_truncate(p.get('message'))}",
            f"    current_topic_summary: {_truncate(p.get('current_topic_summary'))}",
            f"    result: {p.get('result_type')} {p.get('result_label') or ''}",
            f"    response: {_truncate(p.get('response'))}",
        ])

    def _format_extraction(self, p: dict) -> str:
        items = p.get("items", [])
        lines = [
            f"    user_message: {_truncate(p.get('user_message'))}",
            f"    items ({len(items)}):",
        ]
        lines += [f"      - [{item.get('item_type')}] {_truncate(item.get('content'))}" for item in items]
        if not items:
            lines.append(f"    raw response: {_truncate(p.get('response'))}")
        return "\n".join(lines)

    def _format_fact_consolidation(self, p: dict) -> str:
        upserts = p.get("upserts", [])
        deletes = p.get("deletes", [])
        lines = [
            f"    peer_id: {p.get('peer_id')}  new_items: {p.get('new_item_count')}  "
            f"total_facts_after: {p.get('total_facts_after')}",
            f"    upserts ({len(upserts)}):",
        ]
        lines += [f"      + {u.get('category')}.{u.get('key')} = {u.get('value')}" for u in upserts]
        lines.append(f"    deletes ({len(deletes)}):")
        lines += [f"      - {d.get('category')}.{d.get('key')}" for d in deletes]
        if not upserts and not deletes:
            lines.append(f"    raw response: {_truncate(p.get('response'))}")
        return "\n".join(lines)

    def _format_fact_correction(self, p: dict) -> str:
        return (
            f"    peer_id: {p.get('peer_id')}  action: {p.get('action')}\n"
            f"    {p.get('category')}.{p.get('key')} = {p.get('value')}"
        )

    def _format_topic_summarization(self, p: dict) -> str:
        return "\n".join([
            f"    topic_id: {p.get('topic_id')}  message_count: {p.get('message_count')}",
            f"    summary: {_truncate(p.get('summary'))}",
            f"    tags: {p.get('tags')}",
        ])

    def _format_embedding(self, p: dict) -> str:
        return "\n".join([
            f"    purpose: {p.get('purpose')}  dimension: {p.get('dimension')}",
            f"    text: {_truncate(p.get('text'))}",
        ])

    def _format_context_retrieval(self, p: dict) -> str:
        facts = p.get("facts", [])
        related_topics = p.get("related_topics", [])
        memory_items = p.get("memory_items", [])
        lines = [
            f"    message: {_truncate(p.get('message'))}",
            f"    topic_label: {p.get('topic_label')}  is_new_topic: {p.get('is_new_topic')}",
            f"    facts: {len(facts)}  related_topics: {len(related_topics)}  memory_items: {len(memory_items)}",
        ]
        lines += [f"      - fact: {f.get('category')}.{f.get('key')} = {f.get('value')}" for f in facts]
        lines += [f"      - topic #{rt.get('id')}: {_truncate(rt.get('summary'))}" for rt in related_topics]
        lines += [
            f"      - item [{mi.get('item_type')}] "
            f"({mi.get('source_peer')} about {mi.get('subject_peer')}): {_truncate(mi.get('content'))}"
            for mi in memory_items
        ]
        return "\n".join(lines)

    def _format_retrieval_candidate_scored(self, p: dict) -> str:
        mark = "✓" if p.get("included") else " "
        return (
            f"    [{mark}] {p.get('candidate_type')} #{p.get('candidate_id')}  "
            f"final={p.get('final_score', 0):.3f}  "
            f"semantic={p.get('semantic_raw', 0):.3f}→{p.get('semantic_weighted', 0):.3f}  "
            f"tag_overlap={p.get('tag_overlap_raw', 0):.3f}→{p.get('tag_overlap_weighted', 0):.3f}  "
            f"person={p.get('person_weighted', 0):.3f}"
        )

    def _format_turn_complete(self, p: dict) -> str:
        return "\n".join([
            f"    user_message: {_truncate(p.get('user_message'))}",
            f"    ai_response: {_truncate(p.get('ai_response'))}",
            f"    topic: {p.get('old_topic_id')} → {p.get('new_topic_id')} "
            f"({p.get('new_topic_label') or 'no change'})",
            f"    items_extracted: {p.get('items_extracted')}  fact_items: {p.get('fact_items')}",
        ])

    def _format_error(self, p: dict) -> str:
        return "\n".join([
            f"    step: {p.get('step')}",
            f"    exception: {_truncate(p.get('exception'))}",
        ])

    # ── Summary formatters (used by format_summary) ────────────────

    def _summary_topic_detection(self, p: dict) -> str:
        label = p.get("result_label")
        return f"matched against topic → {p.get('result_type')}" + (f" '{label}'" if label else "")

    def _summary_embedding(self, p: dict) -> str:
        return f"embedded [{p.get('purpose')}] → dim {p.get('dimension')}"

    def _summary_retrieval_candidate_scored(self, p: dict) -> str:
        mark = "included" if p.get("included") else "scored"
        return f"{p.get('candidate_type')} #{p.get('candidate_id')} {mark} → {p.get('final_score', 0):.3f}"

    def _summary_context_retrieval(self, p: dict) -> str:
        return (
            f"retrieved → {len(p.get('facts', []))} facts, "
            f"{len(p.get('related_topics', []))} topics, "
            f"{len(p.get('memory_items', []))} items"
        )

    def _summary_extraction(self, p: dict) -> str:
        return f"extracted {len(p.get('items', []))} item(s)"

    def _summary_fact_consolidation(self, p: dict) -> str:
        return (
            f"consolidated → {len(p.get('upserts', []))} upsert(s), "
            f"{len(p.get('deletes', []))} delete(s)"
        )

    def _summary_fact_correction(self, p: dict) -> str:
        return f"corrected fact → {p.get('action')} {p.get('category')}.{p.get('key')}"

    def _summary_topic_summarization(self, p: dict) -> str:
        return f"summarized topic #{p.get('topic_id')}"

    def _summary_turn_complete(self, p: dict) -> str:
        return f"turn complete → {p.get('items_extracted')} items, {p.get('fact_items')} facts"

    def _summary_error(self, p: dict) -> str:
        return f"ERROR in {p.get('step')}"


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
