#!/usr/bin/env python3
"""Eidolon trace explorer — interactive drill-down TUI over trace.db.

Requires the `tui` extra: uv sync --extra tui (or pip install eidolon[tui]).

Usage:
    python tools/trace_tui.py
    python tools/trace_tui.py --db ~/.eidolon/trace.db
    python tools/trace_tui.py --limit 50

For a non-interactive flat dump instead, see tools/trace.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path for direct execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from tools.tui.app import TraceExplorerApp
except ImportError:
    print(
        "Error: the 'tui' extra is not installed. Run: uv sync --extra tui",
        file=sys.stderr,
    )
    sys.exit(1)

from tools.trace import TraceReader


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Eidolon trace explorer — interactive drill-down debug TUI",
    )
    parser.add_argument("--db", default="~/.eidolon/trace.db", help="Trace database path")
    parser.add_argument("--limit", type=int, default=20, help="Number of recent turns to list")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        # Fail fast with a clear message before Textual even starts, rather
        # than surfacing a raw traceback inside the alternate screen buffer.
        TraceReader(args.db).close()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    TraceExplorerApp(db_path=args.db, limit=args.limit).run()


if __name__ == "__main__":
    main()
