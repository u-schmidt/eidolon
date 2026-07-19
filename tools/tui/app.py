"""Eidolon trace explorer — interactive drill-down TUI over trace.db.

Read-only: list turns -> pick a turn -> pick prefetch/sync_turn -> pick a
step -> full detail (and, for context_retrieval, drill into the scored
candidate ranking behind what got injected). See tools/trace_tui.py for
the command-line entry point.
"""

from __future__ import annotations

from textual.app import App

from tools.trace import TraceReader
from tools.tui.screens.turn_list import TurnListScreen


class TraceExplorerApp(App):
    """Root app — owns the TraceReader and starts the drill-down stack."""

    TITLE = "Eidolon Trace Explorer"
    CSS = """
    DataTable {
        height: 1fr;
    }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, db_path: str, limit: int = 20) -> None:
        super().__init__()
        self.reader = TraceReader(db_path)
        self.limit = limit

    def on_mount(self) -> None:
        self.push_screen(TurnListScreen())

    def on_unmount(self) -> None:
        self.reader.close()
