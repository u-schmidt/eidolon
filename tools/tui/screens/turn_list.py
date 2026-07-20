"""Turn list — the trace explorer's entry screen (spec item a)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header

from tools.trace import TurnSummary
from tools.tui.screens.turn_detail import TurnDetailScreen

if TYPE_CHECKING:
    from tools.tui.app import TraceExplorerApp


class TurnListScreen(Screen):
    """Lists recent turns, most recent first. Select one to drill in."""

    BINDINGS = [("escape", "app.quit", "Quit")]

    def __init__(self) -> None:
        super().__init__()
        self._turns_by_key: dict[str, TurnSummary] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="turns", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        app = cast("TraceExplorerApp", self.app)
        table = self.query_one("#turns", DataTable)
        table.add_columns("Time", "Message", "Session")
        for turn in app.reader.recent_turns(limit=app.limit):
            key = turn.request_id
            self._turns_by_key[key] = turn
            table.add_row(
                turn.created_at,
                turn.message_preview or "(no message)",
                turn.session_id or "-",
                key=key,
            )
        if not self._turns_by_key:
            self.notify("No trace turns found.", severity="warning")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # Row keys are the str request_ids we set in on_mount, never None.
        key = event.row_key.value
        assert key is not None
        turn = self._turns_by_key[key]
        self.app.push_screen(TurnDetailScreen(turn))
