"""Phase steps — the individual steps within prefetch or sync_turn,
with one-line results (spec item e)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header

from tools.trace import TracePrinter
from tools.tui.screens.step_detail import StepDetailScreen

_formatter = TracePrinter()


class PhaseStepsScreen(Screen):
    """Lists the steps of one phase (prefetch or sync_turn) with a
    one-line summary each, e.g. "matched against topic -> continue"."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, phase_label: str, events: list[dict]) -> None:
        super().__init__()
        self._phase_label = phase_label
        self._events = events

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="steps", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = self._phase_label
        table = self.query_one("#steps", DataTable)
        table.add_columns("Step", "Result", "Duration")
        for i, event in enumerate(self._events):
            duration = f"{event['duration_ms']:.1f}ms" if event["duration_ms"] is not None else "-"
            table.add_row(
                event["event_type"], _formatter.format_summary(event), duration, key=str(i)
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        index = int(event.row_key.value)
        self.app.push_screen(StepDetailScreen(self._events[index], self._events))
