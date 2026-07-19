"""Turn detail — the 2 main phases of one turn (spec item c)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from tools.trace import TurnSummary
from tools.tui.phases import PREFETCH, SYNC_TURN, split_by_phase
from tools.tui.screens.phase_steps import PhaseStepsScreen


class TurnDetailScreen(Screen):
    """Shows the prefetch/sync_turn split for one turn. Select a phase
    to see its steps."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, turn: TurnSummary) -> None:
        super().__init__()
        self._turn = turn
        self._phase_events: dict[str, list[dict]] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(f"[b]{self._turn.message_preview or '(no message)'}[/b]", id="turn-header")
        yield DataTable(id="phases", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        events = self.app.reader.events_for_request(self._turn.request_id)
        prefetch_events, sync_turn_events = split_by_phase(events)
        self._phase_events = {PREFETCH: prefetch_events, SYNC_TURN: sync_turn_events}

        table = self.query_one("#phases", DataTable)
        table.add_columns("Phase", "Events", "Duration")
        for label, key in (("Prefetch", PREFETCH), ("Sync Turn", SYNC_TURN)):
            phase_events = self._phase_events[key]
            total_ms = sum(e["duration_ms"] for e in phase_events if e["duration_ms"])
            row_label = label if phase_events else f"{label} (no events)"
            table.add_row(row_label, str(len(phase_events)), f"{total_ms:.1f}ms", key=key)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        phase_key = event.row_key.value
        phase_events = self._phase_events[phase_key]
        if not phase_events:
            return
        label = "Prefetch" if phase_key == PREFETCH else "Sync Turn"
        self.app.push_screen(PhaseStepsScreen(label, phase_events))
