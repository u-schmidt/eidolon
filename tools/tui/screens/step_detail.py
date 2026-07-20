"""Step detail — full detail for one step, and (for context_retrieval)
what got injected plus a drill-down into the candidates behind it
(spec items e detail, f, and the entry point into g)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from tools.trace import TracePrinter
from tools.tui.screens.candidate_ranking import CandidateRankingScreen

_formatter = TracePrinter()


class StepDetailScreen(Screen):
    """Full detail for one trace event. context_retrieval additionally
    gets a small table offering to drill into the full scored ranking
    behind its related_topics/memory_items — the rest just show
    TracePrinter's formatted detail text."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, event: dict, phase_events: list[dict]) -> None:
        super().__init__()
        self._event = event
        self._phase_events = phase_events
        self._is_context_retrieval = event["event_type"] == "context_retrieval"

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(Static(_formatter.format_detail(self._event)), id="detail-scroll")
        if self._is_context_retrieval:
            yield DataTable(id="drilldown", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        if not self._is_context_retrieval:
            return
        table = self.query_one("#drilldown", DataTable)
        table.add_columns("Full candidate ranking", "Scored")
        for label, candidate_type in (
            ("Related Topics", "topic"),
            ("Memory Items", "memory_item"),
        ):
            summary_event = next(
                (e for e in self._phase_events
                 if e["event_type"] == "retrieval_scoring_summary"
                 and e["payload"].get("candidate_type") == candidate_type),
                None,
            )
            count = summary_event["payload"]["total_scored"] if summary_event else 0
            table.add_row(label, str(count), key=candidate_type)
        # The VerticalScroll above is focusable too (it handles scroll
        # keys) and would otherwise steal focus on mount, leaving the
        # actionable drilldown table unreachable from the keyboard.
        table.focus()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        candidate_type = event.row_key.value
        self.app.push_screen(CandidateRankingScreen(candidate_type, self._phase_events))
