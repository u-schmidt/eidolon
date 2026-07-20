"""Candidate ranking — the full scored list behind what got injected
(spec item g)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header


class CandidateRankingScreen(Screen):
    """Scored candidates of one type ("topic" or "memory_item") for a
    turn — the ranking behind context_retrieval's included subset, as
    persisted by the storage layer's retrieval_scoring_summary event
    (already sorted and capped there, so no re-sort/slice needed here)."""

    BINDINGS = [("escape", "app.pop_screen", "Back")]

    def __init__(self, candidate_type: str, phase_events: list[dict]) -> None:
        super().__init__()
        self._candidate_type = candidate_type
        self._phase_events = phase_events

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="candidates", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = f"{self._candidate_type} candidates"
        summary_event = next(
            (e for e in self._phase_events
             if e["event_type"] == "retrieval_scoring_summary"
             and e["payload"].get("candidate_type") == self._candidate_type),
            None,
        )
        candidates = summary_event["payload"]["top_candidates"] if summary_event else []

        table = self.query_one("#candidates", DataTable)
        table.add_columns("ID", "Final", "Semantic raw→weighted", "Tag overlap raw→weighted", "Person", "Included")
        for c in candidates:
            table.add_row(
                str(c.get("candidate_id")),
                f"{c.get('final_score', 0):.3f}",
                f"{c.get('semantic_raw', 0):.3f}→{c.get('semantic_weighted', 0):.3f}",
                f"{c.get('tag_overlap_raw', 0):.3f}→{c.get('tag_overlap_weighted', 0):.3f}",
                f"{c.get('person_weighted', 0):.3f}",
                "✓" if c.get("included") else "",
            )
        if not candidates:
            self.notify(f"No scored {self._candidate_type} candidates for this turn.", severity="warning")
