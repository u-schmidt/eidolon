"""Loads persisted eval report JSON files into a flat, queryable shape.

Reads the JSON reports written by ``EvalReportFormatter.write`` — never
the markdown copies. Reports written before ``git_commit``/``tag``/
``scorecard`` existed are missing those keys; they're loaded with
placeholder values rather than treated as an error, since old reports
stay valid history, not broken data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.eval.scorecard import Scorecard

UNKNOWN_COMMIT = "unknown"


@dataclass
class PersistedReport:
    """One report file, reduced to what a comparison table needs."""

    path: Path
    recorded_at: datetime
    script_name: str
    model_identity: str
    git_commit: str
    tag: str | None
    scorecard: Scorecard | None


class ReportIndex:
    """Loads every ``*.json`` report in a directory into ``PersistedReport``s."""

    def load(self, directory: str | Path) -> list[PersistedReport]:
        directory = Path(directory)
        return [self._load_one(path) for path in sorted(directory.glob("*.json"))]

    def _load_one(self, path: Path) -> PersistedReport:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        scorecard_data = data.get("scorecard")
        return PersistedReport(
            path=path,
            recorded_at=datetime.fromtimestamp(path.stat().st_mtime),
            script_name=data["script_name"],
            model_identity=data["model_identity"],
            git_commit=data.get("git_commit", UNKNOWN_COMMIT),
            tag=data.get("tag"),
            scorecard=Scorecard(**scorecard_data) if scorecard_data is not None else None,
        )
