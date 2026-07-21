"""CLI entry point for browsing eval reports as a flat comparison table.

Usage:
    uv run python tools/analyze_reports.py
    uv run python tools/analyze_reports.py --tag prompt-17
    uv run python tools/analyze_reports.py --script "bike saga" --model deepseek

Reads every report JSON under --reports-dir (never the markdown copies)
and prints one row per run, oldest first — no grouping or averaging,
just the raw timeline. See tools/eval/report_index.py and
tools/eval/report_table.py for the loading/rendering logic this wires
together.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path for direct execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tools.eval.report_index import PersistedReport, ReportIndex
from tools.eval.report_table import ReportTableFormatter


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browse Eidolon eval reports as a flat comparison table.")
    parser.add_argument(
        "--reports-dir", default="tools/eval/reports", help="Directory containing the report JSON files",
    )
    parser.add_argument("--script", default=None, help="Filter: substring match on script name")
    parser.add_argument("--tag", default=None, help="Filter: substring match on the run tag")
    parser.add_argument("--commit", default=None, help="Filter: substring match on the git commit")
    parser.add_argument("--model", default=None, help="Filter: substring match on model identity")
    return parser


def _matches(report: PersistedReport, args: argparse.Namespace) -> bool:
    if args.script and args.script.lower() not in report.script_name.lower():
        return False
    if args.tag and (report.tag is None or args.tag.lower() not in report.tag.lower()):
        return False
    if args.commit and args.commit.lower() not in report.git_commit.lower():
        return False
    return not (args.model and args.model.lower() not in report.model_identity.lower())


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    reports = ReportIndex().load(args.reports_dir)
    reports = [r for r in reports if _matches(r, args)]
    reports.sort(key=lambda r: r.recorded_at)

    if not reports:
        print(f"No reports found under {args.reports_dir} matching the given filters.")
        return

    print(ReportTableFormatter().format(reports))
    print(f"\n{len(reports)} report(s)")


if __name__ == "__main__":
    main()
