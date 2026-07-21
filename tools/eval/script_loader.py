"""Loads an eval script YAML file into an EvalScript."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tools.eval.models import (
    EndExpectation,
    EvalScript,
    EvalTurn,
    FactExpectation,
    MemoryItemHint,
    TopicHint,
    TurnExpectation,
)


class EvalScriptLoader:
    """Parses the locked eval-script YAML format (see plan-phase2.md) into
    an ``EvalScript``."""

    def load(self, path: str | Path) -> EvalScript:
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        turns = [self._parse_turn(raw) for raw in data.get("turns", [])]
        expect_end = self._parse_end(data.get("expect_end", {}))
        return EvalScript(
            name=data["name"],
            peer_id=data["peer_id"],
            turns=turns,
            language=data.get("language", "en"),
            expect_end=expect_end,
        )

    def _parse_turn(self, raw: dict[str, Any]) -> EvalTurn:
        expect_raw = raw.get("expect", {})
        memory_items = [
            MemoryItemHint(
                item_type=mi["item_type"],
                content_hint=mi["content_hint"],
                subject_peer=mi.get("subject_peer"),
            )
            for mi in expect_raw.get("memory_items", [])
        ]
        expect = TurnExpectation(
            topic_change=expect_raw.get("topic_change"),
            topic_label_hint=expect_raw.get("topic_label_hint"),
            memory_items=memory_items,
        )
        return EvalTurn(user=raw["user"], ai=raw["ai"], expect=expect)

    def _parse_end(self, raw: dict[str, Any]) -> EndExpectation:
        facts = [
            FactExpectation(category=f["category"], key=f["key"], value=f["value"])
            for f in raw.get("facts", [])
        ]
        topics = [TopicHint(label_hint=t["label_hint"]) for t in raw.get("topics", [])]
        return EndExpectation(facts=facts, topics=topics)
