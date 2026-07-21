"""Tests for EvalScriptLoader."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tools.eval.script_loader import EvalScriptLoader

_SCRIPT_YAML = """
name: "ok-continuation and a genuine topic switch"
peer_id: "eval-peer-01"
language: "de"

turns:
  - user: "Ich will ein neues Rennrad kaufen."
    ai: "Welche Preisklasse?"
    expect:
      topic_change: NEW_TOPIC
      topic_label_hint: "buying a road bike"
      memory_items:
        - item_type: essence
          subject_peer: user
          content_hint: "user wants a road bike"

  - user: "so um die 2000 Euro"
    ai: "Verstanden."
    expect:
      topic_change: CONTINUE

expect_end:
  facts:
    - {category: preference, key: bike_budget, value: "2000 EUR"}
  topics:
    - label_hint: "buying a road bike"
"""


@pytest.fixture
def script_path() -> str:
    tmp = tempfile.mkdtemp()
    path = Path(tmp) / "script.yaml"
    path.write_text(_SCRIPT_YAML, encoding="utf-8")
    return str(path)


class TestEvalScriptLoader:
    def test_loads_top_level_fields(self, script_path: str):
        script = EvalScriptLoader().load(script_path)

        assert script.name == "ok-continuation and a genuine topic switch"
        assert script.peer_id == "eval-peer-01"
        assert script.language == "de"
        assert len(script.turns) == 2

    def test_loads_turn_expectations_and_memory_item_hints(self, script_path: str):
        script = EvalScriptLoader().load(script_path)

        first = script.turns[0]
        assert first.user == "Ich will ein neues Rennrad kaufen."
        assert first.ai == "Welche Preisklasse?"
        assert first.expect.topic_change == "NEW_TOPIC"
        assert first.expect.topic_label_hint == "buying a road bike"
        assert len(first.expect.memory_items) == 1
        hint = first.expect.memory_items[0]
        assert hint.item_type == "essence"
        assert hint.subject_peer == "user"
        assert hint.content_hint == "user wants a road bike"

    def test_turn_without_memory_items_defaults_to_empty_list(self, script_path: str):
        script = EvalScriptLoader().load(script_path)

        second = script.turns[1]
        assert second.expect.topic_change == "CONTINUE"
        assert second.expect.memory_items == []

    def test_loads_expect_end(self, script_path: str):
        script = EvalScriptLoader().load(script_path)

        assert len(script.expect_end.facts) == 1
        fact = script.expect_end.facts[0]
        assert (fact.category, fact.key, fact.value) == ("preference", "bike_budget", "2000 EUR")
        assert script.expect_end.topics == [script.expect_end.topics[0]]
        assert script.expect_end.topics[0].label_hint == "buying a road bike"

    def test_missing_expect_end_defaults_to_empty(self):
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / "minimal.yaml"
        path.write_text('name: "minimal"\npeer_id: "p1"\nturns: []\n', encoding="utf-8")

        script = EvalScriptLoader().load(str(path))

        assert script.expect_end.facts == []
        assert script.expect_end.topics == []
        assert script.language == "en"
