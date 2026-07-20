"""Simulated LLM and embedder for Eidolon.

Zero-dependency stand-ins for real providers — used as the zero-config
default for the ``llm``/``embedder`` model roles (see
``core/config.py::default_models``) and for standalone CLI testing
without Ollama, LM Studio, or any other external service running.
"""

from __future__ import annotations


class SimulatedLLM:
    """A simulated LLM that uses keyword matching.

    Implements ``LLMProvider`` protocol structurally (duck typing).
    Simulates topic detection and fragmentary extraction without a real LLM.
    """

    def __init__(self) -> None:
        self._turn_count = 0

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self._turn_count += 1

        if system_prompt and "topic detection" in system_prompt.lower():
            return self._simulate_detection(prompt)
        if system_prompt and "memory extraction" in system_prompt.lower():
            return self._simulate_extraction(prompt)
        return ""

    def _simulate_detection(self, prompt: str) -> str:
        lines = prompt.strip().split("\n")
        user_line = next((l for l in lines if l.startswith("User:")), "")
        message = user_line[5:].strip().lower() if user_line else ""

        change_signals = [
            "lass uns", "übrigens", "ach ja", "was mich noch interessiert",
            "neues thema", "wechseln", "sprechen wir über", "erzähl mir von",
            "let's", "by the way", "speaking of", "what about",
            "change topic", "new topic", "talk about",
        ]
        for signal in change_signals:
            if signal in message:
                words = message.split()
                for i, w in enumerate(words):
                    if w in ("über", "von", "about", "of") and i + 1 < len(words):
                        label = " ".join(words[i+1:i+4]).strip(".,!?")
                        if label:
                            return f"NEW_TOPIC:{label.title()}"
                label = " ".join(words[-3:]).strip(".,!?")
                if label:
                    return f"NEW_TOPIC:{label.title()}"
                return "NEW_TOPIC:New Topic"

        return "CONTINUE"

    def _simulate_extraction(self, prompt: str) -> str:
        lines = prompt.strip().split("\n")
        user_line = next((l for l in lines if l.startswith("User:")), "")
        message = user_line[5:].strip() if user_line else ""

        if not message or len(message) < 5:
            return "[]"

        return (
            '[{"content": "User said: ' + message[:80] + '", '
            '"item_type": "essence", '
            '"source_peer": "user", '
            '"subject_peer": "user", '
            '"attribution_type": "self_statement"}]'
        )


class SimulatedEmbedder:
    """A simulated embedder that returns a deterministic vector.

    Implements ``EmbedderProvider`` protocol structurally (duck typing).
    Vector is based on text hash for reproducibility — no real model.

    Parameters
    ----------
    dimension : int
        Vector dimension (default: 4).
    """

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        h = hash(text) % 1000
        return [float((h + i) % 10) / 10.0 for i in range(self._dimension)]
