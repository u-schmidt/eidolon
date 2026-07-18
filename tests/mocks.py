"""Mock providers for testing Eidolon components.

These implement the same Protocols as real providers (LLMProvider, EmbedderProvider)
but return deterministic/simulated responses. No external dependencies needed.
"""

from __future__ import annotations

from core.models import TopicChange


class MockLLM:
    """A mock LLM that returns a predefined response.

    Implements ``LLMProvider`` protocol structurally (duck typing).
    """

    def __init__(self, response: str = "CONTINUE") -> None:
        self.response = response
        self.last_prompt: str | None = None
        self.last_system: str | None = None
        self.last_max_tokens: int | None = None

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 100,
    ) -> str:
        self.last_prompt = prompt
        self.last_system = system_prompt
        self.last_max_tokens = max_tokens
        return self.response


class MockEmbedder:
    """A mock embedder that returns a fixed vector.

    Implements ``EmbedderProvider`` protocol structurally (duck typing).
    Useful for testing without a real embedding model.
    """

    def __init__(self, vector: list[float] | None = None, dimension: int = 4) -> None:
        self.vector = vector or [0.1, 0.2, 0.3, 0.4][:dimension]
        self.last_text: str | None = None

    def embed(self, text: str) -> list[float]:
        self.last_text = text
        return self.vector


class SimulatedLLM:
    """A simulated LLM for the CLI that uses keyword matching.

    Implements ``LLMProvider`` protocol.
    Simulates topic detection and fragmentary extraction without a real LLM.
    Useful for quick CLI testing without any external dependencies.
    """

    def __init__(self) -> None:
        self._turn_count = 0

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 100,
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


class SpyTracer:
    """A tracer that records every call for test assertions.

    Implements ``Tracer`` protocol structurally (duck typing).
    """

    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.events: list[dict] = []

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.closed = True

    def record(
        self,
        event_type: str,
        payload: dict,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        self.events.append({
            "event_type": event_type,
            "payload": payload,
            "request_id": request_id,
            "session_id": session_id,
            "duration_ms": duration_ms,
        })

    def events_of_type(self, event_type: str) -> list[dict]:
        return [e for e in self.events if e["event_type"] == event_type]


class SimulatedEmbedder:
    """A simulated embedder that returns a deterministic vector.

    Implements ``EmbedderProvider`` protocol.
    Vector is based on text length for reproducibility.
    Useful for quick CLI testing without LM Studio or sentence-transformers.
    """

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension

    def embed(self, text: str) -> list[float]:
        h = hash(text) % 1000
        return [float((h + i) % 10) / 10.0 for i in range(self._dimension)]
