"""Mock providers for testing Eidolon components.

These implement the same Protocols as real providers (LLMProvider, EmbedderProvider)
but return deterministic/simulated responses. No external dependencies needed.
"""

from __future__ import annotations


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
