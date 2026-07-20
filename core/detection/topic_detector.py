"""Topic detection for Eidolon.

Detects whether a user message continues the current topic or starts a new one.
Uses a mini-LLM call (~50 tokens) per message for real-time detection.
"""

from __future__ import annotations

import time

from core.interfaces import LLMProvider, Tracer
from core.models import TopicChange
from core.trace.null_tracer import NullTracer

# System prompt for the topic detection mini-LLM call.
# Designed to be small (~50 tokens) and produce structured output.
_DETECTION_SYSTEM_PROMPT = """You are a topic detection system for a conversation.
Your job is to determine whether the user's message continues the current topic
or starts a new one.

Respond with exactly one line:
- CONTINUE — if the message is a natural continuation of the current topic
- NEW_TOPIC:<label> — if the message introduces a new topic, where <label> is a short (2-5 word) description of the new topic

Rules:
- Short confirmations like "ok", "jep", "mach weiter", "ja bitte" are CONTINUE
- Explicit signals like "Lass uns über X reden", "was mich noch interessiert", "übrigens" indicate NEW_TOPIC
- If the message references a different subject without explicitly changing topic, it's still CONTINUE
- If there is no current topic (first message), always respond with NEW_TOPIC:<label>"""


def _build_detection_prompt(
    message: str,
    current_topic_summary: str | None,
    recent_messages: list[dict] | None = None,
) -> str:
    """Build the prompt for the topic detection LLM call.

    Parameters
    ----------
    message : str
        The user's message to classify.
    current_topic_summary : str | None
        Summary of the current/active topic, or None if no topic exists yet.
    recent_messages : list[dict] | None
        The current topic's most recent messages (oldest first), if any.
        A topic's `summary` is only ever set when it closes, so while it
        stays open the only anchor is its short, static creation-time
        label — recent_messages gives the detector real, evolving content
        to compare the new message against instead of just that frozen
        string.

    Returns
    -------
    str
        The prompt string to send to the LLM.
    """
    lines: list[str] = []
    if current_topic_summary:
        lines.append(f"Current topic: {current_topic_summary}")
    if recent_messages:
        lines.append("Recent conversation:")
        lines.extend(f"{m['role']}: {m['content']}" for m in recent_messages)
    lines.append(f"User: {message}")
    lines.append("→")
    return "\n".join(lines)


def _parse_detection_response(response: str) -> TopicChange:
    """Parse the LLM response into a TopicChange.

    Parameters
    ----------
    response : str
        Raw LLM response (e.g. "CONTINUE" or "NEW_TOPIC:label").

    Returns
    -------
    TopicChange
        Parsed result.
    """
    text = response.strip()

    # Case-insensitive check for NEW_TOPIC
    upper = text.upper()
    if upper.startswith("NEW_TOPIC:"):
        label = text[len("NEW_TOPIC:"):]  # keep original case
        # Also handle "NEW_TOPIC: label" with space after colon
        if label.startswith(" "):
            label = label[1:]
        # Clean up common LLM artifacts
        label = label.strip(".").strip('"').strip("'")
        if not label:
            label = "Unnamed topic"
        return TopicChange(type="NEW_TOPIC", label=label)

    if upper == "CONTINUE":
        return TopicChange(type="CONTINUE")

    # Fallback: if the LLM output something unexpected, try to interpret
    if "NEW_TOPIC" in upper:
        # Extract whatever comes after NEW_TOPIC
        idx = upper.index("NEW_TOPIC")
        label = text[idx + len("NEW_TOPIC"):]  # keep original case from that point
        label = label.strip(": .\"'")
        if not label:
            label = "Unnamed topic"
        return TopicChange(type="NEW_TOPIC", label=label)

    # Default to CONTINUE for unrecognized responses
    return TopicChange(type="CONTINUE")


class TopicDetector:
    """Detects topic transitions in real-time using a mini-LLM call.

    Parameters
    ----------
    llm : LLMProvider
        The LLM provider to use for detection calls.
    tracer : Tracer | None
        Debug tracer for recording the prompt/response/duration of each
        detection call. Defaults to a no-op tracer.
    """

    def __init__(self, llm: LLMProvider, tracer: Tracer | None = None) -> None:
        self._llm = llm
        self._tracer = tracer or NullTracer()

    def detect(
        self,
        message: str,
        current_topic_summary: str | None = None,
        recent_messages: list[dict] | None = None,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> TopicChange:
        """Determine whether a message continues or changes the topic.

        Parameters
        ----------
        message : str
            The user's message to classify.
        current_topic_summary : str | None
            Summary of the current topic, or None if this is the first message.
        recent_messages : list[dict] | None
            The current topic's most recent messages (oldest first), if any
            — see ``_build_detection_prompt`` for why this matters.
        request_id : str | None
            Correlates this call's trace event with others from the same
            top-level ``Eidolon`` call.
        session_id : str | None
            Adapter-level conversation identifier, for trace filtering.

        Returns
        -------
        TopicChange
            CONTINUE or NEW_TOPIC with a label.
        """
        prompt = _build_detection_prompt(message, current_topic_summary, recent_messages)
        started = time.perf_counter()
        response = ""
        try:
            response = self._llm.complete(
                prompt=prompt,
                system_prompt=_DETECTION_SYSTEM_PROMPT,
            )
        except Exception as exc:
            self._tracer.record(
                "error",
                {"step": "topic_detection", "exception": repr(exc)},
                request_id=request_id, session_id=session_id,
            )
        duration_ms = (time.perf_counter() - started) * 1000
        # An empty response (from a failed call above) naturally falls
        # through to CONTINUE — the same designed fallback as any other
        # unparseable response, see _parse_detection_response.
        result = _parse_detection_response(response)

        self._tracer.record(
            "topic_detection",
            {
                "message": message,
                "current_topic_summary": current_topic_summary,
                "recent_message_count": len(recent_messages or []),
                "prompt": prompt,
                "response": response,
                "result_type": result.type,
                "result_label": result.label,
            },
            request_id=request_id,
            session_id=session_id,
            duration_ms=duration_ms,
        )
        return result
