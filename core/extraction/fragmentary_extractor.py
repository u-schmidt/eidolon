"""Fragmentary extraction for Eidolon.

Extracts memory items from conversation turns — not raw chat logs.
Runs async after the LLM responds, so it does not block the turn.

Extraction targets per turn:
- Essence: "What was this about?"
- Poignant facts: specific, notable information
- Surprising moments: unexpected or unusual content
- Emotional coloring: tone or feeling (optional, experimental)

Every extracted item carries perspective attribution (source + subject).
"""

from __future__ import annotations

import json
import time
from typing import Literal

from core.interfaces import LLMProvider, Tracer
from core.models import EnvironmentSnapshot, MemoryItem
from core.trace.null_tracer import NullTracer

# System prompt for fragmentary extraction.
# Always in English — the LLM's working language.
# The `language` parameter only affects which stop words are used for
# tag extraction in the retriever, not the extraction prompt itself.
_EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction system for a conversation.
Your job is to extract meaningful fragments from a conversation turn — not the full transcript.

Analyze the user message and AI response, then extract 0-3 memory items.

Each item has two separate fields:

1. **item_type** — what kind of fragment this is (MUST be one of):
   - "essence" — What was this turn about? (1-2 sentence summary)
   - "fact" — A specific, notable piece of information revealed
   - "surprise" — Something unexpected or unusual
   - "feeling" — The emotional tone or feeling (optional)

2. **attribution_type** — who said what about whom (MUST be one of):
   - "self_statement" — user stated something about themselves
   - "observation" — AI observed something about the user
   - "inference" — AI inferred something about the user
   - "user_about_ai" — user stated something about the AI
   - "user_about_other" — user stated something about a third party
   - "ai_about_self" — AI stated something about itself
   - "system" — system-level information

IMPORTANT: "inference" is ONLY valid as attribution_type, NEVER as item_type.
If the AI makes an inference about the user, use item_type="essence" or item_type="fact"
with attribution_type="inference".

For each item, provide:
- content: The extracted fragment (1 sentence, concise)
- item_type: One of: essence, fact, surprise, feeling
- source_peer: Who said/contributed this? ("user" or "ai")
- subject_peer: Who is this about? ("user" or "ai" or "other")
- attribution_type: One of: self_statement, observation, inference, user_about_ai, user_about_other, ai_about_self, system

Respond with a JSON array only. Example:
[
  {
    "content": "User is frustrated with FitZone because equipment is often occupied",
    "item_type": "essence",
    "source_peer": "user",
    "subject_peer": "user",
    "attribution_type": "self_statement"
  }
]

If nothing worth extracting, respond with an empty array: []"""


def _build_extraction_prompt(
    user_message: str,
    ai_response: str,
    topic_label: str | None = None,
) -> str:
    """Build the prompt for the extraction LLM call.

    Parameters
    ----------
    user_message : str
        The user's message from this turn.
    ai_response : str
        The AI's response from this turn.
    topic_label : str | None
        The current topic label, if known.

    Returns
    -------
    str
        The prompt string to send to the LLM.
    """
    parts = []
    if topic_label:
        parts.append(f"Current topic: {topic_label}")
    parts.append(f"User: {user_message}")
    parts.append(f"AI: {ai_response}")
    return "\n".join(parts)


def _parse_extraction_response(response: str) -> list[dict]:
    """Parse the LLM response into a list of raw extraction dicts.

    Parameters
    ----------
    response : str
        Raw LLM response (expected to be a JSON array).

    Returns
    -------
    list[dict]
        Parsed extraction items.

    Raises
    ------
    ValueError
        If the response isn't valid JSON, or isn't a JSON array — the
        caller's error handling treats that as a failed pass (visible as
        an ``error`` trace event), not as "nothing to extract".
    """
    text = response.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        # Find the first [ or { after the opening fence
        start = text.find("[")
        if start == -1:
            start = text.find("{")
        if start != -1:
            text = text[start:]
        # Remove trailing fence
        end = text.rfind("```")
        if end != -1:
            text = text[:end]
        text = text.strip()

    try:
        items = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"malformed extraction response: {text!r}") from exc
    if not isinstance(items, list):
        raise ValueError(f"extraction response is not a JSON array: {text!r}")
    return items


class FragmentaryExtractor:
    """Extracts memory fragments from conversation turns.

    Runs async after the LLM responds — does not block the turn.

    Parameters
    ----------
    llm : LLMProvider
        The LLM provider to use for extraction calls.
    language : str
        Language code (default: "en").
        Note: extraction prompts are always in English regardless of this setting.
        The language parameter is reserved for future use (e.g., language-specific
        extraction tuning) and for consistency with the rest of the API.
    tracer : Tracer | None
        Debug tracer for recording the prompt/response/parsed items/duration
        of each extraction call. Defaults to a no-op tracer.
    """

    def __init__(
        self,
        llm: LLMProvider,
        language: str = "en",
        tracer: Tracer | None = None,
    ) -> None:
        self._llm = llm
        self._language = language
        self._tracer = tracer or NullTracer()

    def extract(
        self,
        user_message: str,
        ai_response: str,
        source_peer: str = "user",
        topic_label: str | None = None,
        environment: EnvironmentSnapshot | None = None,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> list[MemoryItem]:
        """Extract memory fragments from a conversation turn.

        Parameters
        ----------
        user_message : str
            The user's message from this turn.
        ai_response : str
            The AI's response from this turn.
        source_peer : str
            Default source peer for extracted items (default: "user").
        topic_label : str | None
            The current topic label, if known.
        environment : EnvironmentSnapshot | None
            The environment context at the time of this turn (V2).
        request_id : str | None
            Correlates this call's trace event with others from the same
            top-level ``Eidolon`` call.
        session_id : str | None
            Adapter-level conversation identifier, for trace filtering.

        Returns
        -------
        list[MemoryItem]
            Extracted memory items (0-3 per turn).
        """
        prompt = _build_extraction_prompt(user_message, ai_response, topic_label)
        started = time.perf_counter()
        response = ""
        try:
            response = self._llm.complete(
                prompt=prompt,
                system_prompt=_EXTRACTION_SYSTEM_PROMPT,
                max_tokens=1000,
            )
            raw_items = _parse_extraction_response(response)
        except Exception as exc:
            raw_items = []
            self._tracer.record(
                "error",
                {"step": "extraction", "exception": repr(exc), "response": response},
                request_id=request_id, session_id=session_id,
            )
        duration_ms = (time.perf_counter() - started) * 1000

        items = [
            MemoryItem(
                content=item.get("content", ""),
                item_type=item.get("item_type", "essence"),
                source_peer=item.get("source_peer", source_peer),
                subject_peer=item.get("subject_peer", source_peer),
                attribution_type=item.get("attribution_type", "self_statement"),
                context=environment,  # V2
            )
            for item in raw_items
            if item.get("content")
        ]

        self._tracer.record(
            "extraction",
            {
                "user_message": user_message,
                "ai_response": ai_response,
                "topic_label": topic_label,
                "response": response,
                "items": [
                    {"content": i.content, "item_type": i.item_type, "attribution_type": i.attribution_type}
                    for i in items
                ],
            },
            request_id=request_id,
            session_id=session_id,
            duration_ms=duration_ms,
        )
        return items
