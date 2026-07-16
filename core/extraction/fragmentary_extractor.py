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
from typing import Literal

from core.interfaces import LLMProvider
from core.models import EnvironmentSnapshot, MemoryItem

# System prompt for fragmentary extraction.
# Always in English — the LLM's working language.
# The `language` parameter only affects which stop words are used for
# tag extraction in the retriever, not the extraction prompt itself.
_EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction system for a conversation.
Your job is to extract meaningful fragments from a conversation turn — not the full transcript.

Analyze the user message and AI response, then extract 0-3 memory items.
Each item must be one of these types:

- essence: What was this turn about? (1-2 sentence summary)
- fact: A specific, notable piece of information revealed
- surprise: Something unexpected or unusual
- feeling: The emotional tone or feeling (optional)

For each item, determine:
1. content: The extracted fragment (1 sentence, concise)
2. item_type: One of: essence, fact, surprise, feeling
3. source_peer: Who said/contributed this? ("user" or "ai")
4. subject_peer: Who is this about? ("user" or "ai" or "other")
5. attribution_type: The relationship between source and subject:
   - "self_statement" — user stated something about themselves
   - "observation" — AI observed something about the user
   - "inference" — AI inferred something about the user
   - "user_about_ai" — user stated something about the AI
   - "user_about_other" — user stated something about a third party
   - "ai_about_self" — AI stated something about itself
   - "system" — system-level information

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
        Parsed extraction items, or empty list on parse failure.
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
        if isinstance(items, list):
            return items
        return []
    except (json.JSONDecodeError, TypeError):
        return []


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
    """

    def __init__(
        self,
        llm: LLMProvider,
        language: str = "en",
    ) -> None:
        self._llm = llm
        self._language = language

    def extract(
        self,
        user_message: str,
        ai_response: str,
        source_peer: str = "user",
        topic_label: str | None = None,
        environment: EnvironmentSnapshot | None = None,
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

        Returns
        -------
        list[MemoryItem]
            Extracted memory items (0-3 per turn).
        """
        prompt = _build_extraction_prompt(user_message, ai_response, topic_label)
        response = self._llm.complete(
            prompt=prompt,
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            max_tokens=300,
        )

        raw_items = _parse_extraction_response(response)

        return [
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
