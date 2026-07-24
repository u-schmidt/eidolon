"""Mid-conversation topic relabeling for Eidolon.

Runs once per topic, mid-conversation, the first time a still-pending
topic (label == "") accumulates enough messages — see
``Eidolon.store_turn``'s CONTINUE branch and ``core/config.py``'s
``RelabelConfig``. A cheaper, narrower cousin of ``TopicSummarizer``:
produces only a label from the transcript so far, never closes the
topic, never touches summary/tags. ``TopicSummarizer`` (at topic close)
is the fallback for topics that close before ever reaching the
threshold — whichever mechanism finalizes a topic's label first wins;
this class only ever acts on a topic whose label is still "".
"""

from __future__ import annotations

import logging
import time

from core.interfaces import LLMProvider, StorageProvider, Tracer
from core.trace.null_tracer import NullTracer

logger = logging.getLogger(__name__)

_RELABEL_SYSTEM_PROMPT = (
    "You label conversation topics. Output a short 2-5 word label only, nothing else."
)

# Only the first N characters of each message go into the transcript —
# mirrors TopicSummarizer's own preview length.
_MESSAGE_PREVIEW_CHARS = 200


def _build_relabel_prompt(messages: list[dict]) -> str:
    """Build the relabeling prompt from a topic's messages so far."""
    transcript = "\n".join(
        f"{m['role']}: {m['content'][:_MESSAGE_PREVIEW_CHARS]}"
        for m in messages
    )
    return f"Give this ongoing conversation topic a short label:\n\n{transcript}\n\nLabel:"


def _parse_relabel_response(response: str) -> str:
    """Clean up the raw label line. Blank/unparseable -> "" (caller skips persisting)."""
    return response.strip().strip(".").strip('"').strip("'")


class TopicRelabeler:
    """Produces a real label for a still-pending, still-open topic.

    Parameters
    ----------
    storage : StorageProvider
        Backend for reading the topic's messages so far and persisting
        the new label.
    llm : LLMProvider
        The LLM provider used for the relabeling call.
    tracer : Tracer | None
        Debug tracer recording each relabel pass and otherwise-silent
        failures. Defaults to a no-op tracer.
    """

    def __init__(
        self,
        storage: StorageProvider,
        llm: LLMProvider,
        tracer: Tracer | None = None,
    ) -> None:
        self._storage = storage
        self._llm = llm
        self._tracer = tracer or NullTracer()

    def relabel(
        self,
        topic_id: int,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Produce and persist a label for a still-pending topic.

        Loads the topic's messages so far and asks the LLM for a short
        label — never closes the topic, never touches summary/tags/
        embedding (see module docstring for why).

        Returns
        -------
        bool
            True if a label was produced and persisted. False on any
            failure (no messages, LLM failure, unparseable/empty
            response) — non-critical, the topic just stays pending for
            TopicSummarizer to try again at eventual close.
        """
        started = time.perf_counter()
        try:
            messages = self._storage.get_messages_by_topic(topic_id)
            if not messages:
                return False

            prompt = _build_relabel_prompt(messages)
            response = self._llm.complete(
                prompt=prompt,
                system_prompt=_RELABEL_SYSTEM_PROMPT,
            )
            label = _parse_relabel_response(response)

            if label:
                self._storage.update_topic_label(topic_id, label)

            self._tracer.record(
                "topic_relabel",
                {
                    "topic_id": topic_id,
                    "message_count": len(messages),
                    "prompt": prompt,
                    "response": response,
                    "label": label,
                },
                request_id=request_id,
                session_id=session_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            return bool(label)
        except Exception as exc:
            self._tracer.record(
                "error",
                {"step": "topic_relabel", "topic_id": topic_id, "exception": repr(exc)},
                request_id=request_id,
                session_id=session_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            logger.warning("Eidolon topic_relabel failed (topic_id=%s): %s", topic_id, exc)
            return False  # Relabeling failure is non-critical; TopicSummarizer falls back at close
