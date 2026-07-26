"""Topic summarization for Eidolon.

Generates a concise summary and tags for a topic from its messages when
the topic closes. Called by ``Eidolon.store_turn`` at a topic transition:
its tags feed tag-overlap retrieval, and the topic is re-embedded from
the richer summary so semantic search stops matching the original
one-line guess. It also finalizes the topic's label if it's still
pending (label == "", see ``core/config.py``'s ``RelabelConfig`` and
``core/summarization/topic_relabeler.py``) — the fallback for any topic
that closes before ``TopicRelabeler`` ever got a chance to relabel it
mid-conversation. Never overwrites a label ``TopicRelabeler`` already
finalized.
"""

from __future__ import annotations

import logging
import time

from core.interfaces import EmbedderProvider, LLMProvider, MetricsRecorder, StorageProvider, Tracer
from core.metrics.null_metrics_recorder import NullMetricsRecorder
from core.trace.null_tracer import NullTracer

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation summarizer. "
    "Output a concise summary and relevant tags only."
)

# Only the first N characters of each message go into the transcript —
# enough to capture what a topic was about without sending whole turns.
_MESSAGE_PREVIEW_CHARS = 200


def _build_summary_prompt(messages: list[dict]) -> str:
    """Build the summarization prompt from a topic's messages.

    Always asks for a label alongside the summary/tags — cheap (a few
    extra output tokens on a call already happening), even when the
    topic's label was already finalized mid-conversation by
    TopicRelabeler and the produced one ends up discarded by the
    caller. Simpler than branching the prompt shape on whether a label
    is still needed.
    """
    transcript = "\n".join(
        f"{m['role']}: {m['content'][:_MESSAGE_PREVIEW_CHARS]}"
        for m in messages
    )
    return (
        f"Summarize this conversation topic in 1-2 sentences, "
        f"give it a short 2-5 word label, "
        f"and extract 3-5 key tags (comma-separated, lowercase).\n\n"
        f"{transcript}\n\n"
        f"Format:\n"
        f"Label: <2-5 word label>\n"
        f"Summary: <1-2 sentences>\n"
        f"Tags: <tag1, tag2, tag3>"
    )


def _parse_summary_response(response: str) -> tuple[str, str, list[str]]:
    """Parse the LLM response into (label, summary, tags).

    A missing/blank ``Summary:`` line yields an empty summary — the caller
    treats that as "no summary produced" and falls back rather than
    closing the topic with an empty string. A missing/blank ``Label:``
    line yields an empty label — the caller only persists a non-empty
    one, and only when the topic doesn't already have a real one.
    """
    label = ""
    summary = ""
    tags: list[str] = []
    for line in response.strip().split("\n"):
        if line.lower().startswith("label:"):
            label = line[len("label:"):].strip()
        elif line.lower().startswith("summary:"):
            summary = line[len("summary:"):].strip()
        elif line.lower().startswith("tags:"):
            raw_tags = line[len("tags:"):].strip()
            tags = [t.strip().lower() for t in raw_tags.split(",") if t.strip()]
    return label, summary, tags


class TopicSummarizer:
    """Summarizes a topic from its messages and closes it.

    Parameters
    ----------
    storage : StorageProvider
        Backend for reading the topic's messages and persisting the
        summary, tags, and updated embedding.
    llm : LLMProvider
        The LLM provider used for the summarization call.
    embedder : EmbedderProvider
        Provider for re-embedding the topic from its final summary.
    tracer : Tracer | None
        Debug tracer recording the prompt/response/duration of each
        summarization pass and otherwise-silent failures. Defaults to a
        no-op tracer.
    metrics : MetricsRecorder | None
        Always-on operational-metrics recorder. Defaults to a no-op
        recorder.
    """

    def __init__(
        self,
        storage: StorageProvider,
        llm: LLMProvider,
        embedder: EmbedderProvider,
        tracer: Tracer | None = None,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self._storage = storage
        self._llm = llm
        self._embedder = embedder
        self._tracer = tracer or NullTracer()
        self._metrics = metrics or NullMetricsRecorder()

    def summarize(
        self,
        topic_id: int,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Summarize ``topic_id`` from its messages, then close it.

        Loads all of the topic's messages, asks the LLM for a concise
        summary and tags, and — if a real summary comes back — closes the
        topic with it, stores the tags, and re-embeds it from the summary.

        Parameters
        ----------
        topic_id : int
            The topic to summarize and close.
        request_id : str | None
            Correlates this call's trace events with the enclosing
            ``store_turn`` call.
        session_id : str | None
            Adapter-level conversation identifier, for trace filtering.

        Returns
        -------
        bool
            True if a real summary was generated and the topic closed with
            it. False if summarization couldn't produce one (no messages,
            LLM failure, unparseable response) — the caller should close
            the topic with a fallback summary itself in that case.
        """
        started = time.perf_counter()
        try:
            messages = self._storage.get_messages_by_topic(topic_id)
            if not messages:
                return False

            # Fetched before the LLM call so persistence below only ever
            # checks a value nothing else in this method has touched yet —
            # a topic still pending (label == "") gets the label this call
            # produces; one TopicRelabeler already finalized is left alone.
            topic = self._storage.get_topic(topic_id)
            label_pending = topic is not None and topic["label"] == ""

            prompt = _build_summary_prompt(messages)
            response = self._llm.complete(
                prompt=prompt,
                system_prompt=_SUMMARY_SYSTEM_PROMPT,
            )
            label, summary, tags = _parse_summary_response(response)

            label_persisted = False
            if label and label_pending:
                self._storage.update_topic_label(topic_id, label)
                label_persisted = True

            if summary:
                self._storage.close_topic(topic_id, summary)
                if tags:
                    self._storage.update_topic_tags(topic_id, tags)
                # Re-embed with the final summary. The embedding generated
                # at topic creation (see store_turn) was based on the short
                # provisional label from topic detection, not this richer
                # summary — leaving it stale would mean semantic retrieval
                # keeps matching against the original guess forever.
                self._embed_summary(
                    topic_id, summary, request_id=request_id, session_id=session_id
                )

            self._tracer.record(
                "topic_summarization",
                {
                    "topic_id": topic_id,
                    "message_count": len(messages),
                    "transcript_preview": prompt[:500],
                    "response": response,
                    "label": label,
                    "label_persisted": label_persisted,
                    "summary": summary,
                    "tags": tags,
                },
                request_id=request_id,
                session_id=session_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            self._metrics.record_llm_call(
                "topic_summarization", (time.perf_counter() - started) * 1000, True,
                request_id=request_id, session_id=session_id,
            )
            return bool(summary)
        except Exception as exc:
            self._tracer.record(
                "error",
                {"step": "topic_summarization", "topic_id": topic_id, "exception": repr(exc)},
                request_id=request_id,
                session_id=session_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            self._metrics.record_event(
                "error", "topic_summarization", repr(exc),
                request_id=request_id, session_id=session_id,
            )
            logger.warning("Eidolon topic_summarization failed (topic_id=%s): %s", topic_id, exc)
            return False  # Summarization failure is non-critical; caller falls back

    def _embed_summary(
        self,
        topic_id: int,
        summary: str,
        *,
        request_id: str | None,
        session_id: str | None,
    ) -> None:
        """Re-embed a topic from its final summary. Non-critical — a
        failure is traced and swallowed, leaving the creation-time
        embedding in place."""
        try:
            embed_started = time.perf_counter()
            summary_embedding = self._embedder.embed(summary)
            embed_duration_ms = (time.perf_counter() - embed_started) * 1000
            self._storage.update_topic_embedding(topic_id, summary_embedding)
            self._tracer.record(
                "embedding",
                {"purpose": "topic_summary", "text": summary, "dimension": len(summary_embedding)},
                request_id=request_id, session_id=session_id, duration_ms=embed_duration_ms,
            )
            self._metrics.record_embedding_call(
                "topic_summary", embed_duration_ms, len(summary_embedding), True,
                entity_type="topic", entity_id=str(topic_id),
                request_id=request_id, session_id=session_id,
            )
        except Exception as exc:
            self._tracer.record(
                "error",
                {"step": "topic_summary_embedding", "exception": repr(exc)},
                request_id=request_id, session_id=session_id,
            )
            self._metrics.record_event(
                "error", "topic_summary_embedding", repr(exc),
                request_id=request_id, session_id=session_id,
            )
            logger.warning("Eidolon topic_summary_embedding failed (topic_id=%s): %s", topic_id, exc)
