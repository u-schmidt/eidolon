"""Topic summarization for Eidolon.

Generates a concise summary and tags for a topic from its messages when
the topic closes. Called by ``Eidolon.store_turn`` at a topic transition:
the summary replaces the topic's short creation-time label, its tags feed
tag-overlap retrieval, and the topic is re-embedded from the richer
summary so semantic search stops matching the original one-line guess.
"""

from __future__ import annotations

import time

from core.interfaces import EmbedderProvider, LLMProvider, StorageProvider, Tracer
from core.trace.null_tracer import NullTracer

# Historical value carried over unchanged from Eidolon._summarize_topic.
# NOTE: extraction was raised to 32000 for reasoning-model headroom; this
# path still has the same mid-output truncation risk (lower, since
# summaries are short) but is left at 1000 for now — a deliberate
# behavior-preserving carry-over, not an endorsement of the value.
_SUMMARY_MAX_TOKENS = 1000

_SUMMARY_SYSTEM_PROMPT = (
    "You are a conversation summarizer. "
    "Output a concise summary and relevant tags only."
)

# Only the first N characters of each message go into the transcript —
# enough to capture what a topic was about without sending whole turns.
_MESSAGE_PREVIEW_CHARS = 200


def _build_summary_prompt(messages: list[dict]) -> str:
    """Build the summarization prompt from a topic's messages."""
    transcript = "\n".join(
        f"{m['role']}: {m['content'][:_MESSAGE_PREVIEW_CHARS]}"
        for m in messages
    )
    return (
        f"Summarize this conversation topic in 1-2 sentences "
        f"and extract 3-5 key tags (comma-separated, lowercase).\n\n"
        f"{transcript}\n\n"
        f"Format:\n"
        f"Summary: <1-2 sentences>\n"
        f"Tags: <tag1, tag2, tag3>"
    )


def _parse_summary_response(response: str) -> tuple[str, list[str]]:
    """Parse the LLM response into (summary, tags).

    A missing/blank ``Summary:`` line yields an empty summary — the caller
    treats that as "no summary produced" and falls back rather than
    closing the topic with an empty string.
    """
    summary = ""
    tags: list[str] = []
    for line in response.strip().split("\n"):
        if line.lower().startswith("summary:"):
            summary = line[len("summary:"):].strip()
        elif line.lower().startswith("tags:"):
            raw_tags = line[len("tags:"):].strip()
            tags = [t.strip().lower() for t in raw_tags.split(",") if t.strip()]
    return summary, tags


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
    """

    def __init__(
        self,
        storage: StorageProvider,
        llm: LLMProvider,
        embedder: EmbedderProvider,
        tracer: Tracer | None = None,
    ) -> None:
        self._storage = storage
        self._llm = llm
        self._embedder = embedder
        self._tracer = tracer or NullTracer()

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

            prompt = _build_summary_prompt(messages)
            response = self._llm.complete(
                prompt=prompt,
                system_prompt=_SUMMARY_SYSTEM_PROMPT,
                max_tokens=_SUMMARY_MAX_TOKENS,
            )
            summary, tags = _parse_summary_response(response)

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
                    "summary": summary,
                    "tags": tags,
                },
                request_id=request_id,
                session_id=session_id,
                duration_ms=(time.perf_counter() - started) * 1000,
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
        except Exception as exc:
            self._tracer.record(
                "error",
                {"step": "topic_summary_embedding", "exception": repr(exc)},
                request_id=request_id, session_id=session_id,
            )
