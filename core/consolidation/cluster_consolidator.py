"""Recurring-topic clustering for Eidolon.

Runs once per closed topic, right after it's summarized (see
``Eidolon.store_turn``). Decides whether the newly closed topic is a
continuation of an existing recurring thread — grouping it into a
single, continuously-rewritten "cluster" summary instead of leaving it
to compete individually in associative retrieval, where a long-lived
thread with more instances than the retrieval limit would otherwise
lose its older pieces to context.
"""

from __future__ import annotations

import json
import logging
import time

from core.interfaces import EmbedderProvider, LLMProvider, MetricsRecorder, StorageProvider, Tracer
from core.metrics.null_metrics_recorder import NullMetricsRecorder
from core.trace.null_tracer import NullTracer

logger = logging.getLogger(__name__)


def _cluster_system_prompt(target_max_milestones: int, target_max_tags: int) -> str:
    return f"""You track recurring conversation threads over time — the same underlying
subject (e.g. "buying a bike") that resurfaces across separate, closed conversation topics on
different days. Your job is to decide whether a newly closed topic is a continuation of one of the
candidate threads shown below, or something new and unrelated to all of them.

Each candidate is either an existing recurring thread ("cluster") with its current state and
milestone history, or a single past topic that hasn't been grouped with anything yet.

If the new topic continues one of the candidates, produce:

- current_state: a short (2-4 sentence), present-tense description of where this thread stands
  now. Rewritten fresh every time — do not try to preserve old phrasing or accumulate a
  chronological log of every session. Someone reading only this should understand what the thread
  is about and where it stands today, not its full history.
- milestones: a small list of genuine shifts in direction or scope for this thread (e.g. "started
  as an image viewer", "pivoted to a generation helper") — not bug fixes, session recaps, or minor
  decisions. Try to keep this near {target_max_milestones} entries — if you're at or over that,
  merge adjacent/redundant entries or drop ones superseded by a later one (e.g. "got engaged"
  folding into "got married") rather than just appending.
- tags: up to {target_max_tags}, most relevant first. Exclude file paths, filenames, and
  config/doc files (e.g. "frontend/", "agents.md", ".gitignore") and generic technology, library,
  or framework names (e.g. "fastapi", "pytorch", "sqlite") — keep the words that would actually
  help find this thread again.
- label: a short label for the thread.

If the new topic isn't related to any candidate, say so — don't force a match.

Respond with JSON only:
{{"match_index": <1-based index of the matching candidate, or null>, "label": "<short label for the thread>", "current_state": "<rewritten current-state description, or empty string if match_index is null>", "milestones": ["<short milestone>", ...], "tags": ["<tag>", ...]}}"""


def _build_cluster_prompt(candidates: list[dict], topic: dict) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
        if c["candidate_kind"] == "cluster":
            milestones_block = " | ".join(c.get("milestones") or []) or "(none yet)"
            lines.append(
                f"{i}. [existing recurring thread, {c['topic_count']} topic(s) so far] {c['label']}\n"
                f"   Current state: {c['summary']}\n"
                f"   Milestones so far: {milestones_block}"
            )
        else:
            lines.append(f"{i}. [single past topic, not yet grouped] {c['label']}: {c['summary']}")
    candidates_block = "\n".join(lines)
    tags = ", ".join(topic.get("tags", [])) or "none"
    return (
        f"Candidates:\n{candidates_block}\n\n"
        f"Newly closed topic:\n{topic['label']}: {topic['summary']} "
        f"(tags: {tags}, closed: {topic.get('closed_at') or 'just now'})"
    )


def _parse_cluster_response(response: str) -> dict:
    """Parse the LLM's match decision.

    Raises ValueError on a malformed response — the caller traces and
    swallows it, leaving the topic unclustered rather than guessing.
    """
    text = response.strip()
    if text.startswith("```"):
        start = text.find("{")
        if start != -1:
            text = text[start:]
        end = text.rfind("```")
        if end != -1:
            text = text[:end]
        text = text.strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"malformed cluster consolidation response: {text!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"cluster consolidation response is not a JSON object: {text!r}")
    return data


class ClusterConsolidator:
    """Groups recurring topics into a single, continuously-updated cluster.

    Parameters
    ----------
    storage : StorageProvider
        Backend for topic/cluster reads and writes.
    llm : LLMProvider
        The LLM provider used for the membership-judgment call.
    embedder : EmbedderProvider
        Provider for re-embedding a cluster from its rewritten summary.
    shortlist_size : int
        Max candidates (existing clusters + still-unclustered topics)
        shown to the LLM per topic close — keeps the prompt bounded
        regardless of how much history a workspace has accumulated.
    target_max_milestones : int
        Soft guidance communicated to the LLM for how many milestones a
        cluster's evolution list should hold — not enforced in code,
        only steered via the prompt (same stance as
        ``FactConsolidator.target_max``).
    target_max_tags : int
        Soft guidance communicated to the LLM for how many tags a
        cluster should carry — same not-enforced-in-code stance as
        ``target_max_milestones``.
    tracer : Tracer | None
        Debug tracer recording the prompt/response/duration of each
        pass and otherwise-silent failures. Defaults to a no-op tracer.
    metrics : MetricsRecorder | None
        Always-on operational-metrics recorder. Defaults to a no-op
        recorder.
    """

    def __init__(
        self,
        storage: StorageProvider,
        llm: LLMProvider,
        embedder: EmbedderProvider,
        shortlist_size: int = 5,
        target_max_milestones: int = 8,
        target_max_tags: int = 15,
        tracer: Tracer | None = None,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self._storage = storage
        self._llm = llm
        self._embedder = embedder
        self._shortlist_size = shortlist_size
        self._target_max_milestones = target_max_milestones
        self._target_max_tags = target_max_tags
        self._tracer = tracer or NullTracer()
        self._metrics = metrics or NullMetricsRecorder()

    def consolidate_topic(
        self,
        topic_id: int,
        workspace_id: str,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Decide whether ``topic_id`` continues an existing recurring
        thread, and if so, merge it in.

        No-ops (no LLM call spent) if the cheap embedding/tag shortlist
        finds nothing worth showing the LLM — the common case for a
        genuinely one-off topic. On failure, the topic is simply left
        unclustered; a clustering failure must never break the turn.

        Parameters
        ----------
        topic_id : int
            The just-closed topic to place.
        workspace_id : str
            The workspace whose cluster set to search/update.
        request_id : str | None
            Correlates this call's trace events with the enclosing
            ``store_turn`` call.
        session_id : str | None
            Adapter-level conversation identifier, for trace filtering.
        """
        started = time.perf_counter()
        try:
            topic = self._storage.get_topic(topic_id)
            if topic is None:
                return

            candidates = self._storage.search_cluster_candidates(
                topic_id, workspace_id, limit=self._shortlist_size,
                tracer=self._tracer, request_id=request_id,
            )
            if not candidates:
                return

            prompt = _build_cluster_prompt(candidates, topic)
            system_prompt = _cluster_system_prompt(self._target_max_milestones, self._target_max_tags)
            response = self._llm.complete(prompt=prompt, system_prompt=system_prompt)
            decision = _parse_cluster_response(response)

            match_index = decision.get("match_index")
            label = str(decision.get("label", "")).strip()
            current_state = str(decision.get("current_state", "")).strip()
            raw_milestones = decision.get("milestones", [])
            raw_milestones = raw_milestones if isinstance(raw_milestones, list) else []
            milestones = [str(m).strip() for m in raw_milestones if str(m).strip()]
            raw_tags = decision.get("tags", [])
            raw_tags = raw_tags if isinstance(raw_tags, list) else []
            tags = [str(t).strip().lower() for t in raw_tags if str(t).strip()]
            resulting_cluster_id: int | None = None

            if isinstance(match_index, int) and 1 <= match_index <= len(candidates) and current_state:
                matched = candidates[match_index - 1]
                embedding = self._embed_summary(current_state, request_id=request_id, session_id=session_id)

                if matched["candidate_kind"] == "cluster":
                    resulting_cluster_id = matched["id"]
                    self._storage.update_cluster(
                        resulting_cluster_id, summary=current_state, label=label or matched["label"],
                        tags=tags, milestones=milestones, embedding=embedding,
                    )
                    self._storage.assign_topic_to_cluster(topic_id, resulting_cluster_id)
                else:  # "topic" — a raw, previously unclustered topic. Promote both into a new cluster.
                    resulting_cluster_id = self._storage.create_cluster(
                        workspace_id, label=label or matched["label"], summary=current_state,
                        tags=tags, milestones=milestones, embedding=embedding,
                    )
                    self._storage.assign_topic_to_cluster(matched["id"], resulting_cluster_id)
                    self._storage.assign_topic_to_cluster(topic_id, resulting_cluster_id)

            self._tracer.record(
                "cluster_consolidation",
                {
                    "topic_id": topic_id,
                    "workspace_id": workspace_id,
                    "candidates": [
                        {"kind": c["candidate_kind"], "id": c["id"], "label": c["label"]}
                        for c in candidates
                    ],
                    "response": response,
                    "match_index": match_index,
                    "resulting_cluster_id": resulting_cluster_id,
                },
                request_id=request_id,
                session_id=session_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            self._metrics.record_llm_call(
                "cluster_consolidation", (time.perf_counter() - started) * 1000, True,
                workspace_id=workspace_id, request_id=request_id, session_id=session_id,
            )
        except Exception as exc:
            self._tracer.record(
                "error",
                {"step": "cluster_consolidation", "topic_id": topic_id, "exception": repr(exc)},
                request_id=request_id,
                session_id=session_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            self._metrics.record_event(
                "error", "cluster_consolidation", repr(exc),
                workspace_id=workspace_id, request_id=request_id, session_id=session_id,
            )
            logger.warning("Eidolon cluster_consolidation failed (topic_id=%s): %s", topic_id, exc)

    def _embed_summary(
        self,
        summary: str,
        *,
        request_id: str | None,
        session_id: str | None,
    ) -> list[float] | None:
        """Embed a cluster's rewritten summary. Non-critical — a failure
        is traced and swallowed, leaving the prior embedding in place
        (update_cluster) or the new cluster embedding-less (create_cluster),
        same graceful-degradation pattern as TopicSummarizer's."""
        try:
            embed_started = time.perf_counter()
            embedding = self._embedder.embed(summary)
            embed_duration_ms = (time.perf_counter() - embed_started) * 1000
            self._tracer.record(
                "embedding",
                {"purpose": "cluster_summary", "text": summary, "dimension": len(embedding)},
                request_id=request_id, session_id=session_id,
                duration_ms=embed_duration_ms,
            )
            self._metrics.record_embedding_call(
                "cluster_summary", embed_duration_ms, len(embedding), True,
                request_id=request_id, session_id=session_id,
            )
            return embedding
        except Exception as exc:
            self._tracer.record(
                "error",
                {"step": "cluster_summary_embedding", "exception": repr(exc)},
                request_id=request_id, session_id=session_id,
            )
            self._metrics.record_event(
                "error", "cluster_summary_embedding", repr(exc),
                request_id=request_id, session_id=session_id,
            )
            logger.warning("Eidolon cluster_summary_embedding failed: %s", exc)
            return None
