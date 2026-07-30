"""Per-entity profile-card consolidation for Eidolon.

Track 2 of "fact-extraction hardening" (see core/summarization/
topic_summarizer.py's named-entity extraction, Track 1). Periodically
scans a workspace's newly closed topics for named-entity mentions,
promotes a recurring name into its own small fact profile ("card") once
it crosses a materialization threshold, and keeps that card up to date
via the same batch-review LLM pass FactConsolidator already uses for
the workspace-level profile — just scoped to one entity's own mentions
instead of the whole workspace's fact-type memory items.
"""

from __future__ import annotations

import json
import logging
import time

from core.interfaces import LLMProvider, MetricsRecorder, StorageProvider, Tracer
from core.metrics.null_metrics_recorder import NullMetricsRecorder
from core.models import Fact
from core.trace.null_tracer import NullTracer

logger = logging.getLogger(__name__)

# Mirrors FactsConfig.categories' default in core/config.py — same
# vocabulary, duplicated rather than imported for the same reason
# fact_consolidator.py duplicates it: this module stays usable without
# a config object.
_DEFAULT_CATEGORIES = [
    "identity", "relationship", "work", "health", "routine",
    "technology", "preference", "possession", "instruction",
]

_VALID_IMPORTANCE = {"low", "medium", "high"}


def _entity_consolidation_system_prompt(entity_name: str, target_max: int, categories: list[str]) -> str:
    category_list = ", ".join(categories)
    return f"""You maintain a small, curated profile of durable facts about "{entity_name}" — a
specific person, place, or organization the user has mentioned in conversation
(not the user themselves). Facts should be the kind of detail a close friend
would remember for years, not a passing mention.

Categories you must choose from: {category_list}.

Good profile facts (short, stable, atomic):
- identity: role = coworker
- relationship: connection_to_user = childhood friend
- preference: coffee_order = oat milk latte

Bad profile facts — do NOT add these, they belong to ordinary conversation
memory, not this profile:
- "came up once while talking about the weekend" (episodic, not durable)
- a fact that is actually about the user, not about "{entity_name}"

You will be shown the current profile for "{entity_name}" — each entry tagged
with its importance — and summaries of conversation topics that mentioned
"{entity_name}" since the last review. Decide what changes:
- If new material confirms or corrects an existing fact, reuse that fact's
  exact category and key — this updates it in place instead of creating a
  duplicate.
- Only add a new fact if it is genuinely durable and not already covered.
- Only delete a fact if it has been explicitly contradicted or is no longer true.
- Assign each fact an importance: "high" (core to who/what "{entity_name}"
  is to the user), "medium" (most stable facts), or "low" (minor detail).
  When genuinely unsure between "medium" and "low", prefer "low".
- Try to keep the total profile near {target_max} facts.
- Do not repeat facts that are unchanged — only output what changed.

Respond with JSON only:
{{"upserts": [{{"category": "...", "key": "...", "value": "...", "importance": "low|medium|high"}}], "deletes": [{{"category": "...", "key": "..."}}]}}
If nothing should change, respond with {{"upserts": [], "deletes": []}}"""


def _build_entity_consolidation_prompt(entity_name: str, current_facts: list[Fact], source_topics: list[dict]) -> str:
    facts_block = "\n".join(
        f"- {f.category}: {f.key} = {f.value} [importance={f.importance}]" for f in current_facts
    ) or "(none yet)"
    topics_block = "\n".join(
        f"- {t.get('label') or '(untitled)'}: {t.get('summary', '')}" for t in source_topics
    ) or "(none)"
    return (
        f'Current profile for "{entity_name}":\n{facts_block}\n\n'
        f'Conversation topics that mentioned "{entity_name}" since the last review:\n{topics_block}'
    )


def _parse_consolidation_response(response: str) -> tuple[list[dict], list[dict]]:
    """Parse the LLM's diff response into (upserts, deletes).

    Same shape as fact_consolidator.py's parser (mirrors it exactly,
    duplicated rather than imported across a private-name module
    boundary) — raises ValueError on a malformed response, which the
    caller treats as a failed pass for this one entity (its own
    checkpoint isn't advanced, retried next time) without affecting any
    other entity in the same batch.
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
        raise ValueError(f"malformed consolidation response: {text!r}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"consolidation response is not a JSON object: {text!r}")

    upserts = data.get("upserts", [])
    deletes = data.get("deletes", [])
    return (
        upserts if isinstance(upserts, list) else [],
        deletes if isinstance(deletes, list) else [],
    )


class EntityProfileConsolidator:
    """Materializes and reconciles per-named-entity profile cards.

    Parameters
    ----------
    storage : StorageProvider
        Backend for topics, entity mention counts, entity facts, and
        the two checkpoints (workspace scan cursor + per-entity
        consolidation cursor).
    llm : LLMProvider
        The LLM provider used for each entity's reconciliation call.
    threshold : int
        Distinct topics that must mention a name before it materializes
        into a real profile card — see ``EntityProfilesConfig``.
    target_max : int
        Soft guidance communicated to the LLM for one entity's target
        profile size — not enforced in code, only steered via the prompt.
    categories : list[str] | None
        Closed category list offered to the LLM. ``None`` uses the same
        default list as ``FactsConfig.categories``.
    tracer : Tracer | None
        Debug tracer recording each pass's prompt/response/duration and
        otherwise-silent failures. Defaults to a no-op tracer.
    metrics : MetricsRecorder | None
        Always-on operational-metrics recorder. Defaults to a no-op
        recorder.
    """

    def __init__(
        self,
        storage: StorageProvider,
        llm: LLMProvider,
        threshold: int = 2,
        target_max: int = 20,
        categories: list[str] | None = None,
        tracer: Tracer | None = None,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self._storage = storage
        self._llm = llm
        self._threshold = threshold
        self._target_max = target_max
        self._categories = categories if categories is not None else list(_DEFAULT_CATEGORIES)
        self._tracer = tracer or NullTracer()
        self._metrics = metrics or NullMetricsRecorder()

    def consolidate(
        self,
        workspace_id: str,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Scan `workspace_id`'s newly closed topics for named-entity
        mentions, materialize any entity that crosses the threshold,
        and reconcile every touched, materialized entity's profile card.

        No-ops (no LLM spend) if no topic has closed since the last scan.
        A failure during mention-scanning/materialization leaves the
        scan checkpoint unchanged so the same topics are retried next
        time. A failure reconciling one entity's card is independent —
        see `_consolidate_one` — and doesn't block the rest of the batch.

        Parameters
        ----------
        workspace_id : str
            The workspace whose entity profiles to scan/reconcile.
        request_id : str | None
            Correlates this call's trace events with others from the
            same top-level `Eidolon` call.
        session_id : str | None
            Adapter-level conversation identifier, for trace filtering.
        """
        since_topic_id = self._storage.get_entity_scan_checkpoint(workspace_id)
        new_topics = self._storage.get_topics_since(workspace_id, since_topic_id)
        if not new_topics:
            return

        started = time.perf_counter()
        try:
            touched_entities: set[str] = set()
            for topic in new_topics:
                names = topic.get("named_entities") or []
                if names:
                    self._storage.increment_entity_mentions(workspace_id, names)
                    touched_entities.update(names)

            newly_materialized = self._storage.materialize_entities(workspace_id, self._threshold)
            materialized = set(self._storage.get_materialized_entity_names(workspace_id))

            consolidated: list[str] = []
            for name in sorted(touched_entities):
                if name not in materialized:
                    continue
                checkpoint = self._storage.get_entity_consolidation_checkpoint(workspace_id, name)
                source_topics = self._storage.get_topics_mentioning_entity(workspace_id, name, checkpoint)
                if not source_topics:
                    continue
                if self._consolidate_one(
                    workspace_id, name, source_topics, request_id=request_id, session_id=session_id
                ):
                    consolidated.append(name)

            last_topic_id = max(t["id"] for t in new_topics)
            self._storage.set_entity_scan_checkpoint(workspace_id, last_topic_id)

            self._tracer.record(
                "entity_profile_scan",
                {
                    "workspace_id": workspace_id,
                    "new_topic_count": len(new_topics),
                    "touched_entities": sorted(touched_entities),
                    "newly_materialized": newly_materialized,
                    "consolidated_entities": consolidated,
                },
                request_id=request_id,
                session_id=session_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            self._tracer.record(
                "error",
                {"step": "entity_profile_scan", "workspace_id": workspace_id, "exception": repr(exc)},
                request_id=request_id,
                session_id=session_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            self._metrics.record_event(
                "error", "entity_profile_scan", repr(exc),
                workspace_id=workspace_id, request_id=request_id, session_id=session_id,
            )
            logger.warning("Eidolon entity_profile_scan failed (workspace_id=%s): %s", workspace_id, exc)
            # Scan checkpoint intentionally not advanced — the same
            # topics are rescanned at the next consolidation trigger.

    def _consolidate_one(
        self,
        workspace_id: str,
        entity_name: str,
        source_topics: list[dict],
        *,
        request_id: str | None,
        session_id: str | None,
    ) -> bool:
        """Reconcile one entity's profile card against `source_topics`.

        Returns True if the card was successfully reconciled (its own
        checkpoint advanced), False on failure — never raises, so one
        entity's LLM/parse failure doesn't abort the rest of the batch
        `consolidate` is working through.
        """
        started = time.perf_counter()
        try:
            current_facts = self._storage.get_entity_facts(workspace_id, [entity_name])
            prompt = _build_entity_consolidation_prompt(entity_name, current_facts, source_topics)
            response = self._llm.complete(
                prompt=prompt,
                system_prompt=_entity_consolidation_system_prompt(entity_name, self._target_max, self._categories),
            )
            upserts, deletes = _parse_consolidation_response(response)

            for entry in upserts:
                category = str(entry.get("category", "")).strip().lower()
                key = str(entry.get("key", "")).strip().lower()
                value = str(entry.get("value", "")).strip()
                importance = str(entry.get("importance", "medium")).strip().lower()
                if importance not in _VALID_IMPORTANCE:
                    importance = "medium"
                if category and key and value:
                    self._storage.set_entity_fact(
                        workspace_id, entity_name, category, key, value, source="ai", importance=importance
                    )

            for entry in deletes:
                category = str(entry.get("category", "")).strip().lower()
                key = str(entry.get("key", "")).strip().lower()
                if category and key:
                    self._storage.delete_entity_fact(workspace_id, entity_name, category, key)

            last_topic_id = max(t["id"] for t in source_topics)
            self._storage.set_entity_consolidation_checkpoint(workspace_id, entity_name, last_topic_id)

            self._tracer.record(
                "entity_profile_consolidation",
                {
                    "workspace_id": workspace_id,
                    "entity_name": entity_name,
                    "source_topic_count": len(source_topics),
                    "response": response,
                    "upserts": upserts,
                    "deletes": deletes,
                },
                request_id=request_id,
                session_id=session_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            self._metrics.record_llm_call(
                "entity_profile_consolidation", (time.perf_counter() - started) * 1000, True,
                workspace_id=workspace_id, request_id=request_id, session_id=session_id,
            )
            return True
        except Exception as exc:
            self._tracer.record(
                "error",
                {
                    "step": "entity_profile_consolidation",
                    "workspace_id": workspace_id,
                    "entity_name": entity_name,
                    "exception": repr(exc),
                },
                request_id=request_id,
                session_id=session_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            self._metrics.record_event(
                "error", "entity_profile_consolidation", repr(exc),
                workspace_id=workspace_id, request_id=request_id, session_id=session_id,
            )
            logger.warning(
                "Eidolon entity_profile_consolidation failed (workspace_id=%s, entity_name=%s): %s",
                workspace_id, entity_name, exc,
            )
            # This entity's own checkpoint intentionally not advanced —
            # its source topics are retried next time it's touched again.
            return False
