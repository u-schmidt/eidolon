"""Facts profile consolidation for Eidolon.

Periodically reconciles a peer's durable-facts profile — a small,
curated set of stable identity-level facts — against everything new
learned since the last pass. Unlike per-turn extraction, this looks at
the *entire* current profile in one LLM call, so updates replace stale
values in place instead of accumulating duplicates, and the profile
stays small because the LLM is judging what's actually durable rather
than converting every notable turn detail into a permanent row.
"""

from __future__ import annotations

import json
import time

from core.interfaces import LLMProvider, StorageProvider, Tracer
from core.models import Fact, MemoryItem
from core.trace.null_tracer import NullTracer


def _consolidation_system_prompt(target_max: int) -> str:
    return f"""You maintain a small, curated profile of durable facts about a person — the
kind of information a close friend would remember for years, not everyday
conversation details.

Good profile facts (short, stable, atomic):
- identity: name = Alex
- location: city = Hamburg
- relationships: partner = Sam
- health: condition = seasonal allergies
- work: employer = Acme Corp

Bad profile facts — do NOT add these, they belong to ordinary conversation
memory, not this profile:
- "spent two days debugging a sys.modules cache bug" (episodic, not durable)
- "worked on an Electronic Flight Bag project at a previous company" (narrative detail)

You will be shown the current profile and new material revealed since the
last review. Decide what changes:
- If new material confirms or corrects an existing fact, reuse that fact's
  exact category and key — this updates it in place instead of creating a
  duplicate (e.g. if location changes, update the existing "location" key,
  don't add a new "current_city" key).
- Only add a new fact if it is genuinely durable and not already covered.
- Only delete a fact if it has been explicitly contradicted or is no longer true.
- Try to keep the total profile near {target_max} facts — if you're well
  over, consolidate overlapping entries or drop the least important ones.
- Do not repeat facts that are unchanged — only output what changed.

Respond with JSON only:
{{"upserts": [{{"category": "...", "key": "...", "value": "..."}}], "deletes": [{{"category": "...", "key": "..."}}]}}
If nothing should change, respond with {{"upserts": [], "deletes": []}}"""


def _build_consolidation_prompt(current_facts: list[Fact], new_items: list[MemoryItem]) -> str:
    facts_block = "\n".join(f"- {f.category}: {f.key} = {f.value}" for f in current_facts) or "(none yet)"
    items_block = "\n".join(f"- {item.content}" for item in new_items)
    return f"Current profile:\n{facts_block}\n\nNew information revealed since last review:\n{items_block}"


def _parse_consolidation_response(response: str) -> tuple[list[dict], list[dict]]:
    """Parse the LLM's diff response into (upserts, deletes).

    Raises ValueError on a malformed response — the caller's error
    handling treats that as a failed pass (checkpoint not advanced, so
    the same items are retried next time), not as "nothing changed".
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


class FactConsolidator:
    """Reconciles a peer's durable-facts profile via periodic batch review.

    Parameters
    ----------
    storage : StorageProvider
        Backend for facts, fact-type memory items, and the consolidation
        checkpoint.
    llm : LLMProvider
        The LLM provider used for the reconciliation call.
    target_max : int
        Soft guidance communicated to the LLM for the target profile
        size — not enforced in code, only steered via the prompt.
    tracer : Tracer | None
        Debug tracer recording the prompt/response/duration of each
        consolidation pass and otherwise-silent failures. Defaults to a
        no-op tracer.
    """

    def __init__(
        self,
        storage: StorageProvider,
        llm: LLMProvider,
        target_max: int = 100,
        tracer: Tracer | None = None,
    ) -> None:
        self._storage = storage
        self._llm = llm
        self._target_max = target_max
        self._tracer = tracer or NullTracer()

    def consolidate(
        self,
        peer_id: str,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Reconcile `peer_id`'s durable-facts profile.

        No-ops (no LLM call spent) if nothing new has been extracted
        since the last successful pass. On failure, the checkpoint is
        left unchanged so the same items are retried next time.

        Parameters
        ----------
        peer_id : str
            The peer whose profile to reconcile.
        request_id : str | None
            Correlates this call's trace event with others from the
            same top-level ``Eidolon`` call.
        session_id : str | None
            Adapter-level conversation identifier, for trace filtering.
        """
        since_id = self._storage.get_consolidation_checkpoint(peer_id)
        new_items = self._storage.get_fact_items_since(peer_id, since_id)
        if not new_items:
            return

        started = time.perf_counter()
        try:
            current_facts = self._storage.get_facts(peer_id)
            prompt = _build_consolidation_prompt(current_facts, new_items)
            response = self._llm.complete(
                prompt=prompt,
                system_prompt=_consolidation_system_prompt(self._target_max),
                max_tokens=2000,
            )
            upserts, deletes = _parse_consolidation_response(response)

            for entry in upserts:
                category = str(entry.get("category", "")).strip().lower()
                key = str(entry.get("key", "")).strip().lower()
                value = str(entry.get("value", "")).strip()
                if category and key and value:
                    self._storage.set_fact(peer_id, category, key, value, source="ai")

            for entry in deletes:
                category = str(entry.get("category", "")).strip().lower()
                key = str(entry.get("key", "")).strip().lower()
                if category and key:
                    self._storage.delete_fact(peer_id, category, key)

            latest_item_id = max(item.id for item in new_items if item.id is not None)
            self._storage.set_consolidation_checkpoint(peer_id, latest_item_id)

            self._tracer.record(
                "fact_consolidation",
                {
                    "peer_id": peer_id,
                    "new_item_count": len(new_items),
                    "response": response,
                    "upserts": upserts,
                    "deletes": deletes,
                    "total_facts_after": len(self._storage.get_facts(peer_id)),
                },
                request_id=request_id,
                session_id=session_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            self._tracer.record(
                "error",
                {"step": "fact_consolidation", "peer_id": peer_id, "exception": repr(exc)},
                request_id=request_id,
                session_id=session_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            # Checkpoint intentionally not advanced — these items are
            # retried at the next consolidation trigger.
