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
import logging
import time

from core.interfaces import LLMProvider, MetricsRecorder, StorageProvider, Tracer
from core.metrics.null_metrics_recorder import NullMetricsRecorder
from core.models import Fact, MemoryItem
from core.trace.null_tracer import NullTracer

logger = logging.getLogger(__name__)

# FragmentaryExtractor's literal subject_peer label for the human (see
# core/extraction/fragmentary_extractor.py) — a different value space from
# peer_id, which identifies *whose* profile this is, not who a memory item
# is about. Facts are keyed by the real peer_id; the memory items that
# feed them are always subject_peer="user"/"ai"/"other". Querying
# get_fact_items_since with peer_id instead of this constant silently
# starves consolidation forever for any peer_id that isn't literally the
# string "user".
_PROFILE_SUBJECT = "user"

# Mirrors FactsConfig.categories' default in core/config.py — duplicated
# rather than imported so this module stays usable without a config
# object (matches target_max's own plain-default precedent below).
_DEFAULT_CATEGORIES = [
    "identity", "relationship", "work", "health", "routine",
    "technology", "preference", "possession", "instruction",
]

_VALID_IMPORTANCE = {"low", "medium", "high"}


def _consolidation_system_prompt(target_max: int, categories: list[str]) -> str:
    category_list = ", ".join(categories)
    return f"""You maintain a small, curated profile of durable facts about a person — the
kind of information a close friend would remember for years, not everyday
conversation details.

Categories you must choose from: {category_list}.

Good profile facts (short, stable, atomic):
- identity: name = Alex
- relationship: partner = Sam
- health: condition = seasonal allergies
- work: employer = Acme Corp

Bad profile facts — do NOT add these, they belong to ordinary conversation
memory, not this profile:
- "spent two days debugging a sys.modules cache bug" (episodic, not durable)
- "worked on an Electronic Flight Bag project at a previous company" (narrative detail)

An episodic-sounding mention can still reveal a durable fact underneath —
extract that durable part, don't reject the whole mention just because its
surface framing sounds temporary:
- Mention: "waiting on a reply to a letter to Chris for almost two weeks"
- Bad: skip this entirely because "waiting for a reply" sounds episodic
  (loses a real, durable relationship)
- Good: relationship: correspondent_chris = "regular letter correspondent"
  (the durable fact — a standing correspondence — survives; the specific
  wait, which is genuinely episodic, does not get its own profile entry)

Some categories naturally hold many separate facts (relationship,
technology, preference, possession) — give each distinct fact its own key.
Others are usually singular (identity, health, work, routine) — reuse the
existing key when correcting or updating the same field. Never append a
new, distinct fact onto an existing key's value — that turns a fact into a
story. For example:
- Bad: preference: photography = "prefers manual-focus lenses, warm film
  tones; owns a Nikon FM2 body; owns a 50mm f/1.4 lens; bought a vintage
  enlarger impulsively; likes brand Zeiss" (five distinct facts crammed
  into one key)
- Good: preference: photography_style = "prefers manual-focus lenses,
  warm film tones"; possession: camera_body = "Nikon FM2"; possession:
  camera_lens = "50mm f/1.4"; ... (each fact gets its own key)

You will be shown the current profile — each entry tagged with who
contributed it and how important it is — and new material revealed since
the last review. Decide what changes:
- If new material confirms or corrects an existing fact, reuse that fact's
  exact category and key — this updates it in place instead of creating a
  duplicate (e.g. if a job title changes, update the existing "work: role"
  key, don't add a new key for it).
- Before adding a new key, check whether it actually answers the same
  question as an existing key under different wording — don't only match
  on exact key names. For example, if the profile already has
  preference: ai_model = "Flux 1 Dev" and new material reveals the
  person now uses "Z Image Turbo" instead, that's the same underlying
  fact (which model they currently use), not a new one — update the
  existing "ai_model" key (or delete it and add the replacement under
  one key). Don't add a second key like "base_model" that leaves both
  entries standing side by side as if they were unrelated facts.
- This "same question, different wording" check applies just as much
  within the new material itself as it does against the existing
  profile. If two separate mentions in this batch both describe facets
  of the same underlying fact (e.g. "gave up on the linked-notes app"
  and "now just uses one plain text file" are both answering "how does
  this person take notes now"), fold them into a single new key — don't
  split one fact into several just because it arrived in more than one
  sentence.
- Entries marked [user] were explicitly confirmed by the person, not just
  inferred — only change them on a clear, direct contradiction, not merely
  a refinement or a related new detail.
- Only add a new fact if it is genuinely durable and not already covered.
- Only delete a fact if it has been explicitly contradicted or is no longer true.
- Assign each fact an importance: "high" (core identity, standing
  instructions on how to treat this person, anything that would be
  actively wrong to forget), "medium" (most stable facts), or "low"
  (minor, nice-to-have detail). When genuinely unsure between "medium"
  and "low", prefer "low" — the profile should stay small, and only
  facts that are actually load-bearing should compete for "high"/
  "medium". A specific possession, a one-off preference detail, or
  anything you'd only mention if it came up is "low", not "medium".
- Try to keep the total profile near {target_max} facts — if you're well
  over, consolidate overlapping entries or drop the least important
  ("low", then "medium") ones first.
- Do not repeat facts that are unchanged — only output what changed.

Respond with JSON only:
{{"upserts": [{{"category": "...", "key": "...", "value": "...", "importance": "low|medium|high"}}], "deletes": [{{"category": "...", "key": "..."}}]}}
If nothing should change, respond with {{"upserts": [], "deletes": []}}"""


def _build_consolidation_prompt(current_facts: list[Fact], new_items: list[MemoryItem]) -> str:
    facts_block = "\n".join(
        f"- {f.category}: {f.key} = {f.value} [{f.source}, importance={f.importance}]" for f in current_facts
    ) or "(none yet)"
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
    categories : list[str] | None
        Closed category list offered to the LLM — not hard-enforced (a
        fact under an unlisted category is still stored, just logged).
        Defaults to the same list as ``FactsConfig.categories``.
    tracer : Tracer | None
        Debug tracer recording the prompt/response/duration of each
        consolidation pass and otherwise-silent failures. Defaults to a
        no-op tracer.
    metrics : MetricsRecorder | None
        Always-on operational-metrics recorder. Defaults to a no-op
        recorder.
    """

    def __init__(
        self,
        storage: StorageProvider,
        llm: LLMProvider,
        target_max: int = 100,
        categories: list[str] | None = None,
        tracer: Tracer | None = None,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self._storage = storage
        self._llm = llm
        self._target_max = target_max
        self._categories = categories if categories is not None else list(_DEFAULT_CATEGORIES)
        self._tracer = tracer or NullTracer()
        self._metrics = metrics or NullMetricsRecorder()

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

        `peer_id` only scopes *where* the reconciled profile and
        checkpoint are stored — it plays no part in selecting which
        memory items feed the pass. That selection always looks at
        `_PROFILE_SUBJECT` ("user"), matching FragmentaryExtractor's
        fixed subject_peer vocabulary.

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
        new_items = self._storage.get_fact_items_since(_PROFILE_SUBJECT, since_id)
        if not new_items:
            return

        started = time.perf_counter()
        try:
            current_facts = self._storage.get_facts(peer_id)
            prompt = _build_consolidation_prompt(current_facts, new_items)
            response = self._llm.complete(
                prompt=prompt,
                system_prompt=_consolidation_system_prompt(self._target_max, self._categories),
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
                    if category not in self._categories:
                        # Not enforced — a dropped fact is worse than an
                        # off-list category. Logged so drift is visible.
                        logger.debug(
                            "Eidolon fact_consolidation: category %r not in configured list %s",
                            category, self._categories,
                        )
                    self._storage.set_fact(peer_id, category, key, value, source="ai", importance=importance)

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
            self._metrics.record_llm_call(
                "fact_consolidation", (time.perf_counter() - started) * 1000, True,
                peer_id=peer_id, request_id=request_id, session_id=session_id,
            )
        except Exception as exc:
            self._tracer.record(
                "error",
                {"step": "fact_consolidation", "peer_id": peer_id, "exception": repr(exc)},
                request_id=request_id,
                session_id=session_id,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
            self._metrics.record_event(
                "error", "fact_consolidation", repr(exc),
                peer_id=peer_id, request_id=request_id, session_id=session_id,
            )
            logger.warning("Eidolon fact_consolidation failed (peer_id=%s): %s", peer_id, exc)
            # Checkpoint intentionally not advanced — these items are
            # retried at the next consolidation trigger.
