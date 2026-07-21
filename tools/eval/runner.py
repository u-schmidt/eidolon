"""Drives an EvalScript through the real Eidolon core, turn by turn."""

from __future__ import annotations

from core import Eidolon
from core.interfaces import StorageProvider
from core.models import EnvironmentSnapshot
from tools.eval.models import EvalScript
from tools.eval.results import EvalReport
from tools.eval.scorer import EvalScorer


class EvalRunner:
    """Plays an ``EvalScript`` against a real ``Eidolon`` instance.

    Mirrors what an adapter does turn to turn — tracks the running active
    topic itself from each ``detect_topic`` result, exactly like
    ``adapters/hermes`` does, since the script only describes messages,
    not adapter-internal bookkeeping.
    """

    def __init__(self, eidolon: Eidolon, storage: StorageProvider, scorer: EvalScorer) -> None:
        self._eidolon = eidolon
        self._storage = storage
        self._scorer = scorer

    def run(
        self,
        script: EvalScript,
        *,
        model_identity: str,
        embedder_identity: str,
        db_path: str,
    ) -> EvalReport:
        active_topic_id: int | None = None
        environment = EnvironmentSnapshot()
        turn_results = []

        for turn_number, turn in enumerate(script.turns, start=1):
            topic_change = self._eidolon.detect_topic(turn.user, active_topic_id)

            # old_topic_id is always the currently active topic, not just
            # on a change — store_turn needs it to attach a CONTINUE turn's
            # messages to the right (still-open) topic. Matches
            # adapters/hermes/__init__.py's sync_turn exactly.
            old_topic_id = active_topic_id
            new_topic_label = topic_change.label if topic_change.type == "NEW_TOPIC" else None

            new_topic_id = self._eidolon.store_turn(
                user_message=turn.user,
                ai_response=turn.ai,
                peer_id=script.peer_id,
                environment=environment,
                turn_id=turn_number,
                old_topic_id=old_topic_id,
                new_topic_label=new_topic_label,
            )
            if new_topic_id is not None:
                active_topic_id = new_topic_id

            actual_items = self._storage.get_memory_items_by_turn(turn_number)
            turn_results.append(
                self._scorer.score_turn(turn_number, turn.user, turn.expect, topic_change, actual_items)
            )

        end_result = self._scorer.score_end(
            script.expect_end,
            self._storage.get_facts(script.peer_id),
            self._storage.get_recent_topics(script.peer_id, limit=len(script.turns) or 1),
            self._storage.get_clusters(script.peer_id),
        )

        return EvalReport(
            script_name=script.name,
            peer_id=script.peer_id,
            model_identity=model_identity,
            embedder_identity=embedder_identity,
            db_path=db_path,
            turn_results=turn_results,
            end_result=end_result,
        )
