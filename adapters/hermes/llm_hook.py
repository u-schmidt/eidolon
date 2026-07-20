"""Eidolon pre_llm_call hook for Topic-Only Mode.

Runs alongside Honcho — does not replace it.
Only provides topic summaries and associations, not facts or environment.

This is a lighter integration that complements Honcho's fact management
with Eidolon's topic detection and associative retrieval.
"""

from __future__ import annotations

from typing import Any

from core import Eidolon
from core.models import EnvironmentSnapshot
from core.stores.sqlite_store import SqliteStore
from core.providers.lm_studio import LMStudio


class EidolonTopicHook:
    """Eidolon pre_llm_call hook for Topic-Only Mode.

    Injects topic summaries and associations into the prompt
    without replacing Honcho's fact management.

    Parameters
    ----------
    config : dict
        Hook configuration.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._eidolon: Eidolon | None = None
        self._store: SqliteStore | None = None
        self._active_topic_id: int | None = None

    def initialize(self) -> None:
        """Initialize the Eidolon core for topic-only operation."""
        db_path = self._config.get("db_path", "~/.eidolon/memory.db")
        language = self._config.get("language", "en")

        self._store = SqliteStore(db_path)
        self._store.connect()

        # LLM — inline wrapper: Hermes completion → LLMProvider protocol
        hermes_complete = self._get_hermes_completion()

        class _HookLLM:
            def complete(self, prompt, system_prompt=None, max_tokens=100):
                return hermes_complete(prompt, system=system_prompt, max_tokens=max_tokens)

        llm = _HookLLM()

        embedder = LMStudio(
            base_url=self._config.get("embedder_base_url", "http://localhost:1234"),
            embedder_model=self._config.get("embedder_model", "text-embedding-bge-m3"),
        )

        weights = self._config.get("retrieval_weights")

        self._eidolon = Eidolon(
            storage=self._store,
            llm=llm,
            embedder=embedder,
            language=language,
            retrieval_weights=weights,
        )

    def shutdown(self) -> None:
        if self._store is not None:
            self._store.close()

    def pre_llm_call(
        self,
        user_message: str,
        session_id: str,
        peer_id: str,
        environment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Called before each LLM call. Injects topic context only.

        Parameters
        ----------
        user_message : str
            The user's message.
        session_id : str
            Current Hermes session ID.
        peer_id : str
            Who is speaking.
        environment : dict | None
            Optional environment context.

        Returns
        -------
        dict
            Topic context for prompt injection (does NOT include facts).
        """
        if self._eidolon is None:
            return {}

        env = self._build_environment(environment)

        # Only retrieve context — no fact management
        topic_change = self._eidolon.detect_topic(user_message, self._active_topic_id)
        is_new = topic_change.type == "NEW_TOPIC"

        bundle = self._eidolon.retrieve_context(
            message=user_message,
            peer_id=peer_id,
            environment=env,
            topic_label=topic_change.label,
            is_new_topic=is_new,
            active_topic_id=self._active_topic_id,
        )

        # Store for post-processing
        self._last_topic_label = bundle.current_topic_label
        self._last_topic_change = topic_change

        # Return only topic-related context
        return {
            "eidolon_topics": {
                "current_topic": bundle.current_topic_label,
                "related_topics": [
                    {
                        "id": rt.topic_id,
                        "summary": rt.summary,
                        "tags": rt.tags,
                    }
                    for rt in bundle.related_topics
                ],
                "memory_items": [
                    {
                        "content": mi.content,
                        "type": mi.item_type,
                    }
                    for mi in bundle.memory_items
                ],
            }
        }

    def post_llm_call(
        self,
        user_message: str,
        ai_response: str,
        session_id: str,
        peer_id: str,
        environment: dict[str, Any] | None = None,
    ) -> None:
        """Called after each LLM call. Stores topic data."""
        if self._eidolon is None:
            return

        env = self._build_environment(environment)

        old_topic_id = None
        new_topic_label = None

        if hasattr(self, "_last_topic_label") and self._last_topic_label:
            if hasattr(self, "_last_topic_change") and self._last_topic_change.type == "NEW_TOPIC":
                old_topic_id = self._active_topic_id
                new_topic_label = self._last_topic_label

        self._eidolon.store_turn(
            user_message=user_message,
            ai_response=ai_response,
            peer_id=peer_id,
            environment=env,
            old_topic_id=old_topic_id,
            new_topic_label=new_topic_label,
        )

        if new_topic_label:
            self._active_topic_id = None

    def _build_environment(self, env: dict[str, Any] | None) -> EnvironmentSnapshot:
        if env is None:
            return EnvironmentSnapshot()
        return EnvironmentSnapshot(
            location=env.get("location"),
            time=env.get("time"),
            day_of_week=env.get("day_of_week"),
            situation=env.get("situation"),
            rules=env.get("rules", []),
        )

    def _get_hermes_completion(self):
        try:
            from hermes.llm import complete
            return complete
        except ImportError:
            def _mock_complete(prompt, system=None, max_tokens=100):
                return f"[Mock: {prompt[:50]}]"
            return _mock_complete
