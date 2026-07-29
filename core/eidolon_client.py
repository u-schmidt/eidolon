"""The contract a caller uses to talk to Eidolon's core facade.

Unlike the Protocols in ``core/interfaces.py`` (things core depends on
and has injected into it — storage, LLM, embedder, tracer), this is the
reverse direction: it's core's own public facade, depended upon by
callers. Deliberately scoped to what a conversational adapter needs
around one turn (before/after the LLM responds) — not a general-purpose
core API. See ``core/__init__.py``'s ``Eidolon`` class, which this
mirrors exactly and satisfies structurally without any inheritance.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from core.models import ContextBundle, EnvironmentSnapshot, PrefetchResult, TopicChange


class EidolonClient(Protocol):
    """Abstract entry point into Eidolon's core facade.

    V1: ``Eidolon`` (in-process, ``core/__init__.py``) or
    ``client.http_client.EidolonHttpClient`` (over HTTP, for the
    core-as-a-service deployment mode).
    """

    def get_active_topic(self, workspace_id: str, session_id: str | None = None) -> int | None:
        """Look up the currently open topic for this workspace and session, if any."""
        ...

    def prepare_context(
        self,
        message: str,
        workspace_id: str,
        environment: EnvironmentSnapshot,
        active_topic_id: int | None = None,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> PrefetchResult:
        """Detect the topic change and retrieve context for it, in one call."""
        ...

    def detect_topic(
        self,
        message: str,
        active_topic_id: int | None = None,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> TopicChange:
        """Detect whether a message continues or changes the topic."""
        ...

    def retrieve_context(
        self,
        message: str,
        workspace_id: str,
        environment: EnvironmentSnapshot,
        topic_label: str | None = None,
        is_new_topic: bool = False,
        active_topic_id: int | None = None,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> ContextBundle:
        """Retrieve relevant context for a message."""
        ...

    def store_turn(
        self,
        user_message: str,
        ai_response: str,
        workspace_id: str,
        environment: EnvironmentSnapshot,
        turn_id: int | None = None,
        old_topic_id: int | None = None,
        new_topic_label: str | None = None,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> int | None:
        """Store a conversation turn after the LLM has responded."""
        ...

    def consolidate_facts(
        self,
        workspace_id: str,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Reconcile a workspace's durable-facts profile now."""
        ...

    def close_stale_topics(
        self,
        cutoff: datetime,
        workspace_id: str | None = None,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> list[int]:
        """Gracefully close every 'active' topic with no activity since `cutoff`."""
        ...

    def correct_fact(
        self,
        workspace_id: str,
        category: str,
        key: str,
        value: str | None = None,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Directly upsert or delete one fact, bypassing consolidation."""
        ...

    def inspect_cluster(self, cluster_id: int) -> list[dict]:
        """Return every topic assigned to `cluster_id`, oldest first."""
        ...

    def inspect_topic(self, topic_id: int) -> list[dict]:
        """Return every memory item extracted for `topic_id`, oldest first."""
        ...
