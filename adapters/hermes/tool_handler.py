"""Implements the four eidolon_* Hermes tool calls — the schema/dispatch/
handler logic factored out of EidolonMemoryProvider.

Constructed fresh per call from EidolonMemoryProvider (cheap — two
attribute references, no setup) rather than once in initialize(), so it
has no dependency on the provider's own initialization lifecycle and
needs no changes to how tests construct a provider.
"""

from __future__ import annotations

import json
from typing import Any

from core.eidolon_client import EidolonClient
from core.models import EnvironmentSnapshot


class EidolonToolHandler:
    """Handles eidolon_search / eidolon_inspect_cluster /
    eidolon_inspect_topic / eidolon_correct_fact tool calls.

    ``eidolon``/``initialized`` mirror EidolonMemoryProvider's own
    ``_eidolon``/``_initialized`` at the moment of the call — each handler
    checks them itself (rather than a single guard in handle_tool_call)
    so an unknown tool name still raises NotImplementedError regardless
    of initialization state, matching the pre-extraction behavior.
    """

    def __init__(self, eidolon: EidolonClient | None, initialized: bool, peer_id: str) -> None:
        self._eidolon = eidolon
        self._initialized = initialized
        self._peer_id = peer_id

    @staticmethod
    def get_tool_schemas() -> list[dict[str, Any]]:
        """Return tool schemas for memory-related tools.

        These tools are exposed to the LLM for explicit memory operations.
        """
        return [
            {
                "name": "eidolon_search",
                "description": (
                    "Search the Eidolon topic memory system for related topics "
                    "and memory items. Use this to explicitly recall past "
                    "conversation topics and extracted memory fragments."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for associative retrieval",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "eidolon_inspect_cluster",
                "description": (
                    "Look inside a recurring-topic cluster shown in memory context "
                    "(e.g. 'Cluster #17') to see the full list of individual past "
                    "topics it consolidates — each with its own summary and date. "
                    "Use this when the consolidated summary isn't enough detail."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cluster_id": {
                            "type": "integer",
                            "description": "The numeric cluster id shown in memory context, e.g. 17 for 'Cluster #17'.",
                        },
                    },
                    "required": ["cluster_id"],
                },
            },
            {
                "name": "eidolon_inspect_topic",
                "description": (
                    "Look inside a specific past topic (e.g. topic_id 35 from an "
                    "eidolon_search result) to see every individual memory fragment "
                    "extracted from it, not just the compact injected summary. Use "
                    "this when the summary shown in memory context isn't detailed "
                    "enough and you need the specific facts underneath."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic_id": {
                            "type": "integer",
                            "description": "The numeric topic id, e.g. from an eidolon_search result's related_topics.",
                        },
                    },
                    "required": ["topic_id"],
                },
            },
            {
                "name": "eidolon_correct_fact",
                "description": (
                    "Correct or remove one fact in the user's durable profile "
                    "immediately. Use this when the user explicitly says a "
                    "remembered fact is wrong, outdated, or should be forgotten. "
                    "Note: a later automatic consolidation pass may still revise "
                    "this fact based on other conversation evidence."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Fact category, e.g. 'location', 'work'.",
                        },
                        "key": {
                            "type": "string",
                            "description": "Fact key within the category, e.g. 'city'.",
                        },
                        "value": {
                            "type": "string",
                            "description": "New value. Omit to delete this fact entirely.",
                        },
                    },
                    "required": ["category", "key"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any]) -> str:
        """Handle a tool call for one of this provider's tools."""
        if tool_name == "eidolon_search":
            return self._handle_search(args)
        if tool_name == "eidolon_inspect_cluster":
            return self._handle_inspect_cluster(args)
        if tool_name == "eidolon_inspect_topic":
            return self._handle_inspect_topic(args)
        if tool_name == "eidolon_correct_fact":
            return self._handle_correct_fact(args)
        raise NotImplementedError(
            f"Eidolon provider does not handle tool {tool_name}"
        )

    def _handle_search(self, args: dict[str, Any]) -> str:
        """Handle eidolon_search tool call."""
        if not self._eidolon or not self._initialized:
            return '{"error": "Eidolon not initialized"}'

        query = args.get("query", "")
        if not query:
            return '{"error": "query is required"}'

        try:
            bundle = self._eidolon.retrieve_context(
                message=query,
                peer_id=self._peer_id,
                environment=EnvironmentSnapshot(situation="search"),
                topic_label=None,
                is_new_topic=True,
            )

            results = {
                "facts": [
                    {"category": f.category, "key": f.key, "value": f.value}
                    for f in (bundle.facts or [])
                ],
                "related_topics": [
                    {
                        "topic_id": rt.topic_id,
                        "summary": rt.summary,
                        "tags": rt.tags,
                    }
                    for rt in (bundle.related_topics or [])
                ],
                "memory_items": [
                    {
                        "content": mi.content,
                        "item_type": mi.item_type,
                        "source_peer": mi.source_peer,
                        "subject_peer": mi.subject_peer,
                    }
                    for mi in (bundle.memory_items or [])
                ],
            }

            return json.dumps(results, ensure_ascii=False)

        except Exception as e:
            return f'{{"error": "Search failed: {e}"}}'

    def _handle_inspect_cluster(self, args: dict[str, Any]) -> str:
        """Handle eidolon_inspect_cluster tool call."""
        if not self._eidolon or not self._initialized:
            return json.dumps({"error": "Eidolon not initialized"})

        cluster_id = args.get("cluster_id")
        if not isinstance(cluster_id, int):
            return json.dumps({"error": "cluster_id (integer) is required"})

        try:
            topics = self._eidolon.inspect_cluster(cluster_id)
            return json.dumps(
                {
                    "cluster_id": cluster_id,
                    "topics": [
                        {
                            "label": t.get("label"),
                            "summary": t.get("summary"),
                            "closed_at": t.get("closed_at"),
                        }
                        for t in topics
                    ],
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": f"Inspect failed: {e}"})

    def _handle_inspect_topic(self, args: dict[str, Any]) -> str:
        """Handle eidolon_inspect_topic tool call."""
        if not self._eidolon or not self._initialized:
            return json.dumps({"error": "Eidolon not initialized"})

        topic_id = args.get("topic_id")
        if not isinstance(topic_id, int):
            return json.dumps({"error": "topic_id (integer) is required"})

        try:
            items = self._eidolon.inspect_topic(topic_id)
            return json.dumps(
                {
                    "topic_id": topic_id,
                    "memory_items": [
                        {
                            "content": i.get("content"),
                            "item_type": i.get("item_type"),
                            "source_peer": i.get("source_peer"),
                            "subject_peer": i.get("subject_peer"),
                        }
                        for i in items
                    ],
                },
                ensure_ascii=False,
            )
        except Exception as e:
            return json.dumps({"error": f"Inspect failed: {e}"})

    def _handle_correct_fact(self, args: dict[str, Any]) -> str:
        """Handle eidolon_correct_fact tool call."""
        if not self._eidolon or not self._initialized:
            return json.dumps({"error": "Eidolon not initialized"})

        category = args.get("category", "").strip()
        key = args.get("key", "").strip()
        if not category or not key:
            return json.dumps({"error": "category and key are required"})

        value = args.get("value")
        try:
            self._eidolon.correct_fact(self._peer_id, category, key, value)
            return json.dumps({
                "status": "ok",
                "action": "deleted" if not value else "updated",
                "category": category,
                "key": key,
            })
        except Exception as e:
            return json.dumps({"error": f"Fact correction failed: {e}"})
