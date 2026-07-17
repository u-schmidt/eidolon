"""Eidolon Hermes Memory Provider Plugin.

Full Mode: Replaces Honcho as the active memory provider.
Provides topic-based, associative memory with perspective attribution.

Directory structure (inside Hermes plugins/):
    plugins/eidolon/
    ├── __init__.py      # MemoryProvider implementation + register() entry point
    ├── plugin.yaml      # Metadata
    └── README.md        # Setup instructions
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the plugin root is on sys.path so core/, providers/ etc. are importable
_plugin_root = str(Path(__file__).resolve().parent)
if _plugin_root not in sys.path:
    sys.path.insert(0, _plugin_root)

# Hermes plugin system imports — resolved at runtime
try:
    from agent.memory_provider import MemoryProvider
except ImportError:
    MemoryProvider = object

# Eidolon core
from core import Eidolon
from core.config import EidolonConfig
from core.config_loader import ConfigLoader
from core.models import EnvironmentSnapshot
from core.provider_factory import ProviderFactory

logger = logging.getLogger(__name__)


class EidolonMemoryProvider(MemoryProvider):
    """Eidolon memory provider for Hermes Agent.

    Implements the MemoryProvider interface to provide topic-based,
    associative memory with perspective attribution.
    """

    def __init__(self) -> None:
        self._eidolon: Eidolon | None = None
        self._store = None
        self._active_topic_id: int | None = None
        self._hermes_home: str = ""
        self._initialized = False

    @property
    def name(self) -> str:
        return "eidolon"

    def is_available(self) -> bool:
        """Check if eidolon.yaml exists in the profile directory.

        No network calls — just checks for the config file.
        """
        try:
            from hermes_constants import get_hermes_home
            config_path = Path(get_hermes_home()) / "eidolon.yaml"
            return config_path.exists()
        except Exception:
            return False

    # ── Core Lifecycle ───────────────────────────────────

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize the Eidolon core and connect to storage.

        Called once when Hermes loads the plugin.
        Loads configuration from eidolon.yaml in the profile directory.
        """
        hermes_home = kwargs.get("hermes_home", "")
        self._hermes_home = hermes_home

        # Resolve config path — look for eidolon.yaml in HERMES_HOME
        config_path = Path(hermes_home) / "eidolon.yaml"

        if not config_path.exists():
            raise RuntimeError(
                f"Eidolon config not found at {config_path}. "
                f"Copy eidolon.example.yaml to {config_path} and adjust."
            )

        # Load configuration
        loader = ConfigLoader(config_path=str(config_path))
        config: EidolonConfig = loader.load()

        # Create providers via factory
        factory = ProviderFactory(config)
        store = factory.create_storage()
        llm = factory.create_llm()
        embedder = factory.create_embedder()

        # Store reference for shutdown
        self._store = store

        # Eidolon core
        self._eidolon = Eidolon(
            storage=store,
            llm=llm,
            embedder=embedder,
            language=config.language,
            retrieval_weights={
                "semantic": config.retrieval.semantic,
                "tag_overlap": config.retrieval.tag_overlap,
                "person": config.retrieval.person,
                "temporal": config.retrieval.temporal,
            },
        )

        self._initialized = True

    def shutdown(self) -> None:
        """Clean up resources when the plugin is unloaded."""
        if self._store is not None:
            self._store.close()
        self._eidolon = None
        self._store = None
        self._initialized = False

    # ── Prefetch (called SYNC before each LLM call) ──────

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context for the upcoming turn.

        Called before each API call. Performs topic detection and
        context retrieval, returning formatted text for prompt injection.

        Returns empty string if nothing relevant is found.
        """
        if not self._eidolon or not self._initialized:
            return ""

        try:
            # Detect topic
            topic_change = self._eidolon.detect_topic(
                message=query,
                active_topic_id=self._active_topic_id,
            )

            # Retrieve context
            is_new = topic_change.type == "NEW_TOPIC"
            bundle = self._eidolon.retrieve_context(
                message=query,
                peer_id="user",
                environment=EnvironmentSnapshot(situation="conversation"),
                topic_label=topic_change.label,
                is_new_topic=is_new,
            )

            # Build formatted context string
            parts: list[str] = []

            if bundle.current_topic_label:
                parts.append(f"[Current topic: {bundle.current_topic_label}]")

            if bundle.facts:
                fact_lines = []
                for f in bundle.facts:
                    fact_lines.append(f"- {f.category}: {f.key} = {f.value}")
                parts.append("[Known facts]\n" + "\n".join(fact_lines))

            if bundle.related_topics:
                topic_lines = []
                for rt in bundle.related_topics:
                    tags = ", ".join(rt.tags) if rt.tags else ""
                    tag_str = f" ({tags})" if tags else ""
                    topic_lines.append(f"- {rt.summary}{tag_str}")
                parts.append("[Related topics]\n" + "\n".join(topic_lines))

            if bundle.memory_items:
                item_lines = []
                for mi in bundle.memory_items:
                    item_lines.append(
                        f"- [{mi.item_type}] ({mi.source_peer} about {mi.subject_peer}): {mi.content}"
                    )
                parts.append("[Memory fragments]\n" + "\n".join(item_lines))

            return "\n\n".join(parts)

        except Exception as e:
            logger.debug("Eidolon prefetch failed (non-fatal): %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background recall for the NEXT turn.

        Eidolon's prefetch is lightweight (topic detection + retrieval),
        so there's no need for a separate background queue. The actual
        work happens inline in prefetch() on the next turn.
        """
        pass

    # ── Sync Turn (called ASYNC after each LLM response) ─

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist a completed turn to Eidolon's memory.

        Called after each turn. Performs fragmentary extraction,
        fact extraction, and topic management.
        """
        if not self._eidolon or not self._initialized:
            return

        try:
            # Detect topic to determine if we need to close/create
            topic_change = self._eidolon.detect_topic(
                message=user_content,
                active_topic_id=self._active_topic_id,
            )

            is_new = topic_change.type == "NEW_TOPIC"
            old_topic_id = self._active_topic_id
            new_topic_label = topic_change.label if is_new else None

            # Store the turn — returns new topic ID if created
            new_topic_id = self._eidolon.store_turn(
                user_message=user_content,
                ai_response=assistant_content,
                peer_id="user",
                environment=EnvironmentSnapshot(situation="conversation"),
                old_topic_id=old_topic_id,
                new_topic_label=new_topic_label,
            )

            # Update active topic
            if new_topic_id is not None:
                self._active_topic_id = new_topic_id

        except Exception as e:
            logger.warning("Eidolon sync_turn failed: %s", e)

    # ── Tools ────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
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
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call for one of this provider's tools."""
        if tool_name == "eidolon_search":
            return self._handle_search(args)
        raise NotImplementedError(
            f"Eidolon provider does not handle tool {tool_name}"
        )

    def _handle_search(self, args: Dict[str, Any]) -> str:
        """Handle eidolon_search tool call."""
        if not self._eidolon or not self._initialized:
            return '{"error": "Eidolon not initialized"}'

        query = args.get("query", "")
        if not query:
            return '{"error": "query is required"}'

        try:
            bundle = self._eidolon.retrieve_context(
                message=query,
                peer_id="user",
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

            import json
            return json.dumps(results, ensure_ascii=False)

        except Exception as e:
            return f'{{"error": "Search failed: {e}"}}'

    # ── Session Lifecycle Hooks ─────────────────────────

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Called when a session ends.

        Ensures the current topic is properly closed.
        """
        if not self._eidolon or not self._initialized:
            return

        # The active topic will be closed by store_turn() when a new
        # topic is created. If the session ends without a new topic,
        # the current topic remains open — that's fine, it will be
        # resumed or closed on the next session.
        pass

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Called when the agent switches session_id.

        Reset active topic on /reset or /new, keep it on /resume.
        """
        if reset:
            self._active_topic_id = None


# ── Plugin Entry Point ──────────────────────────────────

def register(ctx) -> None:
    """Register the Eidolon memory provider with Hermes.

    Called by the Hermes plugin system when the plugin is loaded.
    """
    ctx.register_memory_provider(EidolonMemoryProvider())
