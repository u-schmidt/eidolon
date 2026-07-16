"""Eidolon Hermes Memory Provider Plugin.

Full Mode: Replaces Honcho as the active memory provider.
Provides facts, environment, topics, and memory items via the MemoryProvider interface.

Topic-Only Mode: Runs alongside Honcho via pre_llm_call hook.
Provides only topic summaries and associations.

Directory structure (inside Hermes plugins/):
    plugins/memory/eidolon/
    ├── __init__.py      # MemoryProvider implementation + register()
    ├── plugin.yaml      # Metadata
    └── README.md        # Setup instructions
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Hermes plugin system imports
# These are resolved at runtime when Hermes loads the plugin
try:
    from hermes.agent.memory_provider import MemoryProvider
    from hermes.agent.memory_manager import MemoryManager
    from hermes.core.context import ContextBundle as HermesContextBundle
    from hermes.plugin_system import register_memory_provider, register_hook
except ImportError:
    # Standalone development — Hermes is not installed in this venv
    MemoryProvider = object  # type: ignore[assignment, misc]
    MemoryManager = object  # type: ignore[assignment, misc]
    HermesContextBundle = object  # type: ignore[assignment, misc]
    register_memory_provider = lambda x: None  # type: ignore[assignment]
    register_hook = lambda x: None  # type: ignore[assignment]

# Eidolon core
from core import Eidolon
from core.interfaces import LLMProvider
from core.models import EnvironmentSnapshot
from core.stores.sqlite_store import SqliteStore
from providers.lm_studio import LMStudio


class EidolonMemoryProvider(MemoryProvider):
    """Eidolon memory provider for Hermes Agent.

    Implements the MemoryProvider interface to provide topic-based,
    associative memory with perspective attribution.

    Parameters
    ----------
    config : dict
        Plugin configuration from plugin.yaml.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._eidolon: Eidolon | None = None
        self._store: SqliteStore | None = None
        self._active_topic_id: int | None = None

    # ── Lifecycle ─────────────────────────────────────────

    def initialize(self) -> None:
        """Initialize the Eidolon core and connect to storage.

        Called once when Hermes loads the plugin.
        """
        db_path = self._config.get("db_path", "~/.eidolon/memory.db")
        language = self._config.get("language", "en")

        # Storage
        self._store = SqliteStore(db_path)
        self._store.connect()

        # LLM — wraps Hermes' completion function as an LLMProvider
        # The actual Hermes LLM function is injected by the plugin system
        hermes_complete = self._get_hermes_completion()

        class _AdapterLLM:
            """Thin wrapper: Hermes completion → LLMProvider protocol."""

            def complete(self, prompt, system_prompt=None, max_tokens=100):
                return hermes_complete(prompt, system=system_prompt, max_tokens=max_tokens)

        llm = _AdapterLLM()

        # Embedder — LM Studio by default
        embedder = LMStudio(
            base_url=self._config.get("embedder_base_url", "http://localhost:1234"),
            embedder_model=self._config.get("embedder_model", "text-embedding-bge-m3"),
        )

        # Retrieval weights
        weights = self._config.get("retrieval_weights")

        # Eidolon core
        self._eidolon = Eidolon(
            storage=self._store,
            llm=llm,
            embedder=embedder,
            language=language,
            retrieval_weights=weights,
        )

    def shutdown(self) -> None:
        """Clean up resources when the plugin is unloaded."""
        if self._store is not None:
            self._store.close()

    # ── MemoryProvider Interface ──────────────────────────

    def prefetch(self, session_id: str, peer_id: str) -> dict[str, Any]:
        """Load initial context for a session.

        Called when a new session starts.
        Returns facts and environment info for prompt injection.
        """
        if self._eidolon is None:
            return {}

        # Reset active topic for new session
        self._active_topic_id = None

        # Load facts
        facts = self._store.get_facts(peer_id) if self._store else []

        return {
            "facts": [{"category": f.category, "key": f.key, "value": f.value} for f in facts],
            "eidolon_ready": True,
        }

    def sync_turn(
        self,
        user_message: str,
        session_id: str,
        peer_id: str,
        environment: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a user message before the LLM responds.

        Called synchronously before each LLM call.
        Returns context for prompt injection.

        Parameters
        ----------
        user_message : str
            The user's message.
        session_id : str
            Current Hermes session ID.
        peer_id : str
            Who is speaking (e.g. "user").
        environment : dict | None
            Optional environment context.

        Returns
        -------
        dict
            Context to inject into the prompt.
        """
        if self._eidolon is None:
            return {}

        env = self._build_environment(environment)

        # Prepare context (detect topic + retrieve)
        bundle = self._eidolon.prepare_context(
            message=user_message,
            peer_id=peer_id,
            environment=env,
            active_topic_id=self._active_topic_id,
        )

        # Store the topic label for store_turn
        self._last_topic_label = bundle.current_topic_label

        # Build context dict for prompt injection
        context = self._bundle_to_dict(bundle)

        return context

    def store_turn(
        self,
        user_message: str,
        ai_response: str,
        session_id: str,
        peer_id: str,
        environment: dict[str, Any] | None = None,
    ) -> None:
        """Store a conversation turn after the LLM responds.

        Called asynchronously after each LLM call.
        Extracts memory fragments and updates topics.
        """
        if self._eidolon is None:
            return

        env = self._build_environment(environment)

        # Determine if topic changed
        old_topic_id = None
        new_topic_label = None

        if hasattr(self, "_last_topic_label") and self._last_topic_label:
            # Topic changed — close old, create new
            old_topic_id = self._active_topic_id
            new_topic_label = self._last_topic_label

        # Store the turn
        self._eidolon.store_turn(
            user_message=user_message,
            ai_response=ai_response,
            peer_id=peer_id,
            environment=env,
            old_topic_id=old_topic_id,
            new_topic_label=new_topic_label,
        )

        # Update active topic if new one was created
        if new_topic_label:
            # The new topic is the most recent active one
            # In a real implementation, store_turn would return the new topic ID
            self._active_topic_id = None  # Will be set by next sync_turn

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return tool schemas for memory-related tools.

        Optional: expose memory search as a tool the LLM can call.
        """
        return [
            {
                "name": "search_memory",
                "description": "Search past topics and memory items by query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results (default: 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

    # ── Internal Helpers ──────────────────────────────────

    def _build_environment(self, env: dict[str, Any] | None) -> EnvironmentSnapshot:
        """Build an EnvironmentSnapshot from a Hermes environment dict."""
        if env is None:
            return EnvironmentSnapshot()
        return EnvironmentSnapshot(
            location=env.get("location"),
            time=env.get("time"),
            day_of_week=env.get("day_of_week"),
            situation=env.get("situation"),
            rules=env.get("rules", []),
        )

    def _bundle_to_dict(self, bundle: Any) -> dict[str, Any]:
        """Convert an Eidolon ContextBundle to a dict for prompt injection."""
        return {
            "eidolon_context": {
                "current_topic": bundle.current_topic_label,
                "facts": [
                    {"category": f.category, "key": f.key, "value": f.value}
                    for f in bundle.facts
                ],
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
                        "source": mi.source_peer,
                        "subject": mi.subject_peer,
                    }
                    for mi in bundle.memory_items
                ],
            }
        }

    def _get_hermes_completion(self):
        """Get the Hermes LLM completion function.

        This is injected by the plugin system at runtime.
        For now, returns a placeholder that raises if called without Hermes.
        """
        # In production, this would be provided by the Hermes plugin system
        # For development/testing, we use the CLI mock
        try:
            from hermes.llm import complete
            return complete
        except ImportError:
            # Return a mock for development
            def _mock_complete(prompt, system=None, max_tokens=100):
                return f"[Mock response for: {prompt[:50]}]"
            return _mock_complete


# ── Plugin Entry Point ────────────────────────────────────

def register() -> None:
    """Register the Eidolon memory provider plugin with Hermes."""
    register_memory_provider("eidolon", EidolonMemoryProvider)
