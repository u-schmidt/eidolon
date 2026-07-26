"""EidolonMemoryProvider — the Hermes MemoryProvider implementation.

See adapters/hermes/__init__.py for the plugin's register() entry point.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Hermes plugin system imports — resolved at runtime
try:
    from agent.memory_provider import MemoryProvider
except ImportError:
    MemoryProvider = object

# Eidolon core
from adapters.hermes.session_state import SessionState
from client.http_client import EidolonHttpClient
from core import Eidolon
from core.config import EidolonConfig, MaintenanceConfig
from core.config_loader import ConfigLoader
from core.context_formatter import ContextFormatter
from core.eidolon_client import EidolonClient
from core.interfaces import StorageProvider, Tracer
from core.models import EnvironmentSnapshot
from core.provider_factory import ProviderFactory

logger = logging.getLogger(__name__)


class EidolonMemoryProvider(MemoryProvider):
    """Eidolon memory provider for Hermes Agent.

    Implements the MemoryProvider interface to provide topic-based,
    associative memory with perspective attribution.
    """

    def __init__(self) -> None:
        self._eidolon: EidolonClient | None = None
        self._store: StorageProvider | None = None
        self._tracer: Tracer | None = None
        # Keyed by Hermes session_id. Hermes gives this instance its own
        # per-session lifecycle, but a single instance can still see more
        # than one session_id (resume/branch/compression handoffs) — see
        # session_state.py.
        self._sessions: dict[str, SessionState] = {}
        # Last known real (non-empty) session_id — Hermes' prefetch_all()
        # call site does not currently forward session_id at all (always
        # ""), while sync_all() does. Falling back to this lets prefetch()
        # land in the same session's state that sync_turn() will look up.
        self._current_session_id: str = ""
        self._hermes_home: str = ""
        # Resolved once in initialize() from Hermes' user_id/user_id_alt
        # kwargs — never supplied again afterward (confirmed: no other
        # MemoryProvider method receives them), and one instance already
        # belongs to one person's session for its whole lifetime. Falls
        # back to the literal "user" for CLI-attached sessions, which
        # Hermes never gives an identity for.
        self._peer_id: str = "user"
        # IANA zone name (e.g. "Europe/Berlin") for displaying the injected
        # [Environment] time — resolved once in initialize() from config.
        self._timezone: str = "UTC"
        # Resolved once in initialize() from config. Disabled by default —
        # see MaintenanceConfig's docstring for why. _last_maintenance_check
        # is None until the first sweep attempt (gated by _maybe_run_
        # maintenance(), called at the end of sync_turn()).
        self._maintenance: MaintenanceConfig = MaintenanceConfig()
        self._last_maintenance_check: datetime | None = None
        self._initialized = False

    def _resolve_session_id(self, session_id: str) -> str:
        """Resolve a possibly-empty session_id to the best-known real one.

        Remembers any non-empty session_id it sees (from any call site) as
        the current session, and returns that as the fallback when given
        an empty one.
        """
        if session_id:
            self._current_session_id = session_id
            return session_id
        return self._current_session_id

    def _get_or_create_session(self, session_id: str) -> SessionState:
        """Get this session's state, creating and resuming it if new.

        `topics.status` in storage is the source of truth for what's
        open, scoped per (peer_id, session_id) — not just once at plugin
        load, but for every session_id seen for the first time, so a
        peer's concurrent sessions (web, Telegram, CLI, ...) each resume
        their own topic thread instead of starting blind and silently
        orphaning whatever another session still has open.
        """
        resolved = self._resolve_session_id(session_id)
        if resolved not in self._sessions:
            session = SessionState()
            try:
                if self._eidolon is not None:
                    session.active_topic_id = self._eidolon.get_active_topic(
                        self._peer_id, session_id=resolved
                    )
            except Exception as e:
                logger.debug("Failed to resume active topic for session %s (non-fatal): %s", resolved, e)
            self._sessions[resolved] = session
        return self._sessions[resolved]

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
        # Mirrors Hermes' own convention for picking the stable participant
        # identity (see gateway/session.py, gateway/slash_commands.py):
        # user_id_alt carries a platform-canonical id (e.g. Feishu's
        # union_id) when the platform has one, user_id is the general case.
        # Both are only populated for gateway-routed platforms — a plain
        # CLI-attached session has neither, hence the "user" fallback.
        self._peer_id = kwargs.get("user_id_alt") or kwargs.get("user_id") or "user"

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
        self._timezone = config.timezone
        self._maintenance = config.maintenance

        if config.deployment.mode == "server":
            # Server mode: core runs as its own process (tools/serve.py) —
            # this adapter is just a thin HTTP client, with no storage/
            # llm/embedder/tracer to build locally at all. No tracer is
            # wired here either: there's nothing on this side to attach
            # one to, and building one would just open a trace.db file
            # for no purpose. Deliberate v1 limitation — see ROADMAP.md's
            # "Core-as-a-service deployment mode" entry: a transport-layer
            # failure (timeout, connection refused) is only visible via
            # the broad except/logger.warning already in prefetch()/
            # sync_turn(), not via tools/trace.py.
            self._eidolon = EidolonHttpClient(
                base_url=config.deployment.server_url,
                token=config.deployment.token,
                timeout=config.deployment.timeout,
            )
        else:
            # Embedded mode (default): build storage/llm/embedder and run
            # core in-process. Create providers via factory. Tracer first —
            # a failure building storage/llm/embedder can then still be
            # recorded before it's re-raised, instead of vanishing before
            # anything capable of recording it exists (this is exactly what
            # happened with a real "database is locked" storage failure: it
            # never reached trace.db). SafeTracer never raises itself (see
            # core/trace/safe_tracer.py), so this reordering can't introduce
            # a new failure mode even if trace.db itself is unavailable. The
            # trace.db event is only ever written when debug.enabled,
            # though — logger.exception() is the channel that's visible
            # regardless of that flag (see also the non-fatal failures
            # logged the same way elsewhere in this file).
            factory = ProviderFactory(config)
            tracer = factory.create_tracer()
            self._tracer = tracer
            try:
                store = factory.create_storage()
                llm = factory.create_llm()
                embedder = factory.create_embedder()
            except Exception as exc:
                tracer.record("error", {"step": "initialize", "exception": repr(exc)}, session_id=session_id)
                logger.exception("Eidolon initialize failed")
                raise

            # Metrics after storage: it opens its own connection into
            # tables storage's own connect() just migrated into
            # existence (see core/stores/migrations.py) — unlike tracer
            # above, there's no failure-recording reason to build it
            # first, and it couldn't record anything yet if there were.
            metrics = factory.create_metrics_recorder()

            # Store reference for shutdown
            self._store = store

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
                retrieval_limits={
                    "related_topics_new": config.retrieval.related_topics_new,
                    "related_topics_continue": config.retrieval.related_topics_continue,
                    "memory_items_new": config.retrieval.memory_items_new,
                    "memory_items_continue": config.retrieval.memory_items_continue,
                    "recent_topics": config.retrieval.recent_topics,
                    "cluster_summaries": config.retrieval.cluster_summaries,
                    "cluster_recent_topics": config.retrieval.cluster_recent_topics,
                    "min_topic_messages": config.retrieval.min_topic_messages,
                },
                facts_consolidation_interval=config.facts.consolidation_interval,
                facts_target_max=config.facts.target_max,
                facts_categories=config.facts.categories,
                clustering_enabled=config.clustering.enabled,
                clustering_shortlist_size=config.clustering.shortlist_size,
                relabel_enabled=config.relabel.enabled,
                relabel_message_threshold=config.relabel.message_threshold,
                tracer=tracer,
                metrics=metrics,
            )

        # Seed the current-session fallback before resolving so an empty
        # session_id from a later prefetch() call still lands here.
        # _get_or_create_session() itself resumes this session's active
        # topic from storage — see its docstring.
        self._get_or_create_session(session_id)

        self._initialized = True

    def shutdown(self) -> None:
        """Clean up resources when the plugin is unloaded."""
        if self._store is not None:
            self._store.close()
        if self._tracer is not None:
            self._tracer.close()
        self._eidolon = None
        self._store = None
        self._tracer = None
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
            session = self._get_or_create_session(session_id)
            resolved_session_id = self._resolve_session_id(session_id)
            # One id for this whole turn — shared by both calls below, and
            # reused by the matching sync_turn() call later, so a turn's
            # trace events all land under one request_id instead of three
            # unrelated ones (see SessionState.pending_request_id).
            turn_request_id = uuid.uuid4().hex[:8]
            # Captured once and reused for both the injected "current time"
            # and the relative-age calculations in ContextFormatter, so
            # they can't drift apart within a single turn. Explicitly UTC
            # (not bare datetime.now()) so it's correct regardless of the
            # host's own OS timezone — ContextFormatter converts it to
            # self._timezone only for display.
            turn_now = datetime.now(UTC)

            # Detect topic + retrieve context in one call — the user's
            # message is the only input that ever matters for the topic
            # decision (see sync_turn(), which reuses it instead of
            # recomputing from the AI's reply). Fused into one
            # prepare_context() call (rather than detect_topic() then
            # retrieve_context() separately) so an HTTP-backed
            # EidolonClient only needs one round trip per prefetch — see
            # core.models.PrefetchResult.
            result = self._eidolon.prepare_context(
                message=query,
                peer_id=self._peer_id,
                environment=EnvironmentSnapshot(
                    situation="conversation", time=turn_now, day_of_week=turn_now.strftime("%A"),
                ),
                active_topic_id=session.active_topic_id,
                session_id=resolved_session_id,
                request_id=turn_request_id,
            )
            session.pending_topic_change = result.topic_change
            session.pending_query = query
            session.pending_request_id = turn_request_id

            return ContextFormatter.format(result.bundle, now=turn_now, timezone=self._timezone)

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
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist a completed turn to Eidolon's memory.

        Called after each turn. Performs fragmentary extraction,
        fact extraction, and topic management.
        """
        if not self._eidolon or not self._initialized:
            return

        try:
            session = self._get_or_create_session(session_id)
            resolved_session_id = self._resolve_session_id(session_id)

            # Reuse the topic decision prefetch() already made for this
            # exact message — it's driven entirely by the user's message,
            # which hasn't changed, so recomputing it here would just be
            # the same LLM call twice. Only recompute if there's no cached
            # decision for this message (e.g. sync_turn() was called
            # without a preceding prefetch()) — never trust a cached
            # decision for a different message.
            cache_hit = (
                session.pending_topic_change is not None
                and session.pending_query == user_content
            )
            if cache_hit:
                topic_change = session.pending_topic_change
                turn_request_id = session.pending_request_id
            else:
                turn_request_id = None  # store_turn() mints its own below
                topic_change = self._eidolon.detect_topic(
                    message=user_content,
                    active_topic_id=session.active_topic_id,
                    session_id=resolved_session_id,
                    request_id=turn_request_id,
                )
            session.pending_topic_change = None
            session.pending_query = None
            session.pending_request_id = None

            # Non-None in both branches above (cache_hit implies a cached
            # change; the else branch just computed one) — assert narrows
            # it for the type checker.
            assert topic_change is not None
            is_new = topic_change.type == "NEW_TOPIC"
            old_topic_id = session.active_topic_id
            new_topic_label = topic_change.label if is_new else None

            # Store the turn — returns new topic ID if created.
            # turn_id=session.turn_number: see SessionState.turn_number for
            # why this is cached state rather than a value computed here.
            new_topic_id = self._eidolon.store_turn(
                user_message=user_content,
                ai_response=assistant_content,
                peer_id=self._peer_id,
                environment=EnvironmentSnapshot(situation="conversation"),
                turn_id=session.turn_number,
                old_topic_id=old_topic_id,
                new_topic_label=new_topic_label,
                session_id=resolved_session_id,
                request_id=turn_request_id,
            )

            # Update active topic
            if new_topic_id is not None:
                session.active_topic_id = new_topic_id

            # Activity-gated maintenance sweep — see _maybe_run_maintenance()'s
            # docstring for why this runs here (after store_turn(), not before).
            self._maybe_run_maintenance()

        except Exception as e:
            logger.warning("Eidolon sync_turn failed: %s", e)

    def _maybe_run_maintenance(self) -> None:
        """Close stale-open topics as a side effect of real activity.

        Deliberately activity-gated, not timer-driven: closing a topic
        while its owning peer is away risks fragmenting a conversation
        they'd have naturally continued — once closed, topic detection
        has nothing "active" left to compare against, so a returning
        peer's continuation is forced into a NEW_TOPIC instead of
        resuming the old one. Piggybacking on *any* peer's real turn
        avoids that: a topic only gets closed once some turn proves
        enough time has passed, never on a fixed schedule independent
        of activity. Called at the end of sync_turn(), after
        store_turn() — that ordering means the topic THIS turn just
        touched already has a fresh last_activity, so it can never be
        caught by the sweep its own turn triggers, with no exclusion
        logic needed.

        No-op unless config.maintenance.enabled — see MaintenanceConfig's
        docstring for why this defaults off. Never raises: a failure
        (e.g. eidolon-server briefly unreachable) is logged and
        swallowed like every other non-critical failure in this class,
        and still updates _last_maintenance_check so a persistent
        failure waits a full interval before retrying rather than
        retrying every single turn.
        """
        if not self._maintenance.enabled or not self._eidolon:
            return

        now = datetime.now(UTC)
        interval = timedelta(hours=self._maintenance.interval_hours)
        if self._last_maintenance_check is not None and now - self._last_maintenance_check < interval:
            return
        self._last_maintenance_check = now

        try:
            cutoff = now - timedelta(hours=self._maintenance.stale_after_hours)
            self._eidolon.close_stale_topics(cutoff, peer_id=self._maintenance.peer_id)
        except Exception as e:
            logger.warning("Eidolon maintenance sweep failed: %s", e)

    # ── Tools ────────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict[str, Any]]:
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

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        """Handle a tool call for one of this provider's tools."""
        if tool_name == "eidolon_search":
            return self._handle_search(args)
        if tool_name == "eidolon_inspect_cluster":
            return self._handle_inspect_cluster(args)
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

    # ── Session Lifecycle Hooks ─────────────────────────

    def on_turn_start(self, turn_number: int, message: str, **kwargs: Any) -> None:
        """Cache Hermes' per-agent turn counter for the current session.

        Hermes calls this via ``agent._memory_manager.on_turn_start(
        agent._user_turn_count, user_message)`` with no session_id at all
        (confirmed against the live Hermes source: neither a positional
        arg nor a kwarg carries it) — so unlike prefetch()/sync_turn(),
        which both receive session_id explicitly, this has to be
        attributed to whichever session is "current." That's exactly the
        same fallback ``_current_session_id`` already exists for (see its
        docstring re: prefetch_all() not forwarding session_id either),
        reused here rather than adding a second mechanism.

        Wrapped defensively because Hermes' own error handling for this
        hook (``MemoryManager.on_turn_start``) only logs a failure at
        ``logger.debug`` — easy to miss — unlike ``on_session_end``, which
        logs at ``warning`` with a traceback. See SessionState.turn_number
        for why the value is cached instead of used here directly, and for
        the known blind spot if Hermes ever runs two sessions' turns
        concurrently through one provider instance.
        """
        if not self._initialized:
            return
        try:
            session = self._get_or_create_session("")
            session.turn_number = turn_number
        except Exception as e:
            logger.warning("Eidolon on_turn_start failed: %s", e)

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
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

        Reset active topic on /reset or /new; carry it forward under the
        new session_id on /resume or /branch (same logical conversation,
        new id). On a rewind, drop the pending prefetch->sync_turn cache
        (it was computed for a turn that no longer exists) but leave
        active_topic_id alone — knowing whether the rewind crossed a topic
        boundary would need transcript inspection this method doesn't do.

        The fresh ``SessionState()`` on reset also zeroes ``turn_number``
        back to 0 — deliberately, since Hermes' own counter behind it
        (``agent._user_turn_count``) does NOT reset here; see
        SessionState.turn_number for why that matters.
        """
        if not new_session_id:
            return
        self._current_session_id = new_session_id

        if reset:
            self._sessions[new_session_id] = SessionState()
            return

        if new_session_id not in self._sessions:
            self._sessions[new_session_id] = self._sessions.pop(
                parent_session_id, SessionState()
            )

        if rewound:
            session = self._sessions[new_session_id]
            session.pending_topic_change = None
            session.pending_query = None
