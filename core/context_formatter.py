"""Formats a ContextBundle into the plain-text block injected into the LLM prompt."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from core.models import ClusterSummary, ContextBundle, EnvironmentSnapshot, Fact, RelatedTopic


class ContextFormatter:
    """Renders a ContextBundle as the text handed to prefetch()'s caller.

    The single source of truth for injection formatting — used by the
    Hermes adapter's prefetch(), tools/cli.py, and
    tools/simulate_injection.py, so a replayed trace and a live call can
    never drift apart in how they render the same bundle.
    """

    _PREAMBLE = (
        "[Memory Context]\n"
        "The following is background context recalled for this turn — not something the user just said. "
        "Facts are stable, durable information; recently discussed topics are what came up most recently "
        "regardless of relevance; related topics are past conversation summaries chosen for relevance to "
        "this message; recurring topics are consolidated narratives built from several related past "
        "conversations on the same underlying subject; memory fragments are specific extracted moments. "
        "Use only what's relevant to the current message."
    )

    @staticmethod
    def format(bundle: ContextBundle, now: datetime | None = None, timezone: str = "UTC") -> str:
        """Render `bundle` as injected text.

        `now` is the reference point for the relative-age suffixes on
        facts/topics/memory items (e.g. "3 weeks ago") — defaults to the
        actual current time, but callers replaying a past turn should pass
        that turn's own timestamp so ages reflect how old things were
        *then*, not how old they are today.

        `timezone` is an IANA zone name (e.g. "Europe/Berlin") used only to
        display the `[Environment]` time — stored timestamps stay UTC
        internally, so relative-age math is unaffected by this setting.
        """
        now = now or datetime.now(UTC)
        parts: list[str] = []

        if bundle.environment:
            env_lines = ContextFormatter._format_environment(bundle.environment, timezone)
            if env_lines:
                parts.append("[Environment]\n" + "\n".join(env_lines))

        if bundle.current_topic_label:
            parts.append(f"[Current topic: {bundle.current_topic_label}]")

        if bundle.facts:
            fact_lines = [
                f"- {f.category}: {f.key} = {f.value}"
                f"{ContextFormatter._age_suffix(f.created_at, now)}"
                for f in bundle.facts
            ]
            parts.append("[Known facts]\n" + "\n".join(fact_lines))

        if bundle.entity_facts:
            parts.append(ContextFormatter._format_entity_facts_block(bundle.entity_facts, now))

        if bundle.recent_topics:
            lines = ContextFormatter._format_topic_lines(bundle.recent_topics, now)
            parts.append("[Recently discussed]\n" + "\n".join(lines))

        if bundle.related_topics:
            lines = ContextFormatter._format_topic_lines(bundle.related_topics, now)
            parts.append("[Related topics]\n" + "\n".join(lines))

        if bundle.cluster_summaries:
            lines = ContextFormatter._format_cluster_lines(bundle.cluster_summaries, now)
            parts.append("[Recurring topics]\n" + "\n".join(lines))

        if bundle.memory_items:
            item_lines = [
                f"- [{mi.item_type}] ({mi.source_peer} about {mi.subject_peer}): {mi.content}"
                f"{ContextFormatter._age_suffix(mi.timestamp, now)}"
                for mi in bundle.memory_items
            ]
            parts.append("[Memory fragments]\n" + "\n".join(item_lines))

        if parts:
            parts.insert(0, ContextFormatter._PREAMBLE)

        return "\n\n".join(parts)

    @staticmethod
    def _format_entity_facts_block(entity_facts: list[Fact], now: datetime) -> str:
        """Render one ``[About X]`` section per entity, grouped in the
        order they arrive (``get_entity_facts`` already orders by
        ``entity_name``). Separate from ``[Known facts]`` — these are
        conditionally pulled per-turn (see AssociativeRetriever step
        1b), not the workspace's always-on profile.
        """
        by_entity: dict[str, list[Fact]] = {}
        for f in entity_facts:
            by_entity.setdefault(f.entity_name, []).append(f)
        sections = []
        for entity_name, facts in by_entity.items():
            lines = [
                f"- {f.category}: {f.key} = {f.value}{ContextFormatter._age_suffix(f.created_at, now)}"
                for f in facts
            ]
            sections.append(f"[About {entity_name.title()}]\n" + "\n".join(lines))
        return "\n\n".join(sections)

    @staticmethod
    def _format_topic_lines(topics: list[RelatedTopic], now: datetime) -> list[str]:
        lines = []
        for rt in topics:
            tags = ", ".join(rt.tags) if rt.tags else ""
            tag_str = f" ({tags})" if tags else ""
            age = ContextFormatter._age_suffix(rt.timestamp, now)
            summary = f"{rt.label}: {rt.summary}" if rt.label else rt.summary
            lines.append(f"- {summary}{tag_str}{age}")
        return lines

    @staticmethod
    def _format_cluster_lines(clusters: list[ClusterSummary], now: datetime) -> list[str]:
        """Render each cluster as its current-state summary, an optional
        evolution trail of milestones, and a short "newest: ..." tail of
        its most recently closed member topics — the summary carries
        where things stand now, milestones carry how it got there, the
        tail shows what's fresh.
        """
        lines = []
        for c in clusters:
            newest_bits = [
                f"{rt.label or rt.summary}{ContextFormatter._age_suffix(rt.timestamp, now)}"
                for rt in c.recent_topics
            ]
            newest = f" Newest: {', '.join(newest_bits)}." if newest_bits else ""
            tag_str = f", tags: {', '.join(c.tags)}" if c.tags else ""
            evolution = f" Evolution: {' → '.join(c.milestones)}." if c.milestones else ""
            lines.append(
                f"- Cluster #{c.cluster_id} {c.label} ({c.topic_count} topics{tag_str}): "
                f"{c.summary}{evolution}{newest}"
            )
        return lines

    @staticmethod
    def _format_environment(env: EnvironmentSnapshot, display_timezone: str) -> list[str]:
        lines: list[str] = []
        parsed_time = ContextFormatter._parse_timestamp(env.time)
        if parsed_time:
            local_time = ContextFormatter._to_display_timezone(parsed_time, display_timezone)
            tz_label = local_time.tzname() or display_timezone
            lines.append(f"- time: {local_time.strftime('%A, %Y-%m-%d %H:%M')} {tz_label}")
        elif env.day_of_week:
            lines.append(f"- day: {env.day_of_week}")
        if env.location:
            lines.append(f"- location: {env.location}")
        if env.situation:
            lines.append(f"- situation: {env.situation}")
        if env.rules:
            lines.append("- rules: " + ", ".join(env.rules))
        return lines

    @staticmethod
    def _to_display_timezone(dt: datetime, tz_name: str) -> datetime:
        """Convert dt to tz_name for display. A naive dt is assumed UTC —
        matches how every timestamp in this system is stored/captured
        (SQLite CURRENT_TIMESTAMP is always UTC; live captures use
        datetime.now(timezone.utc)). Falls back to dt unconverted if
        tz_name isn't a recognized IANA zone.
        """
        try:
            target = ZoneInfo(tz_name)
        except Exception:
            return dt
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(target)

    @staticmethod
    def _age_suffix(value: object, now: datetime) -> str:
        dt = ContextFormatter._parse_timestamp(value)
        if dt is None:
            return ""
        return f" [{ContextFormatter._relative_age(dt, now)}]"

    @staticmethod
    def _parse_timestamp(value: object) -> datetime | None:
        """Tolerant timestamp parser.

        Storage-returned timestamps are raw SQLite strings (e.g.
        "2026-07-19 17:48:58"), not real datetime objects — the store
        never sets detect_types on its connection. EnvironmentSnapshot.time
        populated live, however, *is* a real datetime. Handle both.
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except ValueError:
            return None

    @staticmethod
    def _relative_age(dt: datetime, now: datetime) -> str:
        days = (now.date() - dt.date()).days
        if days <= 0:
            return "today"
        if days == 1:
            return "yesterday"
        if days < 7:
            return f"{days} days ago"
        if days < 30:
            weeks = days // 7
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        if days < 365:
            months = days // 30
            return f"{months} month{'s' if months != 1 else ''} ago"
        years = days // 365
        return f"{years} year{'s' if years != 1 else ''} ago"
