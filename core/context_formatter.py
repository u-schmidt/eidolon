"""Formats a ContextBundle into the plain-text block injected into the LLM prompt."""

from __future__ import annotations

from datetime import datetime

from core.models import ContextBundle, EnvironmentSnapshot


class ContextFormatter:
    """Renders a ContextBundle as the text handed to prefetch()'s caller.

    The single source of truth for injection formatting — used by the
    Hermes adapter's prefetch(), tools/cli.py, and
    tools/simulate_injection.py, so a replayed trace and a live call can
    never drift apart in how they render the same bundle.
    """

    @staticmethod
    def format(bundle: ContextBundle, now: datetime | None = None) -> str:
        """Render `bundle` as injected text.

        `now` is the reference point for the relative-age suffixes on
        facts/topics/memory items (e.g. "3 weeks ago") — defaults to the
        actual current time, but callers replaying a past turn should pass
        that turn's own timestamp so ages reflect how old things were
        *then*, not how old they are today.
        """
        now = now or datetime.now()
        parts: list[str] = []

        if bundle.environment:
            env_lines = ContextFormatter._format_environment(bundle.environment)
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

        if bundle.related_topics:
            topic_lines = []
            for rt in bundle.related_topics:
                tags = ", ".join(rt.tags) if rt.tags else ""
                tag_str = f" ({tags})" if tags else ""
                age = ContextFormatter._age_suffix(rt.timestamp, now)
                topic_lines.append(f"- {rt.summary}{tag_str}{age}")
            parts.append("[Related topics]\n" + "\n".join(topic_lines))

        if bundle.memory_items:
            item_lines = [
                f"- [{mi.item_type}] ({mi.source_peer} about {mi.subject_peer}): {mi.content}"
                f"{ContextFormatter._age_suffix(mi.timestamp, now)}"
                for mi in bundle.memory_items
            ]
            parts.append("[Memory fragments]\n" + "\n".join(item_lines))

        return "\n\n".join(parts)

    @staticmethod
    def _format_environment(env: EnvironmentSnapshot) -> list[str]:
        lines: list[str] = []
        parsed_time = ContextFormatter._parse_timestamp(env.time)
        if parsed_time:
            lines.append(f"- time: {parsed_time.strftime('%A, %Y-%m-%d %H:%M')}")
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
    def _age_suffix(value: object, now: datetime) -> str:
        dt = ContextFormatter._parse_timestamp(value)
        if dt is None:
            return ""
        return f" ({ContextFormatter._relative_age(dt, now)})"

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
