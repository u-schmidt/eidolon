"""Formats a ContextBundle into the plain-text block injected into the LLM prompt."""

from __future__ import annotations

from core.models import ContextBundle


class ContextFormatter:
    """Renders a ContextBundle as the text handed to prefetch()'s caller.

    The single source of truth for injection formatting — used by the
    Hermes adapter's prefetch() and by tools/simulate_injection.py, so a
    replayed trace and a live call can never drift apart in how they
    render the same bundle.
    """

    @staticmethod
    def format(bundle: ContextBundle) -> str:
        parts: list[str] = []

        if bundle.current_topic_label:
            parts.append(f"[Current topic: {bundle.current_topic_label}]")

        if bundle.facts:
            fact_lines = [f"- {f.category}: {f.key} = {f.value}" for f in bundle.facts]
            parts.append("[Known facts]\n" + "\n".join(fact_lines))

        if bundle.related_topics:
            topic_lines = []
            for rt in bundle.related_topics:
                tags = ", ".join(rt.tags) if rt.tags else ""
                tag_str = f" ({tags})" if tags else ""
                topic_lines.append(f"- {rt.summary}{tag_str}")
            parts.append("[Related topics]\n" + "\n".join(topic_lines))

        if bundle.memory_items:
            item_lines = [
                f"- [{mi.item_type}] ({mi.source_peer} about {mi.subject_peer}): {mi.content}"
                for mi in bundle.memory_items
            ]
            parts.append("[Memory fragments]\n" + "\n".join(item_lines))

        return "\n\n".join(parts)
