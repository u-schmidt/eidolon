"""Tests for ContextFormatter."""

from __future__ import annotations

from core.context_formatter import ContextFormatter
from core.models import ContextBundle, Fact, MemoryItem, RelatedTopic


class TestContextFormatter:
    def test_empty_bundle_formats_to_empty_string(self):
        assert ContextFormatter.format(ContextBundle()) == ""

    def test_includes_current_topic_label(self):
        bundle = ContextBundle(current_topic_label="Bello Features")
        assert ContextFormatter.format(bundle) == "[Current topic: Bello Features]"

    def test_formats_facts(self):
        bundle = ContextBundle(
            facts=[Fact(peer_id="user", category="work", key="role", value="engineer", source="user")]
        )
        assert ContextFormatter.format(bundle) == "[Known facts]\n- work: role = engineer"

    def test_formats_related_topics_with_and_without_tags(self):
        bundle = ContextBundle(
            related_topics=[
                RelatedTopic(topic_id=1, summary="Discussed gym routine", tags=["fitness", "gym"]),
                RelatedTopic(topic_id=2, summary="No tags here", tags=[]),
            ]
        )
        assert ContextFormatter.format(bundle) == (
            "[Related topics]\n"
            "- Discussed gym routine (fitness, gym)\n"
            "- No tags here"
        )

    def test_formats_memory_items(self):
        bundle = ContextBundle(
            memory_items=[
                MemoryItem(content="User loves Bello", item_type="essence", source_peer="user", subject_peer="user")
            ]
        )
        assert ContextFormatter.format(bundle) == (
            "[Memory fragments]\n- [essence] (user about user): User loves Bello"
        )

    def test_joins_sections_with_blank_line(self):
        bundle = ContextBundle(
            current_topic_label="Bello",
            facts=[Fact(peer_id="user", category="work", key="role", value="engineer", source="user")],
        )
        assert ContextFormatter.format(bundle) == (
            "[Current topic: Bello]\n\n[Known facts]\n- work: role = engineer"
        )
