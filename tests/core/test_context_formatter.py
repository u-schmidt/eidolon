"""Tests for ContextFormatter."""

from __future__ import annotations

from datetime import datetime

from core.context_formatter import ContextFormatter
from core.models import ContextBundle, EnvironmentSnapshot, Fact, MemoryItem, RelatedTopic


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


class TestContextFormatterEnvironment:
    def test_no_environment_section_when_bundle_environment_is_none(self):
        bundle = ContextBundle(current_topic_label="Bello")
        assert "[Environment]" not in ContextFormatter.format(bundle)

    def test_no_environment_section_when_all_fields_empty(self):
        bundle = ContextBundle(environment=EnvironmentSnapshot())
        assert ContextFormatter.format(bundle) == ""

    def test_renders_time_with_day_of_week_folded_in(self):
        env = EnvironmentSnapshot(time=datetime(2026, 7, 19, 17, 52))
        bundle = ContextBundle(environment=env)
        assert ContextFormatter.format(bundle) == "[Environment]\n- time: Sunday, 2026-07-19 17:52"

    def test_falls_back_to_day_of_week_when_no_time(self):
        env = EnvironmentSnapshot(day_of_week="Sunday")
        bundle = ContextBundle(environment=env)
        assert ContextFormatter.format(bundle) == "[Environment]\n- day: Sunday"

    def test_renders_location_situation_and_rules(self):
        env = EnvironmentSnapshot(location="home", situation="conversation", rules=["don't phone"])
        bundle = ContextBundle(environment=env)
        assert ContextFormatter.format(bundle) == (
            "[Environment]\n- location: home\n- situation: conversation\n- rules: don't phone"
        )

    def test_environment_section_comes_before_current_topic(self):
        bundle = ContextBundle(
            environment=EnvironmentSnapshot(situation="conversation"),
            current_topic_label="Bello",
        )
        assert ContextFormatter.format(bundle) == (
            "[Environment]\n- situation: conversation\n\n[Current topic: Bello]"
        )

    def test_parses_sqlite_string_timestamp_for_environment_time(self):
        env = EnvironmentSnapshot(time="2026-07-19 17:52:00")
        bundle = ContextBundle(environment=env)
        assert ContextFormatter.format(bundle) == "[Environment]\n- time: Sunday, 2026-07-19 17:52"


class TestContextFormatterRelativeAge:
    NOW = datetime(2026, 7, 19, 12, 0, 0)

    def test_fact_age_suffix(self):
        bundle = ContextBundle(
            facts=[Fact(peer_id="user", category="work", key="role", value="engineer",
                        source="user", created_at="2026-06-28 10:00:00")]
        )
        assert ContextFormatter.format(bundle, now=self.NOW) == (
            "[Known facts]\n- work: role = engineer (3 weeks ago)"
        )

    def test_related_topic_age_suffix_with_tags(self):
        bundle = ContextBundle(
            related_topics=[
                RelatedTopic(topic_id=1, summary="Gym routine", tags=["fitness"], timestamp="2026-07-18 09:00:00")
            ]
        )
        assert ContextFormatter.format(bundle, now=self.NOW) == (
            "[Related topics]\n- Gym routine (fitness) (yesterday)"
        )

    def test_memory_item_age_suffix(self):
        bundle = ContextBundle(
            memory_items=[
                MemoryItem(content="Loves Bello", item_type="essence", source_peer="user",
                          subject_peer="user", timestamp="2026-07-19 08:00:00")
            ]
        )
        assert ContextFormatter.format(bundle, now=self.NOW) == (
            "[Memory fragments]\n- [essence] (user about user): Loves Bello (today)"
        )

    def test_no_age_suffix_when_timestamp_missing(self):
        bundle = ContextBundle(
            facts=[Fact(peer_id="user", category="work", key="role", value="engineer", source="user")]
        )
        assert ContextFormatter.format(bundle, now=self.NOW) == "[Known facts]\n- work: role = engineer"

    def test_age_bucket_boundaries(self):
        cases = [
            ("2026-07-19 08:00:00", "today"),
            ("2026-07-18 08:00:00", "yesterday"),
            ("2026-07-15 08:00:00", "4 days ago"),
            ("2026-07-05 08:00:00", "2 weeks ago"),
            ("2026-05-19 08:00:00", "2 months ago"),
            ("2024-07-19 08:00:00", "2 years ago"),
        ]
        for created_at, expected in cases:
            bundle = ContextBundle(
                facts=[Fact(peer_id="user", category="c", key="k", value="v", source="user", created_at=created_at)]
            )
            result = ContextFormatter.format(bundle, now=self.NOW)
            assert result.endswith(f"({expected})"), f"{created_at} -> expected '{expected}', got: {result}"
