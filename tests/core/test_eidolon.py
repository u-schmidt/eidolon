"""Tests for the Eidolon core facade."""

from __future__ import annotations

from datetime import UTC, datetime

from core import Eidolon
from core.models import ContextBundle, EnvironmentSnapshot, Fact
from tests.mocks import EidolonMockEmbedder as MockEmbedder
from tests.mocks import EidolonMockLLM as MockLLM
from tests.mocks import MockStorage, SpyTracer


class TestEidolon:
    def test_prepare_context_returns_bundle(self):
        storage = MockStorage()
        storage.facts = [Fact(peer_id="user", category="basic", key="name", value="Alex", source="user")]
        llm = MockLLM(detection_response="CONTINUE")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        env = EnvironmentSnapshot(location="home")
        result = eidolon.prepare_context(
            message="jep, mach weiter",
            peer_id="user",
            environment=env,
            active_topic_id=1,
        )
        bundle = result.bundle

        assert isinstance(bundle, ContextBundle)
        assert len(bundle.facts) == 1
        assert bundle.environment is not None
        assert bundle.environment.location == "home"

    def test_prepare_context_new_topic(self):
        storage = MockStorage()
        llm = MockLLM(detection_response="NEW_TOPIC:Bello Features")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        result = eidolon.prepare_context(
            message="Lass uns über Bello sprechen",
            peer_id="user",
            environment=EnvironmentSnapshot(),
        )
        bundle = result.bundle

        assert isinstance(bundle, ContextBundle)
        # NEW_TOPIC detected → label is set
        assert bundle.current_topic_label == "Bello Features"

    def test_prepare_context_excludes_mini_topics_below_min_topic_messages(self):
        """Locks the facade-to-storage wiring for retrieval_limits'
        min_topic_messages key end-to-end — a thin (1-message) topic must
        not surface in related_topics once the threshold excludes it."""
        storage = MockStorage()
        thin = storage.create_topic("Thin", peer_id="user")
        storage.close_topic(thin, "")
        storage.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=thin)
        thick = storage.create_topic("Thick", peer_id="user")
        storage.close_topic(thick, "")
        for _ in range(3):
            storage.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=thick)
        llm = MockLLM(detection_response="NEW_TOPIC:Bello Features")
        embedder = MockEmbedder()
        eidolon = Eidolon(
            storage, llm, embedder,
            retrieval_limits={
                "related_topics_new": 5, "related_topics_continue": 2,
                "memory_items_new": 3, "memory_items_continue": 1,
                "recent_topics": 2, "cluster_summaries": 3, "cluster_recent_topics": 2,
                "min_topic_messages": 2,
            },
        )

        result = eidolon.prepare_context(
            message="Lass uns über Bello sprechen", peer_id="user", environment=EnvironmentSnapshot(),
        )

        topic_ids = {t.topic_id for t in result.bundle.related_topics}
        assert thick in topic_ids
        assert thin not in topic_ids

    def test_prepare_context_recent_topics_ignores_min_topic_messages(self):
        """[Recently discussed] is documented as unconditional recency
        ('regardless of relevance') — a thin topic must still surface
        there even though the same threshold hides it from related_topics."""
        storage = MockStorage()
        thin = storage.create_topic("Thin", peer_id="user")
        storage.close_topic(thin, "")
        storage.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=thin)
        llm = MockLLM(detection_response="NEW_TOPIC:Bello Features")
        embedder = MockEmbedder()
        eidolon = Eidolon(
            storage, llm, embedder,
            retrieval_limits={
                "related_topics_new": 5, "related_topics_continue": 2,
                "memory_items_new": 3, "memory_items_continue": 1,
                "recent_topics": 5, "cluster_summaries": 3, "cluster_recent_topics": 2,
                "min_topic_messages": 2,
            },
        )

        result = eidolon.prepare_context(
            message="something unrelated", peer_id="user", environment=EnvironmentSnapshot(),
        )

        assert thin in {t.topic_id for t in result.bundle.recent_topics}

    def test_detect_topic_continue(self):
        storage = MockStorage()
        llm = MockLLM(detection_response="CONTINUE")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        result = eidolon.detect_topic("jep, mach weiter", active_topic_id=1)
        assert result.type == "CONTINUE"

    def test_detect_topic_new(self):
        storage = MockStorage()
        llm = MockLLM(detection_response="NEW_TOPIC:Bello")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        result = eidolon.detect_topic("Lass uns über Bello sprechen")
        assert result.type == "NEW_TOPIC"
        assert result.label == "Bello"

    def test_detect_topic_passes_recent_messages_for_active_topic(self):
        """The topic detector's only anchor for an open topic is its
        static creation-time label (summaries only exist once closed) —
        detect_topic must also feed it the topic's recent real messages so
        it isn't stuck comparing against a frozen generic string forever."""
        storage = MockStorage()
        tid = storage.create_topic("General conversation", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="earlier message", peer_id="user", topic_id=tid)
        storage.store_message(turn_id=1, role="ai", content="earlier reply", peer_id="ai", topic_id=tid)
        llm = MockLLM(detection_response="CONTINUE")
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.detect_topic("new message", active_topic_id=tid)

        event = tracer.events_of_type("topic_detection")[0]
        assert event["payload"]["recent_message_count"] == 2

    def test_detect_topic_no_recent_messages_without_active_topic(self):
        storage = MockStorage()
        llm = MockLLM(detection_response="NEW_TOPIC:Bello")
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.detect_topic("first message", active_topic_id=None)

        event = tracer.events_of_type("topic_detection")[0]
        assert event["payload"]["recent_message_count"] == 0

    def test_prepare_context_passes_recent_messages_for_active_topic(self):
        storage = MockStorage()
        tid = storage.create_topic("General conversation", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="earlier message", peer_id="user", topic_id=tid)
        llm = MockLLM(detection_response="CONTINUE")
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.prepare_context(
            message="new message", peer_id="user", environment=EnvironmentSnapshot(), active_topic_id=tid,
        )

        event = tracer.events_of_type("topic_detection")[0]
        assert event["payload"]["recent_message_count"] == 1

    def test_prepare_context_also_returns_topic_change(self):
        storage = MockStorage()
        llm = MockLLM(detection_response="NEW_TOPIC:Bello")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        result = eidolon.prepare_context(
            message="Lass uns über Bello sprechen", peer_id="user", environment=EnvironmentSnapshot(),
        )

        assert result.topic_change.type == "NEW_TOPIC"
        assert result.topic_change.label == "Bello"
        assert result.bundle.current_topic_label == "Bello"

    def test_retrieve_context(self):
        storage = MockStorage()
        storage.facts = [Fact(peer_id="user", category="basic", key="name", value="Alex", source="user")]
        llm = MockLLM()
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        bundle = eidolon.retrieve_context(
            message="test",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            topic_label="Test",
            is_new_topic=True,
        )

        assert len(bundle.facts) == 1
        assert bundle.current_topic_label == "Test"

    def test_prepare_context_continue_keeps_current_topic_label(self):
        """On a CONTINUE turn, prepare_context must resolve the active
        topic's own summary as current_topic_label instead of leaving it
        blank for the rest of the topic's lifetime (topic_change.label is
        only ever set on NEW_TOPIC)."""
        storage = MockStorage()
        tid = storage.create_topic("General", peer_id="user")
        storage.topics[tid]["summary"] = "Talking about Bello's diet"
        llm = MockLLM(detection_response="CONTINUE")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        result = eidolon.prepare_context(
            message="mach weiter", peer_id="user", environment=EnvironmentSnapshot(), active_topic_id=tid,
        )

        assert result.bundle.current_topic_label == "Talking about Bello's diet"

    def test_retrieve_context_continue_resolves_active_topic_label(self):
        storage = MockStorage()
        tid = storage.create_topic("General", peer_id="user")
        storage.topics[tid]["summary"] = "Talking about Bello's diet"
        llm = MockLLM()
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        bundle = eidolon.retrieve_context(
            message="mach weiter", peer_id="user", environment=EnvironmentSnapshot(),
            topic_label=None, is_new_topic=False, active_topic_id=tid,
        )

        assert bundle.current_topic_label == "Talking about Bello's diet"

    def test_store_turn_closes_old_topic(self):
        storage = MockStorage()
        tid = storage.create_topic("Old Topic")
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        eidolon.store_turn(
            user_message="Lass uns über Bello sprechen",
            ai_response="Gerne!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            old_topic_id=tid,
            new_topic_label="Bello Features",
        )

        assert tid in storage.closed_topics

    def test_store_turn_closes_old_topic_with_real_summary_when_available(self):
        """store_turn must keep the LLM-generated summary from the
        TopicSummarizer instead of overwriting it with the
        'Continued to: ...' placeholder."""
        storage = MockStorage()
        tid = storage.create_topic("Old Topic", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded", peer_id="user", topic_id=tid)
        llm = MockLLM(extraction_response="[]", summary_response="Summary: Crowded gym frustration\nTags: fitness")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        eidolon.store_turn(
            user_message="Lass uns über Bello sprechen",
            ai_response="Gerne!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            old_topic_id=tid,
            new_topic_label="Bello Features",
        )

        assert storage.topics[tid]["summary"] == "Crowded gym frustration"

    def test_store_turn_falls_back_to_placeholder_when_summarization_fails(self):
        """If the TopicSummarizer can't produce a real summary (no messages,
        LLM failure, unparseable response), the topic must still end up
        closed — never stuck 'active' forever."""
        storage = MockStorage()
        tid = storage.create_topic("Old Topic", peer_id="user")
        # No messages stored under this topic -> summarize() returns
        # False without closing it.
        llm = MockLLM(extraction_response="[]", summary_response="")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        eidolon.store_turn(
            user_message="Lass uns über Bello sprechen",
            ai_response="Gerne!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            old_topic_id=tid,
            new_topic_label="Bello Features",
        )

        assert storage.topics[tid]["status"] == "closed"
        assert storage.topics[tid]["summary"] == "Continued to: Bello Features"

    def test_store_turn_triggers_cluster_consolidation_on_real_summary(self):
        storage = MockStorage()
        tid = storage.create_topic("Old Topic", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded", peer_id="user", topic_id=tid)
        storage.cluster_candidates[tid] = [
            {"candidate_kind": "topic", "id": 555, "label": "Other topic", "summary": "summary", "topic_count": 1}
        ]
        llm = MockLLM(
            extraction_response="[]",
            summary_response="Summary: Crowded gym frustration\nTags: fitness",
            cluster_response='{"match_index": 1, "label": "Gym saga", "summary": "consolidated"}',
        )
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        eidolon.store_turn(
            user_message="Lass uns über Bello sprechen",
            ai_response="Gerne!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            old_topic_id=tid,
            new_topic_label="Bello Features",
        )

        assert len(storage.clusters) == 1
        cluster = next(iter(storage.clusters.values()))
        assert cluster["label"] == "Gym saga"
        assert cluster["topic_count"] == 2  # the matched raw topic + this one
        assert storage.topics[tid]["cluster_id"] == cluster["id"]

    def test_store_turn_skips_cluster_consolidation_on_fallback_summary(self):
        """No real summary (fallback 'Continued to: ...' path) means
        nothing worth clustering — must not even reach the consolidator,
        regardless of what candidates happen to be sitting there."""
        storage = MockStorage()
        tid = storage.create_topic("Old Topic", peer_id="user")
        storage.cluster_candidates[tid] = [
            {"candidate_kind": "topic", "id": 555, "label": "Other topic", "summary": "summary", "topic_count": 1}
        ]
        llm = MockLLM(extraction_response="[]", summary_response="")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        eidolon.store_turn(
            user_message="msg", ai_response="resp", peer_id="user",
            environment=EnvironmentSnapshot(), old_topic_id=tid, new_topic_label="New",
        )

        assert storage.clusters == {}

    def test_store_turn_skips_cluster_consolidation_when_disabled(self):
        storage = MockStorage()
        tid = storage.create_topic("Old Topic", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="msg", peer_id="user", topic_id=tid)
        storage.cluster_candidates[tid] = [
            {"candidate_kind": "topic", "id": 555, "label": "Other topic", "summary": "summary", "topic_count": 1}
        ]
        llm = MockLLM(
            extraction_response="[]",
            summary_response="Summary: Real summary\nTags: x",
            cluster_response='{"match_index": 1, "label": "X", "summary": "Y"}',
        )
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder, clustering_enabled=False)

        eidolon.store_turn(
            user_message="msg", ai_response="resp", peer_id="user",
            environment=EnvironmentSnapshot(), old_topic_id=tid, new_topic_label="New",
        )

        assert storage.clusters == {}

    def test_store_turn_creates_new_topic(self):
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        eidolon.store_turn(
            user_message="Lass uns über Bello sprechen",
            ai_response="Gerne!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            new_topic_label="Bello Features",
        )

        # Should have created a topic, pending (not yet given a real
        # label) — see TopicRelabeler/TopicSummarizer for how it gets
        # finalized.
        assert len(storage.topics) == 1
        topic = list(storage.topics.values())[0]
        assert topic["label"] == ""

    def test_store_turn_extracts_and_stores_items(self):
        storage = MockStorage()
        llm = MockLLM(
            extraction_response='[{"content": "User wants to discuss Bello", '
            '"item_type": "essence", "source_peer": "user", '
            '"subject_peer": "user", "attribution_type": "self_statement"}]'
        )
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        eidolon.store_turn(
            user_message="Lass uns über Bello sprechen",
            ai_response="Gerne!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            new_topic_label="Bello",
        )

        assert len(storage.stored_items) == 1
        assert storage.stored_items[0].content == "User wants to discuss Bello"
        assert storage.stored_items[0].topic_id is not None

    def test_store_turn_extracts_items_with_topic_id_on_continue(self):
        """Items extracted during a CONTINUE turn (no new topic created)
        must still be linked to the existing active topic, not left with
        topic_id=None — the same resolved_topic_id messages already use."""
        storage = MockStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        llm = MockLLM(
            extraction_response='[{"content": "User is happy about the gym", '
            '"item_type": "feeling", "source_peer": "user", '
            '"subject_peer": "user", "attribution_type": "self_statement"}]'
        )
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        eidolon.store_turn(
            user_message="Ich freu mich schon aufs Training",
            ai_response="Nice!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            old_topic_id=tid,
        )

        assert len(storage.stored_items) == 1
        assert storage.stored_items[0].topic_id == tid

    def test_store_turn_no_topic_change(self):
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        # No old_topic_id and no new_topic_label = no topic change
        eidolon.store_turn(
            user_message="jep",
            ai_response="ok",
            peer_id="user",
            environment=EnvironmentSnapshot(),
        )

        assert len(storage.topics) == 0
        assert len(storage.closed_topics) == 0

    def test_language_parameter(self):
        storage = MockStorage()
        llm = MockLLM()
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder, language="de")
        assert eidolon._language == "de"

    def test_custom_retrieval_weights(self):
        storage = MockStorage()
        llm = MockLLM()
        embedder = MockEmbedder()
        weights = {"semantic": 0.5, "tag_overlap": 0.3, "person": 0.1, "temporal": 0.1}
        eidolon = Eidolon(storage, llm, embedder, retrieval_weights=weights)
        assert eidolon._retriever._weights == weights

    def test_get_active_topic_delegates_to_storage(self):
        storage = MockStorage()
        tid = storage.create_topic("Open Topic", peer_id="user")
        llm = MockLLM()
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        assert eidolon.get_active_topic("user") == tid

    def test_get_active_topic_none_when_no_open_topic(self):
        storage = MockStorage()
        llm = MockLLM()
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        assert eidolon.get_active_topic("user") is None


class TestMidConversationRelabel:
    """Tests for store_turn's step 3b — a CONTINUE turn on a still-pending
    topic (label == "") triggers TopicRelabeler once enough messages have
    accumulated. See core/summarization/topic_relabeler.py."""

    def _continue_turn(self, eidolon: Eidolon, old_topic_id: int) -> None:
        eidolon.store_turn(
            user_message="hi",
            ai_response="hello",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            old_topic_id=old_topic_id,
            new_topic_label=None,
        )

    def test_triggers_relabel_once_threshold_reached(self):
        storage = MockStorage()
        tid = storage.create_topic("", peer_id="user")
        llm = MockLLM(extraction_response="[]", relabel_response="Buying a bike")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder, relabel_message_threshold=2)

        self._continue_turn(eidolon, tid)  # this turn's 2 messages reach the threshold

        assert storage.topics[tid]["label"] == "Buying a bike"

    def test_does_not_trigger_below_threshold(self):
        storage = MockStorage()
        tid = storage.create_topic("", peer_id="user")
        llm = MockLLM(extraction_response="[]", relabel_response="Buying a bike")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder, relabel_message_threshold=10)

        self._continue_turn(eidolon, tid)

        assert storage.topics[tid]["label"] == ""

    def test_does_not_retrigger_already_labeled_topic(self):
        storage = MockStorage()
        tid = storage.create_topic("Already Real", peer_id="user")
        llm = MockLLM(extraction_response="[]", relabel_response="Different Label")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder, relabel_message_threshold=1)

        self._continue_turn(eidolon, tid)

        assert storage.topics[tid]["label"] == "Already Real"

    def test_relabel_enabled_false_suppresses_trigger(self):
        storage = MockStorage()
        tid = storage.create_topic("", peer_id="user")
        llm = MockLLM(extraction_response="[]", relabel_response="Buying a bike")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder, relabel_enabled=False, relabel_message_threshold=1)

        self._continue_turn(eidolon, tid)

        assert storage.topics[tid]["label"] == ""

    def test_new_topic_turn_never_triggers_relabel(self):
        """A NEW_TOPIC turn's just-created topic is nowhere near the
        threshold yet — step 3b only runs on the CONTINUE branch."""
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]", relabel_response="Buying a bike")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder, relabel_message_threshold=1)

        eidolon.store_turn(
            user_message="hi", ai_response="hello", peer_id="user",
            environment=EnvironmentSnapshot(), new_topic_label="Bello Features",
        )

        tid = next(iter(storage.topics))
        assert storage.topics[tid]["label"] == ""

    def test_records_error_on_trigger_failure(self, caplog):
        """A failure in step 3b's own storage reads (get_topic/
        count_messages_by_topic) must not vanish silently or fail the
        turn — matches the file's existing non-critical-failure pattern."""
        class FailingStorage(MockStorage):
            def count_messages_by_topic(self, topic_id):
                raise RuntimeError("db unavailable")

        storage = FailingStorage()
        tid = storage.create_topic("", peer_id="user")
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer, relabel_message_threshold=1)

        with caplog.at_level("WARNING"):
            self._continue_turn(eidolon, tid)  # must not raise

        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "topic_relabel_trigger" for e in error_events)
        assert "topic_relabel_trigger failed" in caplog.text


class TestEidolonTracing:
    """Verifies the tracer actually gets called at the right points with
    the right data — not just that a tracer can be passed in."""

    def test_store_turn_records_turn_complete_event(self):
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.store_turn(
            user_message="Lass uns über Bello sprechen",
            ai_response="Gerne!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            new_topic_label="Bello Features",
            session_id="sess1",
        )

        events = tracer.events_of_type("turn_complete")
        assert len(events) == 1
        event = events[0]
        assert event["session_id"] == "sess1"
        assert event["duration_ms"] is not None
        assert event["payload"]["user_message"] == "Lass uns über Bello sprechen"
        assert event["payload"]["new_topic_label"] == "Bello Features"
        assert event["payload"]["new_topic_id"] is not None

    def test_detect_topic_honors_caller_supplied_request_id(self):
        storage = MockStorage()
        llm = MockLLM()
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.detect_topic("hello", request_id="fixed-id")

        assert {e["request_id"] for e in tracer.events} == {"fixed-id"}

    def test_retrieve_context_honors_caller_supplied_request_id(self):
        storage = MockStorage()
        llm = MockLLM()
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.retrieve_context(
            message="hello", peer_id="user", environment=EnvironmentSnapshot(),
            request_id="fixed-id",
        )

        assert {e["request_id"] for e in tracer.events} == {"fixed-id"}

    def test_store_turn_honors_caller_supplied_request_id(self):
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.store_turn(
            user_message="hi", ai_response="hey", peer_id="user",
            environment=EnvironmentSnapshot(), request_id="fixed-id",
        )

        assert {e["request_id"] for e in tracer.events} == {"fixed-id"}

    def test_detect_topic_and_retrieve_context_share_caller_request_id(self):
        """The core mechanism the Hermes adapter relies on to correlate a
        whole turn: passing the same request_id into both of prefetch()'s
        calls must produce one unified group of trace events, not two."""
        storage = MockStorage()
        llm = MockLLM()
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.detect_topic("hello", request_id="turn-1")
        eidolon.retrieve_context(
            message="hello", peer_id="user", environment=EnvironmentSnapshot(),
            request_id="turn-1",
        )

        assert {e["request_id"] for e in tracer.events} == {"turn-1"}

    def test_store_turn_events_share_request_id(self):
        storage = MockStorage()
        llm = MockLLM(
            extraction_response='[{"content": "fact-worthy", "item_type": "fact", '
            '"source_peer": "user", "subject_peer": "user", "attribution_type": "self_statement"}]',
            consolidation_response='{"upserts": [{"category": "fitness", "key": "gym_frustration", '
            '"value": "crowded gym"}], "deletes": []}',
        )
        embedder = MockEmbedder()
        tracer = SpyTracer()
        # interval=1 so consolidation also fires this turn, so its event's
        # request_id gets exercised by this test too, not just skipped.
        eidolon = Eidolon(storage, llm, embedder, facts_consolidation_interval=1, tracer=tracer)

        eidolon.store_turn(
            user_message="FitZone ist voll",
            ai_response="Nervig!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            new_topic_label="Bello Features",
        )

        assert tracer.events_of_type("fact_consolidation")  # consolidation did fire
        request_ids = {e["request_id"] for e in tracer.events}
        assert len(request_ids) == 1
        assert None not in request_ids

    def test_store_turn_does_not_embed_pending_topic_label(self):
        """A new topic starts pending (label="") — there's nothing
        meaningful to embed yet, so no "topic_label" embedding event
        fires at creation anymore (the topic gets embedded later, once
        TopicRelabeler or TopicSummarizer finalizes a real label/summary)."""
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.store_turn(
            user_message="Hi",
            ai_response="Hello",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            new_topic_label="Bello Features",
        )

        events = tracer.events_of_type("embedding")
        purposes = [e["payload"]["purpose"] for e in events]
        assert "topic_label" not in purposes

    def test_store_turn_records_memory_item_embedding_event(self):
        storage = MockStorage()
        llm = MockLLM(
            extraction_response='[{"content": "User likes hiking", "item_type": "essence", '
            '"source_peer": "user", "subject_peer": "user", "attribution_type": "self_statement"}]'
        )
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        eidolon.store_turn(
            user_message="I love hiking",
            ai_response="Nice!",
            peer_id="user",
            environment=EnvironmentSnapshot(),
        )

        events = tracer.events_of_type("embedding")
        item_events = [e for e in events if e["payload"]["purpose"] == "memory_item"]
        assert len(item_events) == 1
        assert item_events[0]["payload"]["text"] == "User likes hiking"

    def test_store_turn_records_error_on_memory_item_embedding_failure(self, caplog):
        storage = MockStorage()
        llm = MockLLM(
            extraction_response='[{"content": "User likes hiking", "item_type": "essence", '
            '"source_peer": "user", "subject_peer": "user", "attribution_type": "self_statement"}]'
        )
        embedder = MockEmbedder(should_fail=True)
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        with caplog.at_level("WARNING"):
            eidolon.store_turn(
                user_message="I love hiking",
                ai_response="Nice!",
                peer_id="user",
                environment=EnvironmentSnapshot(),
            )

        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "memory_item_embedding" for e in error_events)
        assert "memory_item_embedding failed" in caplog.text

    def test_store_turn_triggers_consolidation_on_interval_multiple(self):
        storage = MockStorage()
        llm = MockLLM(
            extraction_response='[{"content": "fact-worthy", "item_type": "fact", '
            '"source_peer": "user", "subject_peer": "user", "attribution_type": "self_statement"}]',
            consolidation_response='{"upserts": [{"category": "fitness", "key": "gym_frustration", '
            '"value": "crowded gym"}], "deletes": []}',
        )
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, facts_consolidation_interval=2, tracer=tracer)

        eidolon.store_turn(
            user_message="turn 1", ai_response="ok", peer_id="user", environment=EnvironmentSnapshot(),
        )
        assert tracer.events_of_type("fact_consolidation") == []  # turn 1, not a multiple of 2

        eidolon.store_turn(
            user_message="turn 2", ai_response="ok", peer_id="user", environment=EnvironmentSnapshot(),
        )
        events = tracer.events_of_type("fact_consolidation")
        assert len(events) == 1  # turn 2, fires
        assert storage.get_facts("user")  # actually persisted too

    def test_consolidate_facts_records_error_on_malformed_json(self):
        storage = MockStorage()
        llm = MockLLM(
            extraction_response='[{"content": "fact-worthy", "item_type": "fact", '
            '"source_peer": "user", "subject_peer": "user", "attribution_type": "self_statement"}]',
            consolidation_response="not json at all",
        )
        embedder = MockEmbedder()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, facts_consolidation_interval=1, tracer=tracer)

        eidolon.store_turn(
            user_message="turn 1", ai_response="ok", peer_id="user", environment=EnvironmentSnapshot(),
        )

        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "fact_consolidation" for e in error_events)
        assert tracer.events_of_type("fact_consolidation") == []  # never got that far
        assert storage.get_facts("user") == []

    def test_default_tracer_is_noop(self):
        # No tracer passed anywhere — must not raise.
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        eidolon.store_turn(
            user_message="Hi",
            ai_response="Hello",
            peer_id="user",
            environment=EnvironmentSnapshot(),
            new_topic_label="Test",
        )


class TestCloseStaleTopics:
    def test_closes_with_generated_summary_and_runs_clustering(self):
        storage = MockStorage()
        tid = storage.create_topic("Old chat", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=tid)
        storage.stale_topic_ids = {tid}
        llm = MockLLM(summary_response="Summary: wrapped up\nTags: misc")
        eidolon = Eidolon(storage, llm, MockEmbedder())

        closed_ids = eidolon.close_stale_topics(datetime(2026, 1, 1, tzinfo=UTC))

        assert closed_ids == [tid]
        assert storage.topics[tid]["status"] == "closed"
        assert storage.topics[tid]["summary"] == "wrapped up"  # real summary, not the fallback placeholder

    def test_falls_back_to_placeholder_when_summarizer_cannot_produce_one(self):
        storage = MockStorage()
        tid = storage.create_topic("Empty topic", peer_id="user")  # no messages -> summarizer returns False
        storage.stale_topic_ids = {tid}
        eidolon = Eidolon(storage, MockLLM(), MockEmbedder())

        closed_ids = eidolon.close_stale_topics(datetime(2026, 1, 1, tzinfo=UTC))

        assert closed_ids == [tid]
        assert storage.topics[tid]["status"] == "closed"
        assert storage.topics[tid]["summary"] == "Closed: idle timeout"

    def test_only_closes_topics_storage_reports_as_stale(self):
        storage = MockStorage()
        stale = storage.create_topic("Stale", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=stale)
        fresh = storage.create_topic("Fresh", peer_id="user")  # not in stale_topic_ids
        storage.stale_topic_ids = {stale}
        eidolon = Eidolon(storage, MockLLM(summary_response="Summary: x\nTags:"), MockEmbedder())

        closed_ids = eidolon.close_stale_topics(datetime(2026, 1, 1, tzinfo=UTC))

        assert closed_ids == [stale]
        assert storage.topics[fresh]["status"] == "active"

    def test_no_candidates_closes_nothing(self):
        storage = MockStorage()
        storage.create_topic("Active", peer_id="user")  # not marked stale
        eidolon = Eidolon(storage, MockLLM(), MockEmbedder())

        assert eidolon.close_stale_topics(datetime(2026, 1, 1, tzinfo=UTC)) == []

    def test_records_stale_topic_cleanup_trace_event(self):
        storage = MockStorage()
        tid = storage.create_topic("Old chat", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=tid)
        storage.stale_topic_ids = {tid}
        tracer = SpyTracer()
        eidolon = Eidolon(storage, MockLLM(summary_response="Summary: x\nTags:"), MockEmbedder(), tracer=tracer)
        cutoff = datetime(2026, 1, 1, tzinfo=UTC)

        eidolon.close_stale_topics(cutoff, peer_id="user")

        events = tracer.events_of_type("stale_topic_cleanup")
        assert len(events) == 1
        assert events[0]["payload"]["cutoff"] == cutoff.isoformat()
        assert events[0]["payload"]["peer_id"] == "user"
        assert events[0]["payload"]["closed_topic_ids"] == [tid]


class TestCorrectFact:
    def test_upserts_with_source_user(self):
        storage = MockStorage()
        eidolon = Eidolon(storage, MockLLM(), MockEmbedder())

        eidolon.correct_fact("user", "location", "city", "Berlin")

        facts = storage.get_facts("user")
        assert len(facts) == 1
        assert facts[0].category == "location"
        assert facts[0].key == "city"
        assert facts[0].value == "Berlin"
        assert facts[0].source == "user"

    def test_updates_existing_key_in_place(self):
        storage = MockStorage()
        storage.set_fact("user", "location", "city", "Hamburg", source="ai")
        eidolon = Eidolon(storage, MockLLM(), MockEmbedder())

        eidolon.correct_fact("user", "location", "city", "Berlin")

        facts = storage.get_facts("user")
        assert len(facts) == 1
        assert facts[0].value == "Berlin"

    def test_none_value_deletes(self):
        storage = MockStorage()
        storage.set_fact("user", "lifestyle", "smoking", "occasionally", source="ai")
        eidolon = Eidolon(storage, MockLLM(), MockEmbedder())

        eidolon.correct_fact("user", "lifestyle", "smoking", None)

        assert storage.get_facts("user") == []


class TestInspectCluster:
    def test_delegates_to_storage_get_topics_by_cluster(self):
        storage = MockStorage()
        cid = storage.create_cluster("user", "Bike", "summary")
        tid = storage.create_topic("Bought a bike", peer_id="user")
        storage.close_topic(tid, "Bought a bike near Fischmarkt")
        storage.assign_topic_to_cluster(tid, cid)
        eidolon = Eidolon(storage, MockLLM(), MockEmbedder())

        topics = eidolon.inspect_cluster(cid)

        assert [t["id"] for t in topics] == [tid]

    def test_empty_for_unknown_cluster(self):
        storage = MockStorage()
        eidolon = Eidolon(storage, MockLLM(), MockEmbedder())

        assert eidolon.inspect_cluster(999) == []

    def test_blank_value_deletes(self):
        storage = MockStorage()
        storage.set_fact("user", "lifestyle", "smoking", "occasionally", source="ai")
        eidolon = Eidolon(storage, MockLLM(), MockEmbedder())

        eidolon.correct_fact("user", "lifestyle", "smoking", "   ")

        assert storage.get_facts("user") == []

    def test_normalizes_category_and_key_casing(self):
        storage = MockStorage()
        eidolon = Eidolon(storage, MockLLM(), MockEmbedder())

        eidolon.correct_fact("user", "  Location ", "CITY", "Berlin")

        facts = storage.get_facts("user")
        assert facts[0].category == "location"
        assert facts[0].key == "city"

    def test_records_fact_correction_event_on_upsert(self):
        storage = MockStorage()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, MockLLM(), MockEmbedder(), tracer=tracer)

        eidolon.correct_fact(
            "user", "location", "city", "Berlin", request_id="req1", session_id="sess1",
        )

        events = tracer.events_of_type("fact_correction")
        assert len(events) == 1
        event = events[0]
        assert event["request_id"] == "req1"
        assert event["session_id"] == "sess1"
        assert event["payload"] == {
            "peer_id": "user", "category": "location", "key": "city",
            "value": "Berlin", "action": "upsert",
        }

    def test_records_fact_correction_event_on_delete(self):
        storage = MockStorage()
        tracer = SpyTracer()
        eidolon = Eidolon(storage, MockLLM(), MockEmbedder(), tracer=tracer)

        eidolon.correct_fact("user", "lifestyle", "smoking", None)

        events = tracer.events_of_type("fact_correction")
        assert len(events) == 1
        assert events[0]["payload"]["action"] == "delete"
        assert events[0]["payload"]["value"] is None
