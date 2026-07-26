"""Tests for TopicSummarizer.

Reuses the facade-level mocks from tests.core.test_eidolon (MockStorage,
MockLLM, MockEmbedder) so the summarizer is exercised against the same
storage/LLM/embedder behavior the Eidolon facade sees.
"""

from __future__ import annotations

from core.models import MemoryItem
from core.summarization.topic_summarizer import (
    TopicSummarizer,
    _parse_summary_response,
)
from tests.core.test_eidolon import MockEmbedder, MockLLM, MockStorage
from tests.mocks import SpyMetricsRecorder, SpyTracer


def _summarizer(storage, llm, embedder, tracer=None, metrics=None) -> TopicSummarizer:
    return TopicSummarizer(storage, llm, embedder, tracer=tracer, metrics=metrics)


class TestParseSummaryResponse:
    def test_parses_label_summary_and_tags(self):
        label, summary, tags = _parse_summary_response(
            "Label: Crowded Gym\nSummary: Crowded gym frustration\nTags: fitness, gym"
        )
        assert label == "Crowded Gym"
        assert summary == "Crowded gym frustration"
        assert tags == ["fitness", "gym"]

    def test_case_insensitive_labels_and_lowercased_tags(self):
        label, summary, tags = _parse_summary_response(
            "LABEL: A Label\nSUMMARY: A thing\nTAGS: Alpha, BETA"
        )
        assert label == "A Label"
        assert summary == "A thing"
        assert tags == ["alpha", "beta"]

    def test_missing_summary_yields_empty(self):
        label, summary, tags = _parse_summary_response("Tags: only, tags")
        assert label == ""
        assert summary == ""
        assert tags == ["only", "tags"]

    def test_missing_label_yields_empty(self):
        label, summary, tags = _parse_summary_response("Summary: A thing\nTags: only")
        assert label == ""
        assert summary == "A thing"

    def test_blank_tags_yields_empty_list(self):
        label, summary, tags = _parse_summary_response("Summary: A thing\nTags: ")
        assert summary == "A thing"
        assert tags == []


class TestSummarize:
    def test_reembeds_with_summary(self):
        """The embedding generated at topic creation is based on the short
        provisional label, not the richer summary produced when the topic
        closes — summarize() must re-embed so semantic retrieval isn't
        stuck matching against stale text forever."""
        storage = MockStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded again", peer_id="user", topic_id=tid)
        storage.store_message(turn_id=1, role="ai", content="That sounds frustrating!", peer_id="ai", topic_id=tid)
        llm = MockLLM(summary_response="Summary: User frustrated about crowded gym\nTags: fitness, gym")
        embedder = MockEmbedder()

        _summarizer(storage, llm, embedder).summarize(tid)

        assert tid in storage.topic_embeddings
        assert "User frustrated about crowded gym" in embedder.embedded_texts

    def test_returns_true_on_success(self):
        storage = MockStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded", peer_id="user", topic_id=tid)
        llm = MockLLM(summary_response="Summary: Crowded gym frustration\nTags: fitness")
        embedder = MockEmbedder()

        assert _summarizer(storage, llm, embedder).summarize(tid) is True

    def test_returns_false_when_no_messages(self):
        storage = MockStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        llm = MockLLM(summary_response="Summary: Crowded gym frustration\nTags: fitness")
        embedder = MockEmbedder()

        assert _summarizer(storage, llm, embedder).summarize(tid) is False

    def test_returns_false_on_exception(self):
        class FailingStorage(MockStorage):
            def get_messages_by_topic(self, topic_id):
                raise RuntimeError("db unavailable")

        storage = FailingStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        llm = MockLLM(summary_response="Summary: x\nTags: y")
        embedder = MockEmbedder()

        assert _summarizer(storage, llm, embedder).summarize(tid) is False

    def test_closes_topic_with_summary(self):
        storage = MockStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded", peer_id="user", topic_id=tid)
        llm = MockLLM(summary_response="Summary: Crowded gym frustration\nTags: fitness")
        embedder = MockEmbedder()

        _summarizer(storage, llm, embedder).summarize(tid)

        assert storage.topics[tid]["status"] == "closed"
        assert storage.topics[tid]["summary"] == "Crowded gym frustration"

    def test_persists_tags_via_update_topic_tags(self):
        """Tags must be stored through the storage protocol's
        update_topic_tags — not by reaching into a private connection."""
        storage = MockStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded", peer_id="user", topic_id=tid)
        llm = MockLLM(summary_response="Summary: Crowded gym frustration\nTags: fitness, gym")
        embedder = MockEmbedder()

        _summarizer(storage, llm, embedder).summarize(tid)

        assert storage.topic_tags[tid] == ["fitness", "gym"]

    def test_does_not_touch_tags_when_none_parsed(self):
        storage = MockStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded", peer_id="user", topic_id=tid)
        llm = MockLLM(summary_response="Summary: Crowded gym frustration\nTags: ")
        embedder = MockEmbedder()

        _summarizer(storage, llm, embedder).summarize(tid)

        assert tid not in storage.topic_tags

    def test_persists_label_when_topic_still_pending(self):
        """A topic whose label was never finalized mid-conversation
        (still "") gets the close-time pass's produced label — the
        fallback for a topic that closes before TopicRelabeler ever
        got a chance to run."""
        storage = MockStorage()
        tid = storage.create_topic("", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded", peer_id="user", topic_id=tid)
        llm = MockLLM(summary_response="Label: Crowded Gym\nSummary: Crowded gym frustration\nTags: fitness")
        embedder = MockEmbedder()

        _summarizer(storage, llm, embedder).summarize(tid)

        assert storage.topics[tid]["label"] == "Crowded Gym"

    def test_does_not_overwrite_label_topic_relabeler_already_set(self):
        """A topic TopicRelabeler already finalized mid-conversation keeps
        that label — the close-time pass never overwrites it, even if it
        produces a different one."""
        storage = MockStorage()
        tid = storage.create_topic("Already Finalized", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded", peer_id="user", topic_id=tid)
        llm = MockLLM(summary_response="Label: Different Label\nSummary: Crowded gym frustration\nTags: fitness")
        embedder = MockEmbedder()

        _summarizer(storage, llm, embedder).summarize(tid)

        assert storage.topics[tid]["label"] == "Already Finalized"

    def test_records_topic_summarization_event(self):
        storage = MockStorage()
        tid = storage.create_topic("", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded", peer_id="user", topic_id=tid)
        llm = MockLLM(summary_response="Label: Crowded Gym\nSummary: Crowded gym frustration\nTags: fitness, gym")
        embedder = MockEmbedder()
        tracer = SpyTracer()

        _summarizer(storage, llm, embedder, tracer=tracer).summarize(
            tid, request_id="req1", session_id="sess1"
        )

        events = tracer.events_of_type("topic_summarization")
        assert len(events) == 1
        event = events[0]
        assert event["request_id"] == "req1"
        assert event["session_id"] == "sess1"
        assert event["payload"]["topic_id"] == tid
        assert event["payload"]["message_count"] == 1
        assert event["payload"]["label"] == "Crowded Gym"
        assert event["payload"]["label_persisted"] is True
        assert event["payload"]["summary"] == "Crowded gym frustration"
        assert event["payload"]["tags"] == ["fitness", "gym"]

    def test_records_error_on_failure(self, caplog):
        class FailingStorage(MockStorage):
            def get_messages_by_topic(self, topic_id):
                raise RuntimeError("db unavailable")

        storage = FailingStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        llm = MockLLM(summary_response="Summary: x\nTags: y")
        embedder = MockEmbedder()
        tracer = SpyTracer()
        metrics = SpyMetricsRecorder()

        with caplog.at_level("WARNING"):
            _summarizer(storage, llm, embedder, tracer=tracer, metrics=metrics).summarize(tid)  # must not raise

        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "topic_summarization" for e in error_events)
        assert "topic_summarization failed" in caplog.text
        assert any(e["step"] == "topic_summarization" for e in metrics.events)

    def test_records_error_on_embedding_failure(self, caplog):
        storage = MockStorage()
        tid = storage.create_topic("Bello Features", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="hi", peer_id="user", topic_id=tid)
        llm = MockLLM(summary_response="Summary: x\nTags: y")
        embedder = MockEmbedder(should_fail=True)
        tracer = SpyTracer()
        metrics = SpyMetricsRecorder()

        with caplog.at_level("WARNING"):
            _summarizer(storage, llm, embedder, tracer=tracer, metrics=metrics).summarize(tid)  # must not raise

        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "topic_summary_embedding" for e in error_events)
        assert "topic_summary_embedding failed" in caplog.text
        assert any(e["step"] == "topic_summary_embedding" for e in metrics.events)

    def test_records_llm_call_and_embedding_call_metrics(self):
        storage = MockStorage()
        tid = storage.create_topic("", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="Ugh, gym was crowded", peer_id="user", topic_id=tid)
        llm = MockLLM(summary_response="Label: Crowded Gym\nSummary: Crowded gym frustration\nTags: fitness")
        embedder = MockEmbedder()
        metrics = SpyMetricsRecorder()

        _summarizer(storage, llm, embedder, metrics=metrics).summarize(tid, request_id="req1")

        assert len(metrics.llm_calls) == 1
        assert metrics.llm_calls[0]["call_type"] == "topic_summarization"
        assert metrics.llm_calls[0]["succeeded"] is True
        assert metrics.llm_calls[0]["request_id"] == "req1"

        assert len(metrics.embedding_calls) == 1
        assert metrics.embedding_calls[0]["purpose"] == "topic_summary"
        assert metrics.embedding_calls[0]["succeeded"] is True
        assert metrics.embedding_calls[0]["entity_type"] == "topic"
        assert metrics.embedding_calls[0]["entity_id"] == str(tid)


class TestSummaryPromptSources:
    """memory_items are the primary summarization source (full, untruncated
    — items are already short fragments); the raw transcript is a secondary
    fallback, capped, present specifically to catch turns extraction
    produced nothing for."""

    def test_long_memory_item_content_is_not_truncated(self):
        storage = MockStorage()
        tid = storage.create_topic("Story", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="short message", peer_id="user", topic_id=tid)
        long_content = "A" * 800  # longer than any message-preview cap
        storage.store_memory_item(MemoryItem(
            content=long_content, item_type="fact",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            topic_id=tid,
        ))
        llm = MockLLM(summary_response="Summary: x\nTags: y")
        embedder = MockEmbedder()

        _summarizer(storage, llm, embedder).summarize(tid)

        assert long_content in llm.last_prompt

    def test_message_with_no_extracted_items_still_contributes_raw_content(self):
        """A turn extraction produced nothing for (0 items — e.g. a failed
        extraction call) must still be represented via the raw-transcript
        fallback, not silently absent from the summary entirely."""
        storage = MockStorage()
        tid = storage.create_topic("Story", peer_id="user")
        storage.store_message(
            turn_id=1, role="user", content="a message with no extracted fragments",
            peer_id="user", topic_id=tid,
        )
        # No memory_item stored for this topic at all.
        llm = MockLLM(summary_response="Summary: x\nTags: y")
        embedder = MockEmbedder()

        _summarizer(storage, llm, embedder).summarize(tid)

        assert "a message with no extracted fragments" in llm.last_prompt

    def test_prompt_contains_both_blocks_primary_before_secondary(self):
        storage = MockStorage()
        tid = storage.create_topic("Story", peer_id="user")
        storage.store_message(turn_id=1, role="user", content="raw message text", peer_id="user", topic_id=tid)
        storage.store_memory_item(MemoryItem(
            content="extracted fragment text", item_type="fact",
            source_peer="user", subject_peer="user", attribution_type="self_statement",
            topic_id=tid,
        ))
        llm = MockLLM(summary_response="Summary: x\nTags: y")
        embedder = MockEmbedder()

        _summarizer(storage, llm, embedder).summarize(tid)

        prompt = llm.last_prompt
        assert "extracted fragment text" in prompt
        assert "raw message text" in prompt
        assert prompt.index("extracted fragment text") < prompt.index("raw message text")
