"""Tests for the Eidolon core facade."""

from __future__ import annotations

from datetime import UTC, datetime

from core import Eidolon
from core.models import ContextBundle, EnvironmentSnapshot, Fact, MemoryItem
from tests.mocks import SpyTracer


class MockStorage:
    """Mock storage for facade testing."""

    def __init__(self) -> None:
        self.facts: list[Fact] = []
        self.topics: dict[int, dict] = {}
        self.memory_items: list[MemoryItem] = []
        self._next_topic_id = 1
        self.closed_topics: list[int] = []
        self.stored_items: list[MemoryItem] = []
        self.messages: list[dict] = []
        self.topic_embeddings: dict[int, list[float]] = {}
        self.topic_tags: dict[int, list[str]] = {}
        self._checkpoints: dict[str, int] = {}
        self.clusters: dict[int, dict] = {}
        self._next_cluster_id = 1
        self.cluster_candidates: dict[int, list[dict]] = {}
        self.stale_topic_ids: set[int] = set()

    def get_facts(self, peer_id: str) -> list[Fact]:
        return [f for f in self.facts if f.peer_id == peer_id]

    def set_fact(self, peer_id: str, category: str, key: str, value: str, source: str) -> None:
        self.facts = [
            f for f in self.facts if not (f.peer_id == peer_id and f.category == category and f.key == key)
        ]
        self.facts.append(Fact(peer_id=peer_id, category=category, key=key, value=value, source=source))

    def delete_fact(self, peer_id: str, category: str, key: str) -> None:
        self.facts = [
            f for f in self.facts if not (f.peer_id == peer_id and f.category == category and f.key == key)
        ]

    def count_user_turns(self, peer_id: str) -> int:
        return sum(1 for m in self.messages if m["peer_id"] == peer_id and m["role"] == "user")

    def get_fact_items_since(self, subject_peer: str, since_item_id: int) -> list[MemoryItem]:
        return [
            item for item in self.stored_items
            if item.item_type == "fact" and item.subject_peer == subject_peer
            and item.id is not None and item.id > since_item_id
        ]

    def get_consolidation_checkpoint(self, peer_id: str) -> int:
        return self._checkpoints.get(peer_id, 0)

    def set_consolidation_checkpoint(self, peer_id: str, last_item_id: int) -> None:
        self._checkpoints[peer_id] = last_item_id

    def create_topic(self, label: str, summary: str = "", tags: list[str] | None = None,
                     embedding: list[float] | None = None, peer_id: str | None = None,
                     session_id: str | None = None) -> int:
        tid = self._next_topic_id
        self._next_topic_id += 1
        self.topics[tid] = {"id": tid, "label": label, "summary": summary, "tags": tags or [],
                           "status": "active", "peer_id": peer_id, "session_id": session_id}
        return tid

    def get_topic(self, topic_id: int) -> dict | None:
        return self.topics.get(topic_id)

    def close_topic(self, topic_id: int, summary: str) -> None:
        if topic_id in self.topics:
            self.topics[topic_id]["status"] = "closed"
            self.topics[topic_id]["summary"] = summary
            self.closed_topics.append(topic_id)

    def get_active_topic(self, peer_id: str, session_id: str | None = None) -> int | None:
        active = [
            t for t in self.topics.values()
            if t["status"] == "active" and t.get("peer_id") == peer_id
            and t.get("session_id") == session_id
        ]
        return active[-1]["id"] if active else None

    def update_topic_embedding(self, topic_id: int, embedding: list[float]) -> None:
        self.topic_embeddings[topic_id] = embedding

    def update_topic_tags(self, topic_id: int, tags: list[str]) -> None:
        self.topic_tags[topic_id] = tags
        if topic_id in self.topics:
            self.topics[topic_id]["tags"] = tags

    def search_topics(self, query_embedding=None, tags=None, peer_id=None, limit=5, weights=None,
                       tracer=None, request_id=None, now=None):
        return [t for t in self.topics.values() if t["status"] == "closed"][:limit]

    def get_recent_topics(self, peer_id: str, limit: int) -> list[dict]:
        return [t for t in self.topics.values() if t["status"] == "closed"][:limit]

    def create_cluster(self, peer_id: str, label: str, summary: str,
                       tags: list[str] | None = None, embedding: list[float] | None = None) -> int:
        cid = self._next_cluster_id
        self._next_cluster_id += 1
        self.clusters[cid] = {
            "id": cid, "peer_id": peer_id, "label": label, "summary": summary,
            "tags": tags or [], "topic_count": 0, "embedding": embedding,
        }
        return cid

    def get_cluster(self, cluster_id: int) -> dict | None:
        return self.clusters.get(cluster_id)

    def update_cluster(self, cluster_id: int, summary: str, label: str | None = None,
                       tags: list[str] | None = None, embedding: list[float] | None = None) -> None:
        if cluster_id not in self.clusters:
            return
        self.clusters[cluster_id]["summary"] = summary
        if label is not None:
            self.clusters[cluster_id]["label"] = label
        if tags is not None:
            self.clusters[cluster_id]["tags"] = tags
        if embedding is not None:
            self.clusters[cluster_id]["embedding"] = embedding

    def assign_topic_to_cluster(self, topic_id: int, cluster_id: int) -> None:
        if topic_id in self.topics:
            self.topics[topic_id]["cluster_id"] = cluster_id
        if cluster_id in self.clusters:
            self.clusters[cluster_id]["topic_count"] += 1

    def get_topics_by_cluster(self, cluster_id: int) -> list[dict]:
        return [t for t in self.topics.values() if t.get("cluster_id") == cluster_id]

    def get_recent_topics_in_cluster(self, cluster_id: int, limit: int) -> list[dict]:
        return self.get_topics_by_cluster(cluster_id)[:limit]

    def search_clusters(self, query_embedding=None, tags=None, peer_id=None, limit=5, weights=None,
                        tracer=None, request_id=None, now=None) -> list[dict]:
        return [c for c in self.clusters.values() if c["peer_id"] == peer_id][:limit]

    def search_cluster_candidates(self, topic_id: int, peer_id: str, limit: int = 5, weights=None,
                                  tracer=None, request_id=None, now=None) -> list[dict]:
        return self.cluster_candidates.get(topic_id, [])[:limit]

    def get_clusters(self, peer_id: str) -> list[dict]:
        return [c for c in self.clusters.values() if c["peer_id"] == peer_id]

    def store_memory_item(self, item: MemoryItem) -> int:
        self.stored_items.append(item)
        item.id = len(self.stored_items)
        return item.id

    def search_memory_items(self, query_embedding=None, peer_id=None, limit=5, weights=None,
                             tracer=None, request_id=None, now=None):
        return self.memory_items[:limit]

    def get_memory_items_by_turn(self, turn_id: int) -> list[MemoryItem]:
        message_ids = {m["id"] for m in self.messages if m["turn_id"] == turn_id}
        return [item for item in self.stored_items if item.message_id in message_ids]

    def store_message(self, turn_id, role, content, peer_id, topic_id=None):
        msg = {
            "id": len(self.messages) + 1,
            "turn_id": turn_id,
            "role": role,
            "content": content,
            "peer_id": peer_id,
            "topic_id": topic_id,
        }
        self.messages.append(msg)
        return msg["id"]

    def get_messages_by_topic(self, topic_id: int) -> list[dict]:
        return [m for m in self.messages if m["topic_id"] == topic_id]

    def get_recent_messages_by_topic(self, topic_id: int, limit: int) -> list[dict]:
        matching = [m for m in self.messages if m["topic_id"] == topic_id]
        return matching[-limit:]

    def get_stale_active_topics(self, cutoff, peer_id: str | None = None) -> list[dict]:
        # Staleness itself is exercised against the real SqliteStore
        # (tests/core/test_sqlite_store.py); here `stale_topic_ids` lets a
        # test control exactly which candidates Eidolon.close_stale_topics
        # is handed, to isolate its own orchestration logic.
        return [
            t for t in self.topics.values()
            if t["id"] in self.stale_topic_ids and t["status"] == "active"
            and (peer_id is None or t.get("peer_id") == peer_id)
        ]


class MockLLM:
    def __init__(
        self,
        detection_response: str = "CONTINUE",
        extraction_response: str = "[]",
        summary_response: str = "",
        consolidation_response: str = '{"upserts": [], "deletes": []}',
        cluster_response: str = '{"match_index": null, "label": "", "summary": ""}',
    ):
        self.detection_response = detection_response
        self.extraction_response = extraction_response
        self.summary_response = summary_response
        self.consolidation_response = consolidation_response
        self.cluster_response = cluster_response

    def complete(self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None) -> str:
        sp = (system_prompt or "").lower()
        if "topic detection" in sp:
            return self.detection_response
        if "memory extraction" in sp:
            return self.extraction_response
        if "summarizer" in sp:
            return self.summary_response
        if "durable facts" in sp:
            return self.consolidation_response
        if "recurring conversation threads" in sp:
            return self.cluster_response
        return ""


class MockEmbedder:
    def __init__(self, should_fail: bool = False) -> None:
        self.embedded_texts: list[str] = []
        self.should_fail = should_fail

    def embed(self, text: str) -> list[float]:
        if self.should_fail:
            raise RuntimeError("embedding backend unavailable")
        self.embedded_texts.append(text)
        return [0.1, 0.2, 0.3]


class TestEidolon:
    def test_prepare_context_returns_bundle(self):
        storage = MockStorage()
        storage.facts = [Fact(peer_id="user", category="basic", key="name", value="Alex", source="user")]
        llm = MockLLM(detection_response="CONTINUE")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        env = EnvironmentSnapshot(location="home")
        bundle = eidolon.prepare_context(
            message="jep, mach weiter",
            peer_id="user",
            environment=env,
            active_topic_id=1,
        )

        assert isinstance(bundle, ContextBundle)
        assert len(bundle.facts) == 1
        assert bundle.environment is not None
        assert bundle.environment.location == "home"

    def test_prepare_context_new_topic(self):
        storage = MockStorage()
        llm = MockLLM(detection_response="NEW_TOPIC:Bello Features")
        embedder = MockEmbedder()
        eidolon = Eidolon(storage, llm, embedder)

        bundle = eidolon.prepare_context(
            message="Lass uns über Bello sprechen",
            peer_id="user",
            environment=EnvironmentSnapshot(),
        )

        assert isinstance(bundle, ContextBundle)
        # NEW_TOPIC detected → label is set
        assert bundle.current_topic_label == "Bello Features"

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

        bundle = eidolon.prepare_context(
            message="mach weiter", peer_id="user", environment=EnvironmentSnapshot(), active_topic_id=tid,
        )

        assert bundle.current_topic_label == "Talking about Bello's diet"

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

        # Should have created a topic
        assert len(storage.topics) == 1
        topic = list(storage.topics.values())[0]
        assert topic["label"] == "Bello Features"

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

    def test_store_turn_records_topic_label_embedding_event(self):
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
        assert "topic_label" in purposes

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

    def test_store_turn_records_error_on_topic_embedding_failure(self, caplog):
        """An embedding failure must not vanish silently — it should show
        up as an `error` trace event even though store_turn still succeeds
        (embedding failure is non-critical, per the existing except/pass),
        AND as a logger.warning — visible even when debug tracing is off,
        unlike the trace event."""
        storage = MockStorage()
        llm = MockLLM(extraction_response="[]")
        embedder = MockEmbedder(should_fail=True)
        tracer = SpyTracer()
        eidolon = Eidolon(storage, llm, embedder, tracer=tracer)

        with caplog.at_level("WARNING"):
            new_topic_id = eidolon.store_turn(
                user_message="Hi",
                ai_response="Hello",
                peer_id="user",
                environment=EnvironmentSnapshot(),
                new_topic_label="Bello Features",
            )

        assert new_topic_id is not None  # store_turn still succeeds
        error_events = tracer.events_of_type("error")
        assert any(e["payload"]["step"] == "topic_label_embedding" for e in error_events)
        assert "topic_label_embedding failed" in caplog.text

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
