"""Mock providers for testing Eidolon components.

These implement the same Protocols as real providers (LLMProvider, EmbedderProvider)
but return deterministic/simulated responses. No external dependencies needed.
"""

from __future__ import annotations

from core.models import Fact, MemoryItem, MemoryItemBudget


class MockLLM:
    """A mock LLM that returns a predefined response.

    Implements ``LLMProvider`` protocol structurally (duck typing).
    """

    def __init__(self, response: str = "CONTINUE") -> None:
        self.response = response
        self.last_prompt: str | None = None
        self.last_system: str | None = None
        self.last_max_tokens: int | None = None

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.last_prompt = prompt
        self.last_system = system_prompt
        self.last_max_tokens = max_tokens
        return self.response


class MockEmbedder:
    """A mock embedder that returns a fixed vector.

    Implements ``EmbedderProvider`` protocol structurally (duck typing).
    Useful for testing without a real embedding model.
    """

    def __init__(
        self, vector: list[float] | None = None, dimension: int = 4, should_fail: bool = False,
    ) -> None:
        self.vector = vector or [0.1, 0.2, 0.3, 0.4][:dimension]
        self.last_text: str | None = None
        self.should_fail = should_fail

    def embed(self, text: str) -> list[float]:
        if self.should_fail:
            raise RuntimeError("embedding backend unavailable")
        self.last_text = text
        return self.vector


class SpyTracer:
    """A tracer that records every call for test assertions.

    Implements ``Tracer`` protocol structurally (duck typing).
    """

    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.events: list[dict] = []

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.closed = True

    def record(
        self,
        event_type: str,
        payload: dict,
        *,
        request_id: str | None = None,
        session_id: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        self.events.append({
            "event_type": event_type,
            "payload": payload,
            "request_id": request_id,
            "session_id": session_id,
            "duration_ms": duration_ms,
        })

    def events_of_type(self, event_type: str) -> list[dict]:
        return [e for e in self.events if e["event_type"] == event_type]


class SpyMetricsRecorder:
    """A metrics recorder that records every call for test assertions.

    Implements ``MetricsRecorder`` protocol structurally (duck typing).
    Mirrors ``SpyTracer``'s shape — one list per record_* method rather
    than a single merged list, since the three methods are typed/
    single-purpose rather than one generic event stream (see
    core/interfaces.py's MetricsRecorder).
    """

    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.events: list[dict] = []
        self.llm_calls: list[dict] = []
        self.embedding_calls: list[dict] = []

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.closed = True

    def record_event(
        self,
        severity: str,
        step: str,
        message: str,
        *,
        peer_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.events.append({
            "severity": severity, "step": step, "message": message,
            "peer_id": peer_id, "request_id": request_id, "session_id": session_id,
        })

    def record_llm_call(
        self,
        call_type: str,
        duration_ms: float,
        succeeded: bool,
        *,
        peer_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.llm_calls.append({
            "call_type": call_type, "duration_ms": duration_ms, "succeeded": succeeded,
            "peer_id": peer_id, "request_id": request_id, "session_id": session_id,
        })

    def record_embedding_call(
        self,
        purpose: str,
        duration_ms: float,
        dimension: int,
        succeeded: bool,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        peer_id: str | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.embedding_calls.append({
            "purpose": purpose, "duration_ms": duration_ms, "dimension": dimension, "succeeded": succeeded,
            "entity_type": entity_type, "entity_id": entity_id,
            "peer_id": peer_id, "request_id": request_id, "session_id": session_id,
        })


class MockStorage:
    """In-memory StorageProvider for Eidolon facade testing.

    Distinct from the real StorageProvider contract suite (which tests
    SqliteStore/PostgresStore against each other) — this is a minimal,
    dependency-free double so Eidolon's own orchestration logic (and,
    later, EidolonClient parity) can be tested without a real database.
    """

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

    def get_all_facts(self) -> list[Fact]:
        return list(self.facts)

    def set_fact(
        self, peer_id: str, category: str, key: str, value: str, source: str, importance: str = "medium"
    ) -> None:
        self.facts = [
            f for f in self.facts if not (f.peer_id == peer_id and f.category == category and f.key == key)
        ]
        self.facts.append(
            Fact(peer_id=peer_id, category=category, key=key, value=value, source=source, importance=importance)
        )

    def delete_fact(self, peer_id: str, category: str, key: str) -> None:
        self.facts = [
            f for f in self.facts if not (f.peer_id == peer_id and f.category == category and f.key == key)
        ]

    def count_user_turns(self, peer_id: str) -> int:
        return sum(1 for m in self.messages if m["peer_id"] == peer_id and m["role"] == "user")

    def get_fact_items_since(self, subject_peer: str, since_item_id: int) -> list[MemoryItem]:
        return [
            item for item in self.stored_items
            if item.subject_peer == subject_peer
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
                           "status": "active", "peer_id": peer_id, "session_id": session_id,
                           "cluster_id": None, "created_at": None, "closed_at": None, "message_count": 0}
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

    def update_topic_label(self, topic_id: int, label: str) -> None:
        if topic_id in self.topics:
            self.topics[topic_id]["label"] = label

    def count_messages_by_topic(self, topic_id: int) -> int:
        return len(self.get_messages_by_topic(topic_id))

    def search_topics(self, query_embedding=None, tags=None, peer_id=None, limit=5, weights=None,
                       tracer=None, request_id=None, now=None, min_message_count: int = -1):
        return [
            t for t in self.topics.values()
            if t["status"] == "closed"
            and len(self.get_messages_by_topic(t["id"])) > min_message_count
        ][:limit]

    def get_recent_topics(self, peer_id: str, limit: int, min_message_count: int = -1) -> list[dict]:
        return [
            t for t in self.topics.values()
            if t["status"] == "closed"
            and len(self.get_messages_by_topic(t["id"])) > min_message_count
        ][:limit]

    def get_closed_topics_since(self, since) -> list[dict]:
        # Real time-filtering logic is exercised against the real
        # SqliteStore/PostgresStore (contract suite) — same "manually
        # controllable double" philosophy as get_stale_active_topics
        # above. Callers needing specific created_at/closed_at/
        # message_count values set them directly on self.topics[id].
        return [t for t in self.topics.values() if t["status"] == "closed"]

    def list_topics(self, status: str | None = None, peer_id: str | None = None) -> list[dict]:
        # Same "manually controllable double" philosophy as
        # get_closed_topics_since — filters, doesn't try to replicate
        # real ORDER BY created_at semantics.
        return [
            t for t in self.topics.values()
            if (status is None or t["status"] == status)
            and (peer_id is None or t.get("peer_id") == peer_id)
        ]

    def create_cluster(self, peer_id: str, label: str, summary: str,
                       tags: list[str] | None = None, embedding: list[float] | None = None) -> int:
        cid = self._next_cluster_id
        self._next_cluster_id += 1
        self.clusters[cid] = {
            "id": cid, "peer_id": peer_id, "label": label, "summary": summary,
            "tags": tags or [], "topic_count": 0, "embedding": embedding,
            "created_at": None, "updated_at": None,
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

    def list_clusters(self, peer_id: str | None = None) -> list[dict]:
        return [c for c in self.clusters.values() if peer_id is None or c["peer_id"] == peer_id]

    def store_memory_item(self, item: MemoryItem) -> int:
        self.stored_items.append(item)
        item.id = len(self.stored_items)
        return item.id

    def get_memory_item(self, item_id: int) -> dict | None:
        item = next((i for i in self.stored_items if i.id == item_id), None)
        if item is None:
            return None
        return {
            "id": item.id,
            "content": item.content,
            "item_type": item.item_type,
            "source_peer": item.source_peer,
            "subject_peer": item.subject_peer,
            "attribution_type": item.attribution_type,
            "topic_id": item.topic_id,
            "created_at": item.timestamp,
        }

    def search_memory_items(self, query_embedding=None, peer_id=None, budget=None, weights=None,
                             tracer=None, request_id=None, now=None):
        budget = budget or MemoryItemBudget()
        return self.memory_items[:budget.total]

    def get_memory_items_by_turn(self, turn_id: int) -> list[MemoryItem]:
        message_ids = {m["id"] for m in self.messages if m["turn_id"] == turn_id}
        return [item for item in self.stored_items if item.message_id in message_ids]

    def get_memory_items_by_topic(self, topic_id: int) -> list[MemoryItem]:
        return [item for item in self.stored_items if item.topic_id == topic_id]

    def list_memory_items(self, item_type: str | None = None, topic_id: int | None = None) -> list[MemoryItem]:
        return [
            item for item in self.stored_items
            if (item_type is None or item.item_type == item_type)
            and (topic_id is None or item.topic_id == topic_id)
        ]

    def store_message(self, turn_id, role, content, peer_id, topic_id=None, request_id=None):
        msg = {
            "id": len(self.messages) + 1,
            "turn_id": turn_id,
            "role": role,
            "content": content,
            "peer_id": peer_id,
            "topic_id": topic_id,
            "request_id": request_id,
        }
        self.messages.append(msg)
        return msg["id"]

    def get_messages_by_topic(self, topic_id: int) -> list[dict]:
        return [m for m in self.messages if m["topic_id"] == topic_id]

    def get_recent_messages_by_topic(self, topic_id: int, limit: int) -> list[dict]:
        matching = [m for m in self.messages if m["topic_id"] == topic_id]
        return matching[-limit:]

    def get_messages_by_request_id(self, request_id: str) -> list[dict]:
        return [m for m in self.messages if m["request_id"] == request_id]

    def get_recent_turns(self, limit: int, peer_id: str | None = None) -> list[dict]:
        matching = [
            m for m in self.messages
            if m["role"] == "user" and m["request_id"] is not None
            and (peer_id is None or m["peer_id"] == peer_id)
        ]
        return [
            {
                "request_id": m["request_id"], "peer_id": m["peer_id"],
                "topic_id": m["topic_id"], "content": m["content"], "created_at": None,
            }
            for m in reversed(matching[-limit:])
        ]

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


class EidolonMockLLM:
    """Prompt-routing LLM double for exercising a real Eidolon facade —
    distinct from the plain fixed-response ``MockLLM`` above, which
    doesn't need to distinguish detection/extraction/summary/
    consolidation/cluster prompts from each other."""

    def __init__(
        self,
        detection_response: str = "CONTINUE",
        extraction_response: str = "[]",
        summary_response: str = "",
        relabel_response: str = "",
        consolidation_response: str = '{"upserts": [], "deletes": []}',
        cluster_response: str = '{"match_index": null, "label": "", "summary": ""}',
    ):
        self.detection_response = detection_response
        self.extraction_response = extraction_response
        self.summary_response = summary_response
        self.relabel_response = relabel_response
        self.consolidation_response = consolidation_response
        self.cluster_response = cluster_response
        self.last_prompt: str | None = None
        self.last_system: str | None = None

    def complete(self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None) -> str:
        self.last_prompt = prompt
        self.last_system = system_prompt
        sp = (system_prompt or "").lower()
        if "topic detection" in sp:
            return self.detection_response
        if "memory extraction" in sp:
            return self.extraction_response
        if "summarizer" in sp:
            return self.summary_response
        if "label conversation topics" in sp:
            return self.relabel_response
        if "durable facts" in sp:
            return self.consolidation_response
        if "recurring conversation threads" in sp:
            return self.cluster_response
        return ""


class EidolonMockEmbedder:
    """Embedder double for facade testing — distinct from the plain
    fixed-vector ``MockEmbedder`` above, which doesn't track embedded
    texts or take a per-call configurable failure mode."""

    def __init__(self, should_fail: bool = False) -> None:
        self.embedded_texts: list[str] = []
        self.should_fail = should_fail

    def embed(self, text: str) -> list[float]:
        if self.should_fail:
            raise RuntimeError("embedding backend unavailable")
        self.embedded_texts.append(text)
        return [0.1, 0.2, 0.3]
