"""Composes the storage layer into paginated "browse the raw rows" views
over topics, facts, memory items, and clusters — the plain inspection
counterpart to the dashboard's aggregate insight pages (Performance,
Retrieval Quality, Prompt Quality, Topic Lifecycle, Cluster Dynamics),
which only ever summarize this data, never let you look at a row
directly.

Always-on — every entity here is read straight from the primary storage
tables (``StorageProvider``), not the debug-gated trace store, so this
needs no ``debug.enabled`` trace data at all, unlike most of the insight
pages. Lives in ``tools/``, not ``core/``, for the same reason every
other dashboard reader does — read + pagination glue for a dashboard
view, not core orchestration logic.

Pagination is done in Python, not SQL ``OFFSET`` — a genuinely new
pattern would otherwise be needed in both store backends for what is, at
current scale (dozens to low thousands of rows in a single-user memory
store), a single cheap in-memory slice. See ``StorageProvider.list_topics``/
``list_memory_items``/``list_clusters`` docstrings for the bulk reads
this pages over.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import ceil

from core.interfaces import StorageProvider
from core.models import Fact, MemoryItem

_PAGE_SIZE = 50

# Lower rank sorts first — used only by facts(sort_by="importance").
_IMPORTANCE_RANK = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Page[T]:
    items: list[T]
    page: int
    page_size: int
    total: int

    @property
    def total_pages(self) -> int:
        return max(1, ceil(self.total / self.page_size))

    @property
    def has_prev(self) -> bool:
        return self.page > 1

    @property
    def has_next(self) -> bool:
        return self.page < self.total_pages


@dataclass
class TopicRow:
    id: int
    label: str | None
    status: str
    workspace_id: str
    cluster_id: int | None
    message_count: int
    created_at: object
    closed_at: object | None


@dataclass
class ClusterRow:
    id: int
    workspace_id: str
    label: str | None
    summary_preview: str | None
    topic_count: int
    created_at: object


@dataclass
class ClusterTopicRow:
    id: int
    label: str | None
    status: str
    workspace_id: str
    summary_preview: str | None
    created_at: object
    closed_at: object | None


@dataclass
class ClusterDetail:
    id: int
    workspace_id: str
    label: str | None
    summary: str | None
    tags: list[str]
    topic_count: int
    created_at: object
    updated_at: object
    topics: list[ClusterTopicRow]


@dataclass
class OverviewCounts:
    """Aggregate counts for the landing page's at-a-glance cards."""

    total_turns: int
    total_topics: int
    active_topics: int
    total_facts: int
    total_memory_items: int
    total_clusters: int


class BrowseReader:
    """Read-only pagination over ``StorageProvider`` for the dashboard's
    four browse pages (see ``tools/dashboard/app.py``).

    ``facts()`` sorts alphabetically by workspace/category/key (``get_all_facts``'s
    existing order) — the one page here that doesn't sort newest-first
    like the other three, since it reuses that method's order rather
    than introducing a second, differently-sorted facts read.
    """

    def __init__(self, storage: StorageProvider) -> None:
        self._storage = storage

    def overview_counts(self) -> OverviewCounts:
        """Aggregate counts for the landing page — always-on, sourced
        straight from the primary storage tables."""
        topics = self._storage.list_topics()
        return OverviewCounts(
            total_turns=self._storage.count_recent_turns(),
            total_topics=len(topics),
            active_topics=sum(1 for t in topics if t.get("status") == "active"),
            total_facts=len(self._storage.get_all_facts()),
            total_memory_items=len(self._storage.list_memory_items()),
            total_clusters=len(self._storage.list_clusters()),
        )

    def topics(self, page: int, status: str | None = None, workspace_id: str | None = None) -> Page[TopicRow]:
        rows = self._storage.list_topics(status=status, workspace_id=workspace_id)
        items = [
            TopicRow(
                id=r["id"], label=r["label"], status=r["status"], workspace_id=r["workspace_id"],
                cluster_id=r["cluster_id"], message_count=r["message_count"],
                created_at=r["created_at"], closed_at=r["closed_at"],
            )
            for r in rows
        ]
        return self._paginate(items, page)

    def facts(
        self, page: int, category: str | None = None, workspace_id: str | None = None, sort_by: str | None = None,
    ) -> Page[Fact]:
        facts = self._storage.get_all_facts()
        if category is not None:
            facts = [f for f in facts if f.category == category]
        if workspace_id is not None:
            facts = [f for f in facts if f.workspace_id == workspace_id]
        facts = self._sort_facts(facts, sort_by)
        return self._paginate(facts, page)

    def memory_items(
        self, page: int, item_type: str | None = None, topic_id: int | None = None, q: str | None = None,
    ) -> Page[MemoryItem]:
        items = self._storage.list_memory_items(item_type=item_type, topic_id=topic_id, q=q)
        return self._paginate(items, page)

    def clusters(self, page: int, workspace_id: str | None = None) -> Page[ClusterRow]:
        rows = self._storage.list_clusters(workspace_id=workspace_id)
        items = [
            ClusterRow(
                id=r["id"], workspace_id=r["workspace_id"], label=r["label"],
                summary_preview=self._preview(r["summary"]), topic_count=r["topic_count"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
        return self._paginate(items, page)

    def cluster_detail(self, cluster_id: int) -> ClusterDetail | None:
        """One cluster's full detail — id/label/summary/tags plus every
        member topic, newest first. Reuses ``get_cluster``/
        ``get_topics_by_cluster`` unchanged (the same pair backing the
        ``eidolon_inspect_cluster`` tool) rather than adding a new bulk
        storage method — a single-cluster lookup, not a paginated listing.
        ``get_topics_by_cluster`` returns oldest first (right for that
        tool's chronological narration); reversed here to match every
        other browse page's newest-first convention."""
        cluster = self._storage.get_cluster(cluster_id)
        if cluster is None:
            return None
        topic_rows = self._storage.get_topics_by_cluster(cluster_id)
        topics = [
            ClusterTopicRow(
                id=t["id"], label=t["label"], status=t["status"], workspace_id=t["workspace_id"],
                summary_preview=self._preview(t["summary"]), created_at=t["created_at"],
                closed_at=t["closed_at"],
            )
            for t in reversed(topic_rows)
        ]
        return ClusterDetail(
            id=cluster["id"], workspace_id=cluster["workspace_id"], label=cluster["label"],
            summary=cluster["summary"], tags=cluster["tags"], topic_count=cluster["topic_count"],
            created_at=cluster["created_at"], updated_at=cluster["updated_at"], topics=topics,
        )

    @staticmethod
    def _sort_facts(facts: list[Fact], sort_by: str | None) -> list[Fact]:
        """``sort_by=None`` keeps ``get_all_facts``'s own workspace/category/key
        order unchanged. The other options re-sort in Python — the full
        list is already in memory (see module docstring), so this is a
        cheap in-memory sort, not a second query."""
        if sort_by == "created":
            # Partitioned rather than sorted on a single (is_missing, ts)
            # key: real timestamps (all from one get_all_facts() call, so
            # all the same shape — naive str-parsed for SqliteStore,
            # tz-aware for PostgresStore) are sorted newest-first among
            # themselves, then facts with no parseable timestamp are
            # appended at the end — never compared against a fabricated
            # stand-in of possibly-mismatched naive/aware-ness.
            dated_with_ts: list[tuple[Fact, datetime]] = []
            undated: list[Fact] = []
            for f in facts:
                ts = BrowseReader._parse_timestamp(f.created_at)
                if ts is None:
                    undated.append(f)
                else:
                    dated_with_ts.append((f, ts))
            dated = sorted(dated_with_ts, key=lambda p: p[1], reverse=True)
            return [f for f, _ in dated] + undated
        if sort_by == "importance":
            return sorted(facts, key=lambda f: _IMPORTANCE_RANK.get(f.importance, 3))
        if sort_by == "key":
            return sorted(facts, key=lambda f: f.key)
        return facts

    @staticmethod
    def _parse_timestamp(value: object) -> datetime | None:
        """``created_at`` is a raw string for SqliteStore rows and a real
        ``datetime`` for PostgresStore rows — same dual-shape handling as
        ``tools/dashboard/topic_lifecycle_reader.py``'s ``_parse_timestamp``."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _paginate[T](items: list[T], page: int) -> Page[T]:
        page = max(1, page)
        total = len(items)
        start = (page - 1) * _PAGE_SIZE
        return Page(items=items[start:start + _PAGE_SIZE], page=page, page_size=_PAGE_SIZE, total=total)

    @staticmethod
    def _preview(text: str | None, limit: int = 160) -> str | None:
        if not text:
            return text
        text = " ".join(text.split())
        return text if len(text) <= limit else text[:limit] + "…"
