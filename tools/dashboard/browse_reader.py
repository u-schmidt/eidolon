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
from math import ceil

from core.interfaces import StorageProvider
from core.models import Fact, MemoryItem

_PAGE_SIZE = 50


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
    peer_id: str
    cluster_id: int | None
    message_count: int
    created_at: object
    closed_at: object | None


@dataclass
class ClusterRow:
    id: int
    peer_id: str
    label: str | None
    summary_preview: str | None
    topic_count: int
    created_at: object


class BrowseReader:
    """Read-only pagination over ``StorageProvider`` for the dashboard's
    four browse pages (see ``tools/dashboard/app.py``).

    ``facts()`` sorts alphabetically by peer/category/key (``get_all_facts``'s
    existing order) — the one page here that doesn't sort newest-first
    like the other three, since it reuses that method's order rather
    than introducing a second, differently-sorted facts read.
    """

    def __init__(self, storage: StorageProvider) -> None:
        self._storage = storage

    def topics(self, page: int, status: str | None = None, peer_id: str | None = None) -> Page[TopicRow]:
        rows = self._storage.list_topics(status=status, peer_id=peer_id)
        items = [
            TopicRow(
                id=r["id"], label=r["label"], status=r["status"], peer_id=r["peer_id"],
                cluster_id=r["cluster_id"], message_count=r["message_count"],
                created_at=r["created_at"], closed_at=r["closed_at"],
            )
            for r in rows
        ]
        return self._paginate(items, page)

    def facts(self, page: int, category: str | None = None, peer_id: str | None = None) -> Page[Fact]:
        facts = self._storage.get_all_facts()
        if category is not None:
            facts = [f for f in facts if f.category == category]
        if peer_id is not None:
            facts = [f for f in facts if f.peer_id == peer_id]
        return self._paginate(facts, page)

    def memory_items(self, page: int, item_type: str | None = None, topic_id: int | None = None) -> Page[MemoryItem]:
        items = self._storage.list_memory_items(item_type=item_type, topic_id=topic_id)
        return self._paginate(items, page)

    def clusters(self, page: int, peer_id: str | None = None) -> Page[ClusterRow]:
        rows = self._storage.list_clusters(peer_id=peer_id)
        items = [
            ClusterRow(
                id=r["id"], peer_id=r["peer_id"], label=r["label"],
                summary_preview=self._preview(r["summary"]), topic_count=r["topic_count"],
                created_at=r["created_at"],
            )
            for r in rows
        ]
        return self._paginate(items, page)

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
