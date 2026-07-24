"""Document-store implementations of the ``Store`` port.

Two interchangeable backends with identical transactional semantics:

* ``MemoryStore`` — serialized by a process-wide lock; the ``test`` profile.
* ``SQLiteStore`` — single-file durability for the local profiles.

Both buffer writes in an overlay and commit atomically; ``put`` requires the
document to have been read in the same transaction and fails with
``StaleVersion`` when the stored version moved on — the same optimistic-
concurrency contract Firestore transactions provide.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .domain.enums import StaleVersion
from .domain.models import (
    CapacityCounter,
    CostBudgetPeriod,
    CostReservation,
    Document,
    InspectionOutbox,
    Job,
    JobOutbox,
    LedgerEntry,
    Payment,
    PaymentEvent,
    PricingConfig,
    UploadSession,
    User,
)
from .ports import Store, Tx  # re-exported: implementations and protocol live together

__all__ = ["Store", "Tx", "MemoryStore", "SQLiteStore", "doc_version", "bump_version", "collection_of"]

_KINDS: dict[str, type[Document]] = {
    cls.COLLECTION: cls
    for cls in (
        User,
        UploadSession,
        Job,
        LedgerEntry,
        Payment,
        PaymentEvent,
        PricingConfig,
        JobOutbox,
        InspectionOutbox,
        CostReservation,
        CostBudgetPeriod,
        CapacityCounter,
    )
}


def collection_of(kind: type[Document]) -> str:
    collection = kind.COLLECTION
    if not collection:
        raise ValueError(f"{kind.__name__} has no COLLECTION")
    return collection


def doc_version(doc: Document) -> int:
    # Job's public status_version doubles as its optimistic-concurrency token.
    return doc.status_version if isinstance(doc, Job) else doc.version


def bump_version(doc: Document) -> None:
    if isinstance(doc, Job):
        doc.status_version += 1
    else:
        doc.version += 1


class _OverlayTx:
    """Shared read-your-writes overlay with version-checked commits."""

    def __init__(self) -> None:
        self._writes: dict[tuple[str, str], Document | None] = {}
        self._read_versions: dict[tuple[str, str], int | None] = {}

    # -- base access provided by subclasses --------------------------------
    def _base_get(self, collection: str, doc_id: str) -> Document | None:
        raise NotImplementedError

    def _base_list(self, collection: str) -> list[Document]:
        raise NotImplementedError

    def _commit(self, writes: dict[tuple[str, str], Document | None]) -> None:
        raise NotImplementedError

    # -- Tx contract --------------------------------------------------------
    def get(self, kind: type[Document], doc_id: str):
        collection = collection_of(kind)
        key = (collection, doc_id)
        if key in self._writes:
            staged = self._writes[key]
            return copy.deepcopy(staged) if staged is not None else None
        base = self._base_get(collection, doc_id)
        self._read_versions.setdefault(key, None if base is None else doc_version(base))
        return base

    def put(self, doc: Document, doc_id: str) -> None:
        collection = collection_of(type(doc))
        key = (collection, doc_id)
        if key not in self._read_versions:
            raise RuntimeError("put() requires the document to be read in this transaction first")
        self._writes[key] = doc

    def insert(self, doc: Document, doc_id: str) -> None:
        collection = collection_of(type(doc))
        key = (collection, doc_id)
        existing = self.get(type(doc), doc_id)
        if existing is not None:
            raise StaleVersion(f"{collection}/{doc_id} already exists")
        self._writes[key] = doc

    def delete(self, kind: type[Document], doc_id: str) -> None:
        collection = collection_of(kind)
        self.get(kind, doc_id)
        self._writes[(collection, doc_id)] = None

    def query(self, kind, *, where=None, where_in=None, order_by=None, limit=None):
        collection = collection_of(kind)
        results: dict[str, Document] = {}
        for doc in self._base_list(collection):
            results[self._doc_key_of(doc, collection)] = doc
        for (coll, doc_id), staged in self._writes.items():
            if coll != collection:
                continue
            if staged is None:
                results.pop(doc_id, None)
            else:
                results[doc_id] = copy.deepcopy(staged)
        docs = list(results.values())
        if where is not None:
            attr, op, value = where
            if op != "==":
                raise ValueError("only == filters are supported")
            docs = [d for d in docs if getattr(d, attr) == value]
        if where_in is not None:
            attr, values = where_in
            wanted = set(values)
            docs = [d for d in docs if getattr(d, attr) in wanted]
        if order_by is not None:
            docs.sort(key=lambda d: getattr(d, order_by) or 0)
        if limit is not None:
            docs = docs[:limit]
        collection_name = collection
        for doc in docs:
            key = (collection_name, self._doc_key_of(doc, collection_name))
            self._read_versions.setdefault(key, doc_version(doc))
        return docs

    @staticmethod
    def _doc_key_of(doc: Document, collection: str) -> str:
        for attr in _ID_ATTRS[collection]:
            if hasattr(doc, attr):
                return getattr(doc, attr)
        raise ValueError(f"no id attribute for {collection}")

    # -- commit -------------------------------------------------------------
    def flush(self) -> None:
        for (collection, doc_id), staged in self._writes.items():
            base = self._base_get(collection, doc_id)
            base_version = None if base is None else doc_version(base)
            if self._read_versions.get((collection, doc_id)) != base_version:
                raise StaleVersion(f"{collection}/{doc_id} changed concurrently")
            if staged is not None:
                bump_version(staged)
        self._commit(self._writes)


_ID_ATTRS: dict[str, tuple[str, ...]] = {
    "users": ("user_id",),
    "upload_sessions": ("upload_id",),
    "jobs": ("job_id",),
    "ledger_entries": ("ledger_entry_id",),
    "payments": ("payment_id",),
    "payment_events": ("event_key",),
    "pricing_configs": ("pricing_version",),
    "job_outbox": ("outbox_id",),
    "inspection_outbox": ("outbox_id",),
    "cost_reservations": ("reservation_id",),
    "cost_budget_periods": ("period_id",),
    "capacity_counters": ("counter_id",),
}


class _MemoryTx(_OverlayTx):
    def __init__(self, data: dict[str, dict[str, Document]]):
        super().__init__()
        self._data = data

    def _base_get(self, collection: str, doc_id: str):
        doc = self._data.get(collection, {}).get(doc_id)
        return copy.deepcopy(doc) if doc is not None else None

    def _base_list(self, collection: str):
        return [copy.deepcopy(d) for d in self._data.get(collection, {}).values()]

    def _commit(self, writes) -> None:
        for (collection, doc_id), staged in writes.items():
            bucket = self._data.setdefault(collection, {})
            if staged is None:
                bucket.pop(doc_id, None)
            else:
                bucket[doc_id] = copy.deepcopy(staged)


class MemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, Document]] = {}
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self) -> Iterator[_MemoryTx]:
        with self._lock:
            tx = _MemoryTx(self._data)
            yield tx
            tx.flush()


class _SQLiteTx(_OverlayTx):
    def __init__(self, conn: sqlite3.Connection):
        super().__init__()
        self._conn = conn

    def _row_to_doc(self, collection: str, data: str) -> Document:
        return _KINDS[collection].from_dict(json.loads(data))

    def _base_get(self, collection: str, doc_id: str):
        row = self._conn.execute(
            "SELECT data FROM docs WHERE collection = ? AND id = ?", (collection, doc_id)
        ).fetchone()
        return self._row_to_doc(collection, row[0]) if row else None

    def _base_list(self, collection: str):
        rows = self._conn.execute(
            "SELECT data FROM docs WHERE collection = ?", (collection,)
        ).fetchall()
        return [self._row_to_doc(collection, r[0]) for r in rows]

    def _commit(self, writes) -> None:
        for (collection, doc_id), staged in writes.items():
            if staged is None:
                self._conn.execute(
                    "DELETE FROM docs WHERE collection = ? AND id = ?", (collection, doc_id)
                )
            else:
                payload = json.dumps(staged.to_dict(), ensure_ascii=False)
                self._conn.execute(
                    "INSERT OR REPLACE INTO docs (collection, id, version, data) VALUES (?, ?, ?, ?)",
                    (collection, doc_id, doc_version(staged), payload),
                )


class SQLiteStore:
    def __init__(self, path: str | Path):
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS docs (collection TEXT NOT NULL, id TEXT NOT NULL,"
            " version INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY (collection, id))"
        )
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self) -> Iterator[_SQLiteTx]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            tx = _SQLiteTx(self._conn)
            try:
                yield tx
                tx.flush()
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")
