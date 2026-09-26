"""MongoStore — a durable langgraph BaseStore on MongoDB Atlas.

One document per item in the `lg_store` collection:
    {namespace: ["bookie","rules"], key: "/AGENTS.md", value: {...}, created_at, updated_at}
Unique index on (namespace, key). Namespace tuples are stored as lists and rebuilt as tuples on read.
Sync path uses pymongo.MongoClient, async path uses pymongo.AsyncMongoClient, both from the same URI.
Semantics mirror langgraph.store.memory.InMemoryStore (filter operators, namespace prefix search,
list_namespaces match conditions). Vector search over `query` is not implemented: a search with a
query returns the filtered items unscored, in updated_at-desc order.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from langgraph.store.base import (
    BaseStore, GetOp, Item, ListNamespacesOp, Op, PutOp, Result, SearchItem, SearchOp,
)
from langgraph.store.memory import _compare_values, _does_match
from pymongo import ASCENDING, AsyncMongoClient, MongoClient

COLLECTION = "lg_store"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _prefix_query(prefix: tuple[str, ...]) -> dict:
    return {f"namespace.{i}": v for i, v in enumerate(prefix)}


def _item(doc: dict) -> Item:
    return Item(value=doc["value"], key=doc["key"], namespace=tuple(doc["namespace"]),
                created_at=doc["created_at"], updated_at=doc["updated_at"])


def _matches_filter(value: dict, flt: dict | None) -> bool:
    if not flt:
        return True
    return all(_compare_values(value.get(k), v) for k, v in flt.items())


def _search_items(docs: list[dict], op: SearchOp) -> list[SearchItem]:
    hits = [d for d in docs if _matches_filter(d["value"], op.filter)]
    page = hits[op.offset: op.offset + op.limit]
    return [SearchItem(namespace=tuple(d["namespace"]), key=d["key"], value=d["value"],
                       created_at=d["created_at"], updated_at=d["updated_at"]) for d in page]


def _namespaces(all_ns: list[list[str]], op: ListNamespacesOp) -> list[tuple[str, ...]]:
    ns = [tuple(n) for n in all_ns]
    if op.match_conditions:
        ns = [n for n in ns if all(_does_match(c, n) for c in op.match_conditions)]
    ns = sorted({n[: op.max_depth] for n in ns}) if op.max_depth is not None else sorted(ns)
    return ns[op.offset: op.offset + op.limit]


_NS_PIPELINE = [{"$group": {"_id": "$namespace"}}]   # distinct() would flatten the array field


def _put_update(op: PutOp) -> dict:
    ts = _now()
    return {"$set": {"value": op.value, "updated_at": ts}, "$setOnInsert": {"created_at": ts}}


class MongoStore(BaseStore):
    """Durable BaseStore backed by an Atlas collection. Safe to share across threads/tasks."""

    def __init__(self, uri: str, db_name: str, collection: str = COLLECTION) -> None:
        self._sync = MongoClient(uri, tz_aware=True)
        self._async = AsyncMongoClient(uri, tz_aware=True)
        self._db_name, self._coll_name = db_name, collection
        self._indexed = False

    # ---- collections -------------------------------------------------------------------
    @property
    def coll(self):
        return self._sync[self._db_name][self._coll_name]

    @property
    def acoll(self):
        return self._async[self._db_name][self._coll_name]

    def ensure_index(self) -> None:
        if not self._indexed:
            self.coll.create_index([("namespace", ASCENDING), ("key", ASCENDING)], unique=True, name="ns_key")
            self._indexed = True

    async def aensure_index(self) -> None:
        if not self._indexed:
            await self.acoll.create_index([("namespace", ASCENDING), ("key", ASCENDING)], unique=True, name="ns_key")
            self._indexed = True

    def close(self) -> None:
        self._sync.close()

    async def aclose(self) -> None:
        self._sync.close()
        await self._async.close()

    # ---- sync --------------------------------------------------------------------------
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        self.ensure_index()
        c = self.coll
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                doc = c.find_one({"namespace": list(op.namespace), "key": op.key})
                results.append(_item(doc) if doc else None)
            elif isinstance(op, PutOp):
                sel = {"namespace": list(op.namespace), "key": op.key}
                if op.value is None:
                    c.delete_one(sel)
                else:
                    c.update_one(sel, _put_update(op), upsert=True)
                results.append(None)
            elif isinstance(op, SearchOp):
                docs = list(c.find(_prefix_query(op.namespace_prefix)).sort("updated_at", -1))
                results.append(_search_items(docs, op))
            elif isinstance(op, ListNamespacesOp):
                results.append(_namespaces([d["_id"] for d in c.aggregate(_NS_PIPELINE)], op))
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")
        return results

    # ---- async -------------------------------------------------------------------------
    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        await self.aensure_index()
        c = self.acoll
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                doc = await c.find_one({"namespace": list(op.namespace), "key": op.key})
                results.append(_item(doc) if doc else None)
            elif isinstance(op, PutOp):
                sel = {"namespace": list(op.namespace), "key": op.key}
                if op.value is None:
                    await c.delete_one(sel)
                else:
                    await c.update_one(sel, _put_update(op), upsert=True)
                results.append(None)
            elif isinstance(op, SearchOp):
                docs = await c.find(_prefix_query(op.namespace_prefix)).sort("updated_at", -1).to_list()
                results.append(_search_items(docs, op))
            elif isinstance(op, ListNamespacesOp):
                results.append(_namespaces([d["_id"] async for d in await c.aggregate(_NS_PIPELINE)], op))
            else:
                raise ValueError(f"Unknown operation type: {type(op)}")
        return results
