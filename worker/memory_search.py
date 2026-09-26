"""Vector search over the agent's own past reasoning (Voyage AI embeddings + Atlas Vector Search).

Each approved/pending proposal is embedded once into db.reasoning(); the nightly scorer fills in
hit/actual_f. The agent queries it with `search_past_reasoning` before deciding. Degrades to a
no-op / {"error": ...} when VOYAGE_API_KEY is empty.
"""
from __future__ import annotations

import logging
import os

from pymongo.operations import SearchIndexModel

import db

log = logging.getLogger("bookie.memory_search")

VOYAGE_MODEL = "voyage-3.5-lite"   # 512-dim via output_dimension
DIMS = 512
INDEX_NAME = "reasoning_vec"
NOT_CONFIGURED = {"error": "vector search not configured"}

_client = None


def configured() -> bool:
    return bool(os.environ.get("VOYAGE_API_KEY", "").strip())


def _voyage():
    global _client
    if _client is None:
        import voyageai
        _client = voyageai.AsyncClient(api_key=os.environ["VOYAGE_API_KEY"])
    return _client


async def embed(text: str, *, input_type: str = "document") -> list[float]:
    """512-dim Voyage embedding. Raises RuntimeError if VOYAGE_API_KEY is empty."""
    if not configured():
        raise RuntimeError("VOYAGE_API_KEY not set")
    r = await _voyage().embed([text], model=VOYAGE_MODEL, input_type=input_type, output_dimension=DIMS)
    return list(r.embeddings[0])


def _text(p: dict) -> str:
    return (f"date={p.get('target_date')} bucket={p.get('bucket')} point={p.get('point_f')} "
            f"confidence={p.get('confidence')}\nrationale: {p.get('rationale', '')}\nbiggest risk: {p.get('biggest_risk', '')}")


async def index_reasoning(proposal_doc: dict) -> bool:
    """Upsert one proposal's reasoning + embedding into db.reasoning(). Returns False when skipped."""
    if not configured():
        log.info("VOYAGE_API_KEY empty; skipping reasoning embedding")
        return False
    try:
        vec = await embed(_text(proposal_doc))
    except Exception as e:   # never let embedding failures break the proposal path
        log.warning("embedding failed: %s", e)
        return False
    doc = {"run_id": proposal_doc["run_id"], "target_date": proposal_doc["target_date"], "bucket": proposal_doc["bucket"],
           "point_f": proposal_doc["point_f"], "confidence": proposal_doc["confidence"], "hit": None, "actual_f": None,
           "rationale": proposal_doc.get("rationale", ""), "biggest_risk": proposal_doc.get("biggest_risk", ""), "embedding": vec}
    await db.reasoning().update_one({"run_id": doc["run_id"]}, {"$set": doc}, upsert=True)
    return True


async def mark_scored(target_date: str, hit: bool, actual_f: float) -> None:
    await db.reasoning().update_many({"target_date": target_date}, {"$set": {"hit": hit, "actual_f": actual_f}})


async def search_similar(query: str, k: int = 5) -> list[dict] | dict:
    if not configured():
        return NOT_CONFIGURED
    try:
        vec = await embed(query, input_type="query")
    except Exception as e:   # rate limits etc. must not abort the agent run
        log.warning("query embedding failed: %s", e)
        return {"error": f"vector search unavailable: {type(e).__name__}"}
    pipeline = [
        {"$vectorSearch": {"index": INDEX_NAME, "path": "embedding", "queryVector": vec, "numCandidates": 50, "limit": k}},
        {"$project": {"_id": 0, "rationale": 1, "target_date": 1, "bucket": 1, "point_f": 1, "confidence": 1,
                      "actual": "$actual_f", "hit": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    return [d async for d in await db.reasoning().aggregate(pipeline)]


async def ensure_vector_index() -> None:
    """Create the Atlas Vector Search index `reasoning_vec` if missing (works via the driver on M10+)."""
    try:
        coll = db.reasoning()
        if "reasoning" not in await db.db().list_collection_names():   # Atlas refuses search indexes on missing collections
            await db.db().create_collection("reasoning")
        existing = [i["name"] async for i in await coll.list_search_indexes()]
        if INDEX_NAME in existing:
            return
        model = SearchIndexModel(
            definition={"fields": [{"type": "vector", "path": "embedding", "numDimensions": DIMS, "similarity": "cosine"}]},
            name=INDEX_NAME, type="vectorSearch")
        await coll.create_search_index(model)
        log.info("created Atlas vector index %s", INDEX_NAME)
    except Exception as e:
        log.warning("ensure_vector_index skipped: %s", e)
