"""All application state lives in MongoDB Atlas. One async client per process."""
import os
from pymongo import AsyncMongoClient

_client: AsyncMongoClient | None = None

def client() -> AsyncMongoClient:
    global _client
    if _client is None:
        _client = AsyncMongoClient(os.environ["MONGODB_URI"], tz_aware=True)
    return _client

def db():
    return client()[os.environ.get("MONGODB_DB", "bookie")]

# collections
def forecasts():     return db()["forecasts"]      # NWS hourly forecast snapshots
def observations():  return db()["observations"]   # KNYC hourly obs
def actuals():       return db()["actuals"]        # NCEI official daily TMAX
def proposals():     return db()["proposals"]      # agent's daily bucket call + reasoning
def scores():        return db()["scores"]         # proposal vs actual, graded
def rules():         return db()["rules"]          # versioned AGENTS.md the agent rewrites
def traces():        return db()["reasoning"]      # embedded reasoning for vector search

async def ensure_indexes():
    await forecasts().create_index([("target_date", 1), ("fetched_at", -1)])
    await observations().create_index([("ts", -1)], unique=True)
    await actuals().create_index([("date", 1)], unique=True)
    await proposals().create_index([("target_date", 1), ("created_at", -1)])
    await scores().create_index([("target_date", 1)], unique=True)
    await rules().create_index([("version", -1)], unique=True)
