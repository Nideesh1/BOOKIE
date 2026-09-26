"""bookie — a forecasting agent that grades itself against reality and rewrites its own rules.

Hatchet owns the loop and the human pause. deepagents does the thinking. Atlas holds everything.
"""
import os, json, datetime as dt
from datetime import timedelta
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
os.environ.setdefault("LANGSMITH_TRACING", "false")

import httpx
from pydantic import BaseModel, Field
from hatchet_sdk import Context, DurableContext, Hatchet
from hatchet_sdk.opentelemetry.instrumentor import HatchetInstrumentor
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from deepagents import create_deep_agent
from deepagents.backends import StateBackend

import db, nws

# ---- tracing: plain OTLP to Langfuse ------------------------------------------------
provider = TracerProvider(resource=Resource.create({"service.name": "bookie"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)
hatchet = Hatchet()
HatchetInstrumentor(tracer_provider=provider, enable_hatchet_otel_collector=False).instrument()
LangChainInstrumentor().instrument(tracer_provider=provider)

ET = nws.ET
def now(): return dt.datetime.now(dt.timezone.utc)
def tomorrow_et() -> str: return (dt.datetime.now(ET).date() + timedelta(days=1)).isoformat()

# ---- typed outputs ------------------------------------------------------------------
class Proposal(BaseModel):
    target_date: str
    bucket: str = Field(description='2-degree bucket like "62-63"')
    point_f: float = Field(description="best single-number estimate of the daily high, °F")
    confidence: float = Field(ge=0, le=1)
    biggest_risk: str
    rationale: str

class RulesEdit(BaseModel):
    new_rules_md: str = Field(description="the full updated rules document")
    change_summary: str = Field(description="1-3 sentences: what changed and why, citing scores")

# ---- tools the agent can call (all read from Atlas or public APIs) --------------------
@tool
async def get_forecast(target_date: str) -> dict:
    """Latest NWS hourly forecast snapshot for target_date (YYYY-MM-DD): max temp and the hourly curve."""
    doc = await db.forecasts().find_one({"target_date": target_date}, sort=[("fetched_at", -1)])
    if not doc: return {"error": "no forecast stored yet"}
    hours = [p for p in doc["periods"] if p["t"].startswith(target_date)]
    return {"fetched_at": doc["fetched_at"].isoformat(), "day_max_f": doc["day_max"],
            "hourly": [{"h": p["t"][11:13], "f": p["temp_f"], "pop": p["pop"], "sky": p["short"]} for p in hours]}

@tool
async def get_observations(hours: int = 24) -> list[dict]:
    """Most recent Central Park hourly observations (°F), newest first."""
    cur = db.observations().find({}, {"_id": 0}).sort("ts", -1).limit(hours)
    return [o async for o in cur]

@tool
async def get_my_scores(last_n: int = 14) -> list[dict]:
    """Your recent graded calls: predicted bucket vs the official daily high. Read-only."""
    cur = db.scores().find({}, {"_id": 0}).sort("target_date", -1).limit(last_n)
    return [s async for s in cur]

@tool
async def get_recent_actuals(last_n: int = 10) -> list[dict]:
    """Official NCEI daily highs for recent days (what NWS forecast vs what happened)."""
    cur = db.actuals().find({}, {"_id": 0}).sort("date", -1).limit(last_n)
    return [a async for a in cur]

TOOLS = [get_forecast, get_observations, get_my_scores, get_recent_actuals]

# ---- agent --------------------------------------------------------------------------
def make_model():
    return ChatOpenAI(model=os.environ["MODEL_NAME"], base_url=os.environ["MODEL_BASE_URL"],
                      api_key=os.environ["MODEL_API_KEY"], temperature=0, max_tokens=4096,
                      timeout=httpx.Timeout(180.0, connect=10.0), max_retries=1)

async def current_rules() -> dict:
    return await db.rules().find_one({}, sort=[("version", -1)])

def build_agent(checkpointer, rules_text: str):
    return create_deep_agent(
        model=make_model(), tools=TOOLS, backend=StateBackend(), checkpointer=checkpointer,
        system_prompt=("You are bookie, a disciplined weather forecaster. Use the tools; never guess numbers you "
                       "could look up. Your live rulebook, which you wrote from experience, follows.\n\n"
                       "<rules>\n" + rules_text + "\n</rules>"),
    )

async def run_agent(agent, prompt: str, thread_id: str, schema: type[BaseModel]) -> BaseModel:
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    out = await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}, cfg)
    text = out["messages"][-1].content
    if isinstance(text, list): text = " ".join(b.get("text", "") for b in text if isinstance(b, dict))
    return await make_model().with_structured_output(schema).ainvoke(
        f"Extract the final answer into the schema. Source:\n\n{text}")

# ---- workflows ----------------------------------------------------------------------
class DayInput(BaseModel):
    target_date: str | None = None

class Verdict(BaseModel):
    approved: bool
    note: str = ""

VERDICT_EVENT = "bookie:verdict"
market_day = hatchet.workflow(name="market_day", input_validator=DayInput)

@market_day.task(execution_timeout=timedelta(minutes=10), retries=1)
async def propose(input: DayInput, ctx: Context) -> Proposal:
    target = input.target_date or tomorrow_et()
    rules = await current_rules()
    agent = build_agent(ctx.lifespan["saver"], rules["text"])
    with tracer.start_as_current_span("bookie.propose") as span:
        span.set_attribute("target_date", target); span.set_attribute("rules.version", rules["version"])
        p = await run_agent(agent, f"Forecast the official daily high at Central Park for {target}. "
                            "Follow your rules. End with: bucket, point estimate, confidence, biggest risk, rationale.",
                            thread_id=f"{ctx.workflow_run_id}:propose", schema=Proposal)
    p.target_date = target
    await db.proposals().insert_one({**p.model_dump(), "run_id": ctx.workflow_run_id, "rules_version": rules["version"],
                                     "created_at": now(), "status": "pending"})
    return p

@market_day.durable_task(parents=[propose], execution_timeout=timedelta(days=2), schedule_timeout=timedelta(days=2))
async def await_verdict(input: DayInput, ctx: DurableContext) -> dict:
    p = ctx.task_output(propose)
    with tracer.start_as_current_span("bookie.human_gate"):
        ev = await ctx.aio_wait_for_event(VERDICT_EVENT, scope=ctx.workflow_run_id,
                                          lookback_window=timedelta(hours=1), payload_validator=Verdict)
    v = _find(ev, Verdict)
    await db.proposals().update_one({"run_id": ctx.workflow_run_id},
                                    {"$set": {"status": "approved" if v.approved else "rejected", "note": v.note, "decided_at": now()}})
    return {"target_date": p.target_date, "bucket": p.bucket, "approved": v.approved}

def _find(obj, cls):
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, cls): return x
        if isinstance(x, dict): stack.extend(x.values())
        elif isinstance(x, (list, tuple)): stack.extend(x)
    raise ValueError(f"no {cls.__name__} in {obj!r}")

async def provisional_actual(date: str) -> dict | None:
    """Max hourly observation over the ET calendar day. Used only until NCEI publishes."""
    start = dt.datetime.fromisoformat(date).replace(tzinfo=ET); end = start + timedelta(days=1)
    temps = [o["temp_f"] async for o in db.observations().find({"ts": {"$gte": start.isoformat(), "$lt": end.isoformat()}})]
    return {"date": date, "tmax_f": round(max(temps))} if temps else None

# ---- crons --------------------------------------------------------------------------
@hatchet.task(name="poll_nws", on_crons=["*/15 * * * *"], execution_timeout=timedelta(minutes=2))
async def poll_nws(input, ctx: Context) -> dict:
    async with httpx.AsyncClient() as c:
        fc = await nws.hourly_forecast(c); t = tomorrow_et()
        await db.forecasts().insert_one({"fetched_at": now(), "target_date": t, "source": "nws", "periods": fc, "day_max": nws.day_max(fc, t)})
        for o in await nws.observations(c, 12):
            await db.observations().update_one({"ts": o["ts"]}, {"$set": o}, upsert=True)
    return {"target_date": t, "day_max": nws.day_max(fc, t)}

class ScoreInput(BaseModel):
    target_date: str | None = None   # default: the most recent unscored approved proposal

@hatchet.task(name="score_and_reflect", on_crons=["17 9 * * *"], execution_timeout=timedelta(minutes=10), input_validator=ScoreInput)
async def score_and_reflect(input: ScoreInput, ctx: Context) -> dict:
    # 1. refresh actuals from NCEI
    async with httpx.AsyncClient() as c:
        end = dt.date.today(); rows = await nws.ncei_tmax(c, (end - timedelta(days=7)).isoformat(), end.isoformat())
    for r in rows: await db.actuals().update_one({"date": r["date"]}, {"$set": r}, upsert=True)
    # 2. grade every approved proposal that has an actual and no score yet
    graded = []
    q = {"status": "approved"}; 
    if input.target_date: q["target_date"] = input.target_date
    async for p in db.proposals().find(q).sort("created_at", -1):
        if await db.scores().find_one({"target_date": p["target_date"]}): continue
        a = await db.actuals().find_one({"date": p["target_date"]})
        provisional = False
        if not a:   # NCEI lags a day; fall back to the max of that ET day's station observations
            a = await provisional_actual(p["target_date"]); provisional = a is not None
            if not a: continue
        hit = nws.bucket(a["tmax_f"]) == p["bucket"]
        s = {"target_date": p["target_date"], "bucket": p["bucket"], "point_f": p["point_f"], "actual_f": a["tmax_f"],
             "error_f": round(p["point_f"] - a["tmax_f"], 1), "hit": hit, "confidence": p["confidence"],
             "rules_version": p["rules_version"], "provisional": provisional, "scored_at": now()}
        await db.scores().insert_one(s); graded.append(s)
    if not graded: return {"graded": 0}
    # 3. reflect: agent reads scores + rules and rewrites the rulebook
    rules = await current_rules()
    scores = [s async for s in db.scores().find({}, {"_id": 0, "scored_at": 0}).sort("target_date", -1).limit(30)]
    agent = build_agent(ctx.lifespan["saver"], rules["text"])
    with tracer.start_as_current_span("bookie.reflect") as span:
        span.set_attribute("rules.version", rules["version"]); span.set_attribute("scores.n", len(scores))
        edit = await run_agent(agent,
            "Here are your graded calls, newest first (hit=True means the bucket was right):\n"
            + json.dumps(scores, indent=1) +
            "\n\nRewrite your full rulebook so future calls are better. Keep what works. Add concrete, "
            "testable lessons to '## Known behaviors' (e.g. 'NWS runs 1-2F warm on post-frontal days'). "
            "Do not invent lessons the scores don't support. Return the complete new rules document and a change summary.",
            thread_id=f"{ctx.workflow_run_id}:reflect", schema=RulesEdit)
    v = rules["version"] + 1
    await db.rules().insert_one({"version": v, "created_at": now(), "author": "agent", "text": edit.new_rules_md,
                                 "change_summary": edit.change_summary, "based_on_scores": [s["target_date"] for s in graded]})
    return {"graded": len(graded), "hits": sum(s["hit"] for s in graded), "rules_version": v, "change": edit.change_summary}

# ---- worker -------------------------------------------------------------------------
async def lifespan():
    await db.ensure_indexes()
    mc = MongoClient(os.environ["MONGODB_URI"])
    saver = MongoDBSaver(mc, db_name=os.environ.get("MONGODB_DB", "bookie"),
                         checkpoint_collection_name="lg_checkpoints", writes_collection_name="lg_writes")
    try:
        yield {"saver": saver}
    finally:
        mc.close()

if __name__ == "__main__":
    hatchet.worker("bookie-worker", workflows=[market_day, poll_nws, score_and_reflect], lifespan=lifespan).start()
