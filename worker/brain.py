"""bookie brain — Hatchet durable workflows around a deepagents forecaster.

propose (deepagents, GPT-5.6 Luna) -> gate (Jev auto-approve or wait for a human) -> nightly score -> reflect
(the agent rewrites its own rulebook from its graded history). Every read and write goes to MongoDB Atlas.
Run: uv run python brain.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
from datetime import timedelta

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
os.environ.setdefault("LANGSMITH_TRACING", "false")

import httpx
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from hatchet_sdk import Context, DurableContext, Hatchet
from hatchet_sdk.opentelemetry.instrumentor import HatchetInstrumentor
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.memory import InMemoryStore
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, Field
from pymongo import MongoClient

import db
import jev
import nws
from bus import broker, publish
from streams import OUT_PROPOSAL

# ---- tracing: plain OTLP -> Langfuse -------------------------------------------------
provider = TracerProvider(resource=Resource.create({"service.name": "bookie-brain"}))
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)
hatchet = Hatchet()
HatchetInstrumentor(tracer_provider=provider, enable_hatchet_otel_collector=False).instrument()
LangChainInstrumentor().instrument(tracer_provider=provider)

ET = nws.ET
AUTO_APPROVE_THRESHOLD = float(os.environ.get("AUTO_APPROVE_THRESHOLD", "0.8"))
MEMORY_NS = ("bookie", "rules")
MEMORY_KEY = "/AGENTS.md"


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def tomorrow_et() -> str:
    return (dt.datetime.now(ET).date() + timedelta(days=1)).isoformat()


# ---- typed outputs ---------------------------------------------------------------------
class Proposal(BaseModel):
    bucket: str = Field(description='2-degree bucket for the official daily high, e.g. "62-63"')
    point_f: float = Field(description="best single estimate of the daily high in F")
    confidence: float = Field(ge=0, le=1)
    biggest_risk: str
    rationale: str


class RulesEdit(BaseModel):
    change_summary: str = Field(description="1-3 sentences: what you changed in your rulebook and which scores justify it")


# ---- tools (read-only views over Atlas) ------------------------------------------------
@tool
async def get_forecast(target_date: str) -> dict:
    """Latest NWS hourly forecast for target_date (YYYY-MM-DD, ET): day max and the hourly curve."""
    doc = await db.forecasts().find_one({"target_date": target_date}, sort=[("fetched_at", -1)])
    if not doc:
        return {"error": "no forecast stored for that date"}
    hours = [p for p in doc["periods"] if p["t"].startswith(target_date)]
    return {"fetched_at": doc["fetched_at"].isoformat(), "day_max_f": doc["day_max"],
            "hourly": [{"h": p["t"][11:13], "f": p["temp_f"], "pop": p.get("pop"), "sky": p.get("short")} for p in hours]}


@tool
async def get_observations(hours: int = 24) -> list[dict]:
    """Recent Central Park observations, newest first. Timestamps are UTC; ET = UTC-4."""
    cur = db.observations().find({}, {"_id": 0}).sort("ts", -1).limit(hours)
    return [{"ts": o["ts"].isoformat() if hasattr(o["ts"], "isoformat") else o["ts"], "temp_f": o["temp_f"]} async for o in cur]


@tool
async def get_market(target_date: str) -> dict:
    """Latest prediction-market book for target_date: each bucket's yes bid/ask, and the crowd favorite."""
    s = await db.market_snapshots().find_one({"target_date": target_date}, sort=[("ts", -1)])
    if not s:
        return {"error": "no market snapshot"}
    return {"ts": s["ts"].isoformat(), "favorite_bucket": s.get("favorite_bucket"), "favorite_mid": s.get("favorite_mid"),
            "buckets": [{"label": m["label"], "bid": m["yes_bid"], "ask": m["yes_ask"]} for m in s["markets"]]}


@tool
async def get_my_scores(last_n: int = 14) -> list[dict]:
    """Your graded calls, newest first: bucket vs official high, error, hit. Read-only; written by the scorer."""
    cur = db.scores().find({}, {"_id": 0, "scored_at": 0}).sort("target_date", -1).limit(last_n)
    return [s async for s in cur]


@tool
async def get_recent_actuals(last_n: int = 10) -> list[dict]:
    """Official NCEI daily highs for recent days."""
    cur = db.actuals().find({}, {"_id": 0}).sort("date", -1).limit(last_n)
    return [a async for a in cur]


TOOLS = [get_forecast, get_observations, get_market, get_my_scores, get_recent_actuals]

# ---- agent -----------------------------------------------------------------------------
SYSTEM = (
    "You are bookie, a disciplined forecaster of the official daily high temperature at Central Park, NYC. "
    "Use tools; never guess numbers you could look up. Observations are UTC: a reading at 19:51Z is 3:51pm ET. "
    "The ET calendar day runs 04:00Z to 04:00Z. Your rulebook at /memories/AGENTS.md was written by you from "
    "graded experience; follow it. When asked to reflect, edit that file with edit_file."
)


def make_model() -> ChatOpenAI:
    return ChatOpenAI(model=os.environ["MODEL_NAME"], base_url=os.environ["MODEL_BASE_URL"],
                      api_key=os.environ["MODEL_API_KEY"], temperature=0, max_tokens=4096,
                      timeout=httpx.Timeout(180.0, connect=10.0), max_retries=1)


async def current_rules() -> dict:
    return await db.rules().find_one({}, sort=[("version", -1)])


def seeded_store(rules_text: str) -> InMemoryStore:
    store = InMemoryStore()
    ts = now().isoformat()
    store.put(MEMORY_NS, MEMORY_KEY, {"content": rules_text, "encoding": "utf-8", "created_at": ts, "modified_at": ts})
    return store


def build_agent(saver, store: InMemoryStore, response_format=None):
    backend = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend(namespace=lambda rt: MEMORY_NS, store=store)})
    return create_deep_agent(
        model=make_model(), tools=TOOLS, system_prompt=SYSTEM, backend=backend, store=store,
        memory=["/memories/AGENTS.md"], checkpointer=saver,
        middleware=[ModelCallLimitMiddleware(run_limit=15, exit_behavior="end")],
        response_format=ToolStrategy(response_format) if response_format else None,
    )


async def run_agent(agent, prompt: str, thread_id: str):
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    return await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}, cfg)


# ---- market_day workflow: propose -> gate --------------------------------------------
class DayInput(BaseModel):
    target_date: str | None = None


class Verdict(BaseModel):
    approved: bool
    note: str = ""


VERDICT_EVENT = "bookie:verdict"
market_day = hatchet.workflow(name="market_day", input_validator=DayInput)


@market_day.task(execution_timeout=timedelta(minutes=10), retries=1)
async def propose(input: DayInput, ctx: Context) -> dict:
    target = input.target_date or tomorrow_et()
    rules = await current_rules()
    agent = build_agent(ctx.lifespan["saver"], seeded_store(rules["text"]), Proposal)
    with tracer.start_as_current_span("bookie.propose") as span:
        span.set_attribute("target_date", target)
        span.set_attribute("rules.version", rules["version"])
        out = await run_agent(agent, f"Forecast the official daily high at Central Park for {target}. Follow your rules. "
                                     "Check the forecast, recent observations, the market, and your own scores before answering.",
                              thread_id=f"{ctx.workflow_run_id}:propose")
    p: Proposal = out["structured_response"]
    doc = {**p.model_dump(), "target_date": target, "run_id": ctx.workflow_run_id, "rules_version": rules["version"],
           "model": os.environ["MODEL_NAME"], "created_at": now(), "status": "pending"}
    await db.proposals().insert_one(doc)
    await publish(OUT_PROPOSAL, {k: v for k, v in doc.items() if k != "_id"})
    return {**p.model_dump(), "target_date": target}


@market_day.durable_task(parents=[propose], execution_timeout=timedelta(days=2), schedule_timeout=timedelta(days=2))
async def gate(input: DayInput, ctx: DurableContext) -> dict:
    p = ctx.task_output(propose)
    target = p["target_date"]
    fc = await db.forecasts().find_one({"target_date": target}, sort=[("fetched_at", -1)])
    mk = await db.market_snapshots().find_one({"target_date": target}, sort=[("ts", -1)])
    recent = [s async for s in db.scores().find({}, {"_id": 0, "hit": 1}).sort("target_date", -1).limit(10)]
    state = {"proposal": {k: p[k] for k in ("bucket", "point_f", "confidence", "biggest_risk", "rationale")},
             "nws_day_max_f": fc["day_max"] if fc else None,
             "market_favorite": mk.get("favorite_bucket") if mk else None, "market_favorite_mid": mk.get("favorite_mid") if mk else None,
             "recent_hits": sum(1 for s in recent if s.get("hit")), "recent_calls": len(recent)}
    with tracer.start_as_current_span("bookie.gate") as span:
        prob = await jev.auto_approve_probability(state)
        span.set_attribute("jev.auto_ok", prob)
        auto = prob >= AUTO_APPROVE_THRESHOLD
        span.set_attribute("gate.auto", auto)
        await db.proposals().update_one({"run_id": ctx.workflow_run_id}, {"$set": {"jev_auto_ok": prob}})
        if auto:
            await db.proposals().update_one({"run_id": ctx.workflow_run_id},
                                            {"$set": {"status": "approved", "note": f"auto (jev {prob:.2f})", "decided_at": now()}})
            return {"target_date": target, "bucket": p["bucket"], "approved": True, "by": "jev", "p": prob}
        with tracer.start_as_current_span("bookie.human_gate"):
            ev = await ctx.aio_wait_for_event(VERDICT_EVENT, scope=ctx.workflow_run_id,
                                              lookback_window=timedelta(hours=1), payload_validator=Verdict)
    v = _find(ev, Verdict)
    await db.proposals().update_one({"run_id": ctx.workflow_run_id},
                                    {"$set": {"status": "approved" if v.approved else "rejected", "note": v.note, "decided_at": now()}})
    return {"target_date": target, "bucket": p["bucket"], "approved": v.approved, "by": "human", "p": prob}


def _find(obj, cls):
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, cls):
            return x
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
    raise ValueError(f"no {cls.__name__} in {obj!r}")


# ---- nightly: score, then reflect ------------------------------------------------------
class ScoreInput(BaseModel):
    target_date: str | None = None


async def provisional_actual(date: str) -> dict | None:
    """Max hourly observation over the ET day. Used until NCEI publishes the official number."""
    start = dt.datetime.fromisoformat(date).replace(tzinfo=ET)
    end = start + timedelta(days=1)
    temps = [o["temp_f"] async for o in db.observations().find({"ts": {"$gte": start, "$lt": end}})]
    return {"date": date, "tmax_f": round(max(temps))} if temps else None


@hatchet.task(name="score_and_reflect", on_crons=["17 13 * * *"], execution_timeout=timedelta(minutes=10), input_validator=ScoreInput)
async def score_and_reflect(input: ScoreInput, ctx: Context) -> dict:
    q: dict = {"status": "approved"}
    if input.target_date:
        q["target_date"] = input.target_date
    graded = []
    async for p in db.proposals().find(q).sort("created_at", -1):
        if await db.scores().find_one({"target_date": p["target_date"]}):
            continue
        a = await db.actuals().find_one({"date": p["target_date"]})
        provisional = False
        if not a:
            a = await provisional_actual(p["target_date"])
            provisional = a is not None
            if not a:
                continue
        s = {"target_date": p["target_date"], "bucket": p["bucket"], "point_f": p["point_f"], "actual_f": a["tmax_f"],
             "error_f": round(p["point_f"] - a["tmax_f"], 1), "hit": nws.bucket(a["tmax_f"]) == p["bucket"],
             "confidence": p["confidence"], "rules_version": p["rules_version"], "provisional": provisional, "scored_at": now()}
        await db.scores().insert_one(s)
        graded.append(s)
    if not graded:
        return {"graded": 0}

    rules = await current_rules()
    scores = [s async for s in db.scores().find({}, {"_id": 0, "scored_at": 0}).sort("target_date", -1).limit(30)]
    store = seeded_store(rules["text"])
    agent = build_agent(ctx.lifespan["saver"], store, RulesEdit)
    with tracer.start_as_current_span("bookie.reflect") as span:
        span.set_attribute("rules.version", rules["version"])
        span.set_attribute("scores.n", len(scores))
        out = await run_agent(agent,
            "Your graded calls, newest first (hit=true means the bucket was right):\n" + json.dumps(scores, indent=1, default=str) +
            "\n\nUpdate your rulebook at /memories/AGENTS.md with edit_file so future calls are better. Keep what works. "
            "Add concrete, testable lessons under '## Known behaviors'. Do not invent lessons the scores don't support. "
            "Then answer with a short change summary.",
            thread_id=f"{ctx.workflow_run_id}:reflect")
    edit: RulesEdit = out["structured_response"]
    new_text = store.get(MEMORY_NS, MEMORY_KEY).value["content"]
    v = rules["version"] + 1
    if new_text.strip() != rules["text"].strip():
        await db.rules().insert_one({"version": v, "created_at": now(), "author": "agent", "text": new_text,
                                     "change_summary": edit.change_summary, "based_on_scores": [s["target_date"] for s in graded]})
    else:
        v = rules["version"]
    return {"graded": len(graded), "hits": sum(s["hit"] for s in graded), "rules_version": v, "change": edit.change_summary}


# ---- worker ----------------------------------------------------------------------------
async def lifespan():
    await db.ensure_indexes()
    await broker.connect()
    mc = MongoClient(os.environ["MONGODB_URI"])
    saver = MongoDBSaver(mc, db_name=os.environ.get("MONGODB_DB", "bookie"),
                         checkpoint_collection_name="lg_checkpoints", writes_collection_name="lg_writes")
    try:
        yield {"saver": saver}
    finally:
        mc.close()
        await broker.stop()


if __name__ == "__main__":
    hatchet.worker("bookie-brain", workflows=[market_day, score_and_reflect], lifespan=lifespan).start()
