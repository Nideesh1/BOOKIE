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
from hatchet_sdk import Context, DurableContext, EmptyModel, Hatchet
from hatchet_sdk.opentelemetry.instrumentor import HatchetInstrumentor
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic import BaseModel, Field
from pymongo import MongoClient

import db
import intraday
import jev
import memory_search
import nws
from mongo_store import MongoStore
from bus import broker, publish
from streams import OUT_PROPOSAL
from engine.agents import build_engine_agents, seed_execution_rules
from exchange import KalshiClient
from engine.workflow import build_market_view, build_sync_orders

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
    """Recent Central Park observations, newest first. Timestamps are UTC. Convert to America/New_York (EDT = UTC-4 now, EST = UTC-5 in winter) before attributing a reading to a day."""
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


@tool
async def search_past_reasoning(query: str) -> list[dict] | dict:
    """Find your own past calls on days like this one (semantic search over your rationales) and whether they hit.
    Describe the setup in words, e.g. "warm front, NWS says 78, market favors 76-77, morning obs running cool"."""
    return await memory_search.search_similar(query, k=5)


TOOLS = [get_forecast, get_observations, get_market, get_my_scores, get_recent_actuals, search_past_reasoning]

# ---- agent -----------------------------------------------------------------------------
SYSTEM = (
    "You are bookie, a disciplined forecaster of the official daily high temperature at Central Park, NYC. "
    "Use tools; never guess numbers you could look up. Observations are UTC; the local day is America/New_York (currently EDT, UTC-4): 19:51Z is 3:51pm ET. "
    "The official daily high is the max over the local calendar day (midnight to midnight New York time). Your rulebook at /memories/AGENTS.md was written by you from "
    "graded experience; follow it. When asked to reflect, edit that file with edit_file. "
    "Before deciding, call search_past_reasoning with a short description of today's setup to see how similar past calls turned out."
)


def make_model() -> ChatOpenAI:
    return ChatOpenAI(model=os.environ["MODEL_NAME"], base_url=os.environ["MODEL_BASE_URL"],
                      api_key=os.environ["MODEL_API_KEY"], temperature=0, max_tokens=4096,
                      timeout=httpx.Timeout(180.0, connect=10.0), max_retries=1)


async def current_rules() -> dict:
    return await db.rules().find_one({}, sort=[("version", -1)])


# Durable store in Atlas (collection lg_store). The LIVE rulebook lives here; `rules` is the versioned history.
STORE = MongoStore(os.environ["MONGODB_URI"], os.environ.get("MONGODB_DB", "bookie"))


async def load_rules_into_store(rules: dict) -> dict:
    """Seed/refresh the live copy only when it is missing or older than the latest `rules` version.

    StoreBackend.edit_file rewrites the value without our `version` key, so a live copy with no version
    means the agent edited it outside a reflect cycle: the store is truth, so record it as a new `rules`
    version instead of overwriting it. Returns the (possibly new) latest rules doc.
    """
    item = await STORE.aget(MEMORY_NS, MEMORY_KEY)
    if item is not None and "version" not in item.value:
        live = item.value["content"]
        if live.strip() != rules["text"].strip():
            rules = {"version": rules["version"] + 1, "created_at": now(), "author": "agent", "text": live,
                     "change_summary": "Live rulebook edited outside a reflect cycle; recorded from the store."}
            await db.rules().insert_one(dict(rules))
        await bump_store_version(rules["version"])
        return rules
    if item is not None and item.value.get("version", -1) >= rules["version"]:
        return rules
    ts = now().isoformat()
    await STORE.aput(MEMORY_NS, MEMORY_KEY, {"content": rules["text"], "encoding": "utf-8", "created_at": ts,
                                             "modified_at": ts, "version": rules["version"]})
    return rules


async def rules_from_store() -> str:
    return (await STORE.aget(MEMORY_NS, MEMORY_KEY)).value["content"]


async def bump_store_version(version: int) -> None:
    item = await STORE.aget(MEMORY_NS, MEMORY_KEY)
    await STORE.aput(MEMORY_NS, MEMORY_KEY, {**item.value, "version": version, "modified_at": now().isoformat()})


def build_agent(saver, response_format=None):
    """Called ONCE per process from lifespan(). Tasks reach the compiled graphs via ctx.lifespan."""
    backend = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend(namespace=lambda rt: MEMORY_NS, store=STORE)})
    return create_deep_agent(
        model=make_model(), tools=TOOLS, system_prompt=SYSTEM, backend=backend, store=STORE,
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
market_view = build_market_view(hatchet)   # phase 2/3: tick_and_gate -> form_view -> decide -> gate -> place (engine/workflow.py)
sync_orders_cron = build_sync_orders(hatchet)   # phase 3: every 2 min pull orders + fills into db.orders()


@market_day.task(execution_timeout=timedelta(minutes=10), retries=1)
async def propose(input: DayInput, ctx: Context) -> dict:
    target = input.target_date or tomorrow_et()
    rules = await load_rules_into_store(await current_rules())
    agent = ctx.lifespan["proposer"]
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
    await memory_search.index_reasoning(doc)
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
        await memory_search.mark_scored(s["target_date"], s["hit"], s["actual_f"])
        graded.append(s)
    if not graded:
        return {"graded": 0}

    rules = await load_rules_into_store(await current_rules())
    scores = [s async for s in db.scores().find({}, {"_id": 0, "scored_at": 0}).sort("target_date", -1).limit(30)]
    agent = ctx.lifespan["reflector"]
    with tracer.start_as_current_span("bookie.reflect") as span:
        span.set_attribute("rules.version", rules["version"])
        span.set_attribute("scores.n", len(scores))
        out = await run_agent(agent,
            "Your graded calls, newest first (hit=true means the bucket was right):\n" + json.dumps(scores, indent=1, default=str) +
            "\n\nUpdate your rulebook at /memories/AGENTS.md with edit_file so future calls are better. Keep what works. "
            "Add concrete, testable lessons under '## Known behaviors'. Do not invent lessons the scores don't support. "
            "Then answer with a short change summary.",
            thread_id=f"{ctx.workflow_run_id}:reflect")
    edit = out.get("structured_response")
    if edit is None:   # agent edited the file but answered in prose; keep the prose as the summary
        last = out["messages"][-1].content
        if isinstance(last, list):
            last = " ".join(b.get("text", "") for b in last if isinstance(b, dict))
        edit = RulesEdit(change_summary=(last or "").strip()[:600])
    new_text = await rules_from_store()
    if not edit.change_summary:   # model edited the file silently; summarize the diff ourselves
        import difflib
        added = [l[1:].strip() for l in difflib.unified_diff(rules["text"].splitlines(), new_text.splitlines(), lineterm="", n=0)
                 if l.startswith("+") and not l.startswith("+++") and l[1:].strip()]
        edit.change_summary = ("Rulebook edited by the agent. Added: " + " | ".join(added)[:520]) if added else "No textual change."
    v = rules["version"] + 1
    if new_text.strip() != rules["text"].strip():
        await db.rules().insert_one({"version": v, "created_at": now(), "author": "agent", "text": new_text,
                                     "change_summary": edit.change_summary, "based_on_scores": [s["target_date"] for s in graded]})
    else:
        v = rules["version"]
    await bump_store_version(v)   # edit_file drops the version key; restore it so the next load doesn't re-seed
    return {"graded": len(graded), "hits": sum(s["hit"] for s in graded), "rules_version": v, "change": edit.change_summary}


# ---- intraday gap watch: code only, no LLM, no orders ---------------------------------
def today_et() -> str:
    return dt.datetime.now(ET).date().isoformat()


@hatchet.task(name="intraday_watch", on_crons=["*/5 * * * *"], execution_timeout=timedelta(minutes=2))
async def intraday_watch(input: EmptyModel, ctx: Context) -> dict:
    """Re-estimate today's daily-high bucket probabilities from running max + remaining-day forecast, compare to the
    live book, and record the gaps. The 5-min cron is the safety net; the worker also triggers it on market ticks."""
    target = today_et()
    with tracer.start_as_current_span("bookie.intraday_watch") as span:
        span.set_attribute("target_date", target)
        est = await intraday.estimate_today(target)
        await db.gaps().insert_one({**est, "trigger": "hatchet"})
        span.set_attribute("running_max", est["running_max"] or -1)
        span.set_attribute("hours_left", est["hours_left"])
    top = sorted(est["buckets"], key=lambda b: abs(b["edge_cents"] or 0), reverse=True)[:3]
    return {"target_date": target, "as_of": est["as_of"].isoformat(), "running_max": est["running_max"],
            "remaining_forecast_max": est["remaining_forecast_max"], "hours_left": est["hours_left"],
            "favorite_market": est["favorite_bucket_market"], "favorite_model": est["favorite_bucket_model"],
            "biggest_gaps": [{"label": b["label"], "p_model": b["p_model"], "mid": b["mid"], "edge_cents": b["edge_cents"]} for b in top]}


# ---- worker ----------------------------------------------------------------------------
async def lifespan():
    await db.ensure_indexes()
    await memory_search.ensure_vector_index()
    await broker.connect()
    mc = MongoClient(os.environ["MONGODB_URI"])
    saver = MongoDBSaver(mc, db_name=os.environ.get("MONGODB_DB", "bookie"),
                         checkpoint_collection_name="lg_checkpoints", writes_collection_name="lg_writes")
    proposer = build_agent(saver, Proposal)     # compiled once per process
    reflector = build_agent(saver, RulesEdit)
    await seed_execution_rules(STORE)
    engine = build_engine_agents(saver, STORE)   # phase 2 agents, compiled once
    exchange = None                              # phase 3: one KalshiClient per process; dry_run unless ORDERS_ENABLED=true
    try:
        exchange = KalshiClient.from_env()
        print(f"exchange client ready: {exchange!r}")
    except (KeyError, OSError, ValueError) as e:
        print(f"exchange client NOT configured ({type(e).__name__}); order tasks will skip")
    try:
        yield {"saver": saver, "proposer": proposer, "reflector": reflector, "engine": engine, "exchange": exchange}
    finally:
        if exchange is not None:
            await exchange.aclose()
        mc.close()
        await STORE.aclose()
        await broker.stop()


if __name__ == "__main__":
    hatchet.worker("bookie-brain", workflows=[market_day, score_and_reflect, intraday_watch, market_view, sync_orders_cron], lifespan=lifespan).start()
