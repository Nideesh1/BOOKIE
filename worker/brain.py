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
from engine import risk
from engine.agents import EXEC_MEMORY_KEY, build_engine_agents, seed_execution_rules
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
VIEW_RULES_Q = {"kind": {"$ne": "execution"}}      # rules docs without a kind are the view rulebook (pre-3b)
EXEC_RULES_Q = {"kind": "execution"}


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
    change_summary: str = Field(description="1-3 sentences: what you changed in your rulebooks (AGENTS.md and/or EXECUTION.md) and which scores justify it")


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
    "graded experience; follow it. When asked to reflect, edit that file with edit_file, and edit the execution rulebook at "
    "/memories/EXECUTION.md from the graded trades (take-profit / cut lines under '## Position management', tactics, sizes). "
    "Before deciding, call search_past_reasoning with a short description of today's setup to see how similar past calls turned out."
)


def make_model() -> ChatOpenAI:
    return ChatOpenAI(model=os.environ["MODEL_NAME"], base_url=os.environ["MODEL_BASE_URL"],
                      api_key=os.environ["MODEL_API_KEY"], temperature=0, max_tokens=4096,
                      timeout=httpx.Timeout(180.0, connect=10.0), max_retries=1)


async def current_rules() -> dict:
    return await db.rules().find_one(VIEW_RULES_Q, sort=[("version", -1)])


async def current_exec_rules() -> dict | None:
    return await db.rules().find_one(EXEC_RULES_Q, sort=[("version", -1)])


async def exec_rules_from_store() -> str:
    item = await STORE.aget(MEMORY_NS, EXEC_MEMORY_KEY)
    return item.value.get("content", "") if item is not None else ""


async def ensure_exec_rules_doc(text: str) -> dict:
    """`rules` kind "execution" v1 mirrors the seeded EXECUTION.md so the UI and reflect have a versioned history."""
    doc = await current_exec_rules()
    if doc is None:
        doc = {"kind": "execution", "version": 1, "created_at": now(), "author": "human", "text": text,
               "change_summary": "seeded execution rulebook (engine/agents.py DEFAULT_EXEC_RULES)"}
        await db.rules().insert_one(dict(doc))
    return doc


async def version_exec_rules(before: dict | None, graded_trades: list[dict], change_summary: str) -> int:
    """Record the live EXECUTION.md as a new kind "execution" version when reflect changed it; restore the store's version key."""
    new_text = await exec_rules_from_store()
    v = (before or {}).get("version", 0)
    if new_text.strip() and new_text.strip() != ((before or {}).get("text") or "").strip():
        v += 1
        await db.rules().insert_one({"kind": "execution", "version": v, "created_at": now(), "author": "agent", "text": new_text,
                                     "change_summary": change_summary,
                                     "based_on_trades": [f"{t['target_date']}:{t['ticker']}" for t in graded_trades]})
    item = await STORE.aget(MEMORY_NS, EXEC_MEMORY_KEY)
    if item is not None:
        await STORE.aput(MEMORY_NS, EXEC_MEMORY_KEY, {**item.value, "version": v, "modified_at": now().isoformat()})
    return v


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
            rules = {"kind": "view", "version": rules["version"] + 1, "created_at": now(), "author": "agent", "text": live,
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


def build_agent(saver, response_format=None, memory: list[str] | None = None):
    """Called ONCE per process from lifespan(). Tasks reach the compiled graphs via ctx.lifespan."""
    backend = CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend(namespace=lambda rt: MEMORY_NS, store=STORE)})
    return create_deep_agent(
        model=make_model(), tools=TOOLS, system_prompt=SYSTEM, backend=backend, store=STORE,
        memory=memory or ["/memories/AGENTS.md"], checkpointer=saver,
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


async def _actual_for(date: str) -> tuple[dict | None, bool]:
    a = await db.actuals().find_one({"date": date})
    if a:
        return a, False
    a = await provisional_actual(date)
    return a, a is not None


async def grade_trades(target_date: str | None = None) -> list[dict]:
    """Phase 3b: one `trade_scores` doc per (target_date, ticker) once the day's high is known. Realized P&L on closed
    round trips (fills vs fills, YES-leg average cost), settlement P&L on what was still held (100¢ if the bucket hit
    else 0, minus fees from fills), and the orders that never filled. Idempotent; provisional rows are re-scored
    when NCEI publishes."""
    q: dict = {"status": {"$in": ["resting", "executed", "canceled"]}}
    if target_date:
        q["target_date"] = target_date
    by_key: dict[tuple[str, str], list[dict]] = {}
    async for o in db.orders().find(q):
        by_key.setdefault((o["target_date"], o["ticker"]), []).append(o)
    out = []
    for (date, ticker), docs in by_key.items():
        prev = await db.trade_scores().find_one({"target_date": date, "ticker": ticker})
        if prev and not prev.get("provisional"):
            continue
        a, provisional = await _actual_for(date)
        if not a:
            continue
        legs, fees, never = [], 0.0, []
        for o in sorted(docs, key=lambda d: str(d.get("created_at"))):
            fl = o.get("fills") or []
            if not fl:
                never.append({"client_order_id": o.get("client_order_id"), "status": o.get("status"), "side": o.get("side"),
                              "count": o.get("count"), "price_c": o.get("price_c"), "tactic": o.get("tactic")})
                continue
            for f in sorted(fl, key=lambda f: str(f.get("created_time") or "")):
                leg = risk.fill_yes_leg(f, o)
                if leg:
                    legs.append(leg)
                fees += float(f.get("fee_cost") or 0)
            if not any(f.get("fee_cost") is not None for f in fl):
                fees += float(o.get("fees_usd") or 0)
        bucket = docs[0].get("bucket")
        hit = nws.bucket(a["tmax_f"]) == bucket if bucket else None
        if hit is None:
            try:
                snap = await db.market_snapshots().find_one({"target_date": date, "markets.ticker": ticker}, {"markets.$": 1})
                m = (snap or {}).get("markets", [{}])[0]
                hit = nws.bucket(a["tmax_f"]) == (m.get("label") or bucket)
            except Exception:
                hit = False
        led = risk.ledger_from_fills(legs)
        settle_c = risk.settlement_c(led["open"], led["avg_yes_c"], bool(hit))
        s = {"target_date": date, "ticker": ticker, "bucket": bucket, "actual_f": a["tmax_f"], "bucket_hit": bool(hit),
             "fills_n": len(legs), "realized_c": led["realized_c"], "open_at_settlement": led["open"], "avg_yes_c": led["avg_yes_c"],
             "settlement_c": settle_c, "fees_usd": round(fees, 4), "net_usd": round((led["realized_c"] + settle_c) / 100 - fees, 4),
             "never_filled": never, "outcome": ("never_filled" if not legs else "round_trip" if led["open"] == 0 else "held_to_settlement"),
             "provisional": provisional, "scored_at": now()}
        await db.trade_scores().replace_one({"target_date": date, "ticker": ticker}, s, upsert=True)
        out.append(s)
    return out


def _trade_summary(scores: list[dict]) -> list[dict]:
    return [{"date": t["target_date"], "bucket": t.get("bucket"), "outcome": t.get("outcome"), "hit": t.get("bucket_hit"),
             "fills": t.get("fills_n"), "realized_c": t.get("realized_c"), "settlement_c": t.get("settlement_c"),
             "fees_usd": t.get("fees_usd"), "net_usd": t.get("net_usd"), "never_filled": len(t.get("never_filled") or [])} for t in scores]


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
    graded_trades = await grade_trades(input.target_date)          # phase 3b: fills vs fills, settlement, never filled
    if not graded and not graded_trades:
        return {"graded": 0, "trades_graded": 0}

    rules = await load_rules_into_store(await current_rules())
    exec_before = await ensure_exec_rules_doc(await exec_rules_from_store())
    scores = [s async for s in db.scores().find({}, {"_id": 0, "scored_at": 0}).sort("target_date", -1).limit(30)]
    trades = [t async for t in db.trade_scores().find({}, {"_id": 0}).sort([("target_date", -1), ("scored_at", -1)]).limit(30)]
    agent = ctx.lifespan["reflector"]
    with tracer.start_as_current_span("bookie.reflect") as span:
        span.set_attribute("rules.version", rules["version"])
        span.set_attribute("exec_rules.version", exec_before.get("version", 0))
        span.set_attribute("scores.n", len(scores))
        span.set_attribute("trades.n", len(trades))
        out = await run_agent(agent,
            "Your graded calls, newest first (hit=true means the bucket was right):\n" + json.dumps(scores, indent=1, default=str) +
            "\n\nYour graded trades, newest first (realized_c = closed round trips, settlement_c = what was held to settlement, "
            "net_usd after fees; never_filled = resting orders that never traded):\n" + json.dumps(_trade_summary(trades), indent=1, default=str) +
            "\n\nUpdate your rulebook at /memories/AGENTS.md with edit_file so future calls are better. Keep what works. "
            "Add concrete, testable lessons under '## Known behaviors'. Do not invent lessons the scores don't support. "
            "Then, if the graded trades support it, update your execution rulebook at /memories/EXECUTION.md with edit_file: "
            "the take-profit and cut lines under '## Position management', tactics, sizes, and what never fills. "
            "Hard risk limits are code and not yours to change. Then answer with a short change summary.",
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
        await db.rules().insert_one({"kind": "view", "version": v, "created_at": now(), "author": "agent", "text": new_text,
                                     "change_summary": edit.change_summary, "based_on_scores": [s["target_date"] for s in graded]})
    else:
        v = rules["version"]
    await bump_store_version(v)   # edit_file drops the version key; restore it so the next load doesn't re-seed
    ev = await version_exec_rules(exec_before, graded_trades, edit.change_summary)
    return {"graded": len(graded), "hits": sum(s["hit"] for s in graded), "rules_version": v, "exec_rules_version": ev,
            "trades_graded": len(graded_trades), "trade_net_usd": round(sum(t["net_usd"] for t in graded_trades), 4),
            "change": edit.change_summary}


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
    reflector = build_agent(saver, RulesEdit, memory=["/memories/AGENTS.md", "/memories/EXECUTION.md"])   # edits both rulebooks
    await ensure_exec_rules_doc(await seed_execution_rules(STORE))
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
