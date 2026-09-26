"""Hatchet `market_view` workflow (docs/ENGINE.md): tick_and_gate -> form_view -> decide -> gate.

build_market_view(hatchet) returns the workflow so brain.py can register it on its own Hatchet client without a
circular import. Agents come from ctx.lifespan["engine"] (compiled once in brain.lifespan()). Code (engine.state,
engine.clamps) owns every number; Jev (engine.jev_questions) owns the cheap typed decisions; the agents own judgment.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from datetime import timedelta

from hatchet_sdk import Context, DurableContext, EmptyModel
from langchain_core.messages import ToolMessage
from opentelemetry import trace
from pydantic import BaseModel, ValidationError

import db
import memory_search
import nws
from bus import publish
from engine import jev_questions, state as engine_state
from engine.clamps import Caps, clamp
from engine.contracts import MarketView, OrderProposal, View, WeatherView
from engine.tools import exposure_today
from engine import execute
from streams import OUT_PROPOSAL

tracer = trace.get_tracer("bookie.engine")
ET = nws.ET
VERDICT_EVENT = "bookie:verdict"
RETHINK_THRESHOLD = 0.5
SAFE_THRESHOLD = float(os.environ.get("AUTO_APPROVE_THRESHOLD", str(jev_questions.SAFE_THRESHOLD)))


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def today_et() -> str:
    return dt.datetime.now(ET).date().isoformat()


def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


class ViewInput(BaseModel):
    target_date: str | None = None
    force: bool = False          # bypass the Jev re-think gate (testing / manual kick)


class Verdict(BaseModel):
    approved: bool
    note: str = ""


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


def _subagent_views(messages) -> tuple[dict | None, dict | None]:
    """Pull the WeatherView / MarketView JSON out of the `task` ToolMessages the main agent received (last one wins)."""
    wx = mk = None
    for m in messages:
        if not isinstance(m, ToolMessage) or not isinstance(m.content, str) or not m.content.lstrip().startswith("{"):
            continue
        try:
            d = json.loads(m.content)
        except ValueError:
            continue
        try:
            WeatherView.model_validate(d); wx = d; continue
        except ValidationError:
            pass
        try:
            MarketView.model_validate(d); mk = d
        except ValidationError:
            pass
    return wx, mk


async def index_view_reasoning(view_doc: dict) -> bool:
    """memory_search.index_reasoning expects the phase-1 proposal shape; map the View onto it (bucket = argmax p)."""
    pb = view_doc.get("p_by_bucket") or {}
    top = max(pb.items(), key=lambda kv: kv[1])[0] if pb else None
    return await memory_search.index_reasoning({
        "run_id": view_doc["run_id"], "target_date": view_doc["target_date"], "bucket": top, "point_f": None,
        "confidence": view_doc["confidence"], "rationale": view_doc["rationale"],
        "biggest_risk": view_doc.get("what_would_change_my_mind", "")})


async def run_agent(agent, prompt: str, thread_id: str, recursion_limit: int = 80):
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    return await agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}, cfg)


# ---- workflow --------------------------------------------------------------------------
def build_market_view(hatchet):
    market_view = hatchet.workflow(name="market_view", input_validator=ViewInput)

    @market_view.task(execution_timeout=timedelta(minutes=2), retries=1)
    async def tick_and_gate(input: ViewInput, ctx: Context) -> dict:
        """CODE tick_state -> JEV 're-think?'. Skips unless p >= 0.5, a boundary was crossed, the last view is stale, or forced."""
        target = input.target_date or today_et()
        with tracer.start_as_current_span("bookie.tick_and_gate") as span:
            span.set_attribute("target_date", target)
            tick = await engine_state.tick_state(target)
            await engine_state.record_tick(tick)
            p = await jev_questions.should_rethink(tick, tick.last_view_age_s)
            span.set_attribute("jev.rethink", p)
            span.set_attribute("hours_left", tick.hours_left)
            fresh = tick.last_view_age_s is not None and tick.last_view_age_s < jev_questions.RETHINK_COOLDOWN_S
            skip = (not input.force) and p < RETHINK_THRESHOLD and not tick.boundary_crossed and fresh
            span.set_attribute("gate.skip", skip)
        return {"skip": skip, "target_date": target, "rethink_p": p, "tick": tick.model_dump(mode="json")}

    @market_view.task(parents=[tick_and_gate], execution_timeout=timedelta(minutes=10), retries=1)
    async def form_view(input: ViewInput, ctx: Context) -> dict:
        """Main agent (weather + market subagents inside) -> View. Persist to `views`, embed the rationale, publish."""
        t = ctx.task_output(tick_and_gate)
        if t["skip"]:
            return {"skip": True, "target_date": t["target_date"]}
        target, tick = t["target_date"], t["tick"]
        agent = ctx.lifespan["engine"]["main"]
        with tracer.start_as_current_span("bookie.form_view") as span:
            span.set_attribute("target_date", target)
            span.set_attribute("hours_left", tick["hours_left"])
            out = await run_agent(agent,
                f"Form your View for the official daily high at Central Park on {target}. Current TickState (code-computed):\n"
                + json.dumps(tick, indent=1) +
                "\n\nDelegate to the 'weather' and 'market' subagents first (pass the target date), read your rulebook, call "
                f"gap_table, then return the View with target_date='{target}', as_of='{tick['as_of']}', hours_left={tick['hours_left']}.",
                thread_id=f"{ctx.workflow_run_id}:view")
            view: View | None = out.get("structured_response")
            if view is None:
                raise RuntimeError("main agent returned no structured View (hit the call limit?)")
            wx, mk = _subagent_views(out["messages"])
            span.set_attribute("view.confidence", view.confidence)
            span.set_attribute("view.believed", sum(g.believed for g in view.gaps))
            span.set_attribute("view.has_weather", wx is not None)
            span.set_attribute("view.has_market", mk is not None)
        doc = {**view.model_dump(), "run_id": ctx.workflow_run_id, "tick": tick, "weather_view": wx, "market_view": mk,
               "model": os.environ["MODEL_NAME"], "created_at": now(), "kind": "view"}
        await db.views().insert_one(doc)
        await index_view_reasoning(doc)
        await publish(OUT_PROPOSAL, {k: v for k, v in _jsonable(doc).items() if k not in ("_id", "tick", "weather_view", "market_view")})
        return {"skip": False, "target_date": target, "view": _jsonable(view.model_dump()), "market_view": mk, "tick": tick}

    @market_view.task(parents=[form_view], execution_timeout=timedelta(minutes=10), retries=0)
    async def decide(input: ViewInput, ctx: Context) -> dict:
        """Per believed gap: JEV act/watch/skip -> execution agent -> CODE clamp -> JEV safe-without-human. One `decisions` doc each."""
        f = ctx.task_output(form_view)
        if f["skip"]:
            return {"skip": True, "decisions": [], "needs_human": False}
        target, view, mk = f["target_date"], View.model_validate(f["view"]), f.get("market_view") or {}
        tick = engine_state.TickState.model_validate(f["tick"])
        recent = [s async for s in db.scores().find({}, {"_id": 0, "hit": 1}).sort("target_date", -1).limit(10)]
        hit_rate = (sum(1 for s in recent if s.get("hit")) / len(recent)) if recent else None
        caps = Caps()
        used = await exposure_today(target)
        agent = ctx.lifespan["engine"]["execution"]
        results = []
        for i, gap in enumerate(g for g in view.gaps if g.believed):
            bb = tick.buckets.get(gap.bucket)
            depth = (bb.bid_size, bb.ask_size) if bb else (0, 0)
            spread = bb.spread_c if bb else (mk.get("spread_c") or {}).get(gap.bucket, 0)
            killed = bool(tick.running_max_bucket) and tick.running_max_bucket != gap.bucket and bb is not None and bb.ask <= 2
            doc = {"run_id": ctx.workflow_run_id, "target_date": target, "gap": gap.model_dump(), "hours_left": view.hours_left,
                   "depth": list(depth), "spread_c": spread, "created_at": now(), "status": "skip"}
            with tracer.start_as_current_span("bookie.decide.gap") as span:
                span.set_attribute("bucket", gap.bucket)
                span.set_attribute("edge_c", gap.edge_c)
                choice, probs = await jev_questions.act_watch_skip(gap, depth, spread, view.hours_left, tick.running_max_f, killed)
                span.set_attribute("jev.choice", choice)
                doc.update({"choice": choice, "choice_probs": probs, "jev_choice": choice, "jev_probs": probs, "status": choice if choice != "act" else "skip"})
                if choice == "act":
                    out = await run_agent(agent,
                        f"View for {target} (hours_left={view.hours_left}, confidence={view.confidence}):\n" + json.dumps(f["view"], indent=1) +
                        f"\n\nBelieved gap to trade:\n{json.dumps(gap.model_dump(), indent=1)}\n\nPropose an OrderProposal for bucket '{gap.bucket}' only.",
                        thread_id=f"{ctx.workflow_run_id}:exec:{i}", recursion_limit=40)
                    prop: OrderProposal | None = out.get("structured_response")
                    if prop is None:
                        doc.update({"status": "skip", "skip_reason": "execution agent returned no OrderProposal"})
                    else:
                        side_edge = gap.edge_c if prop.side == "yes" else -gap.edge_c
                        cl = clamp(prop, tick, used, caps, edge_c=side_edge)
                        doc.update({"proposal": prop.model_dump(), "clamped": cl.model_dump()})
                        span.set_attribute("clamp.allowed", cl.allowed)
                        if cl.allowed:
                            remaining = max(caps.daily_cap_usd - used, 0.0)
                            frac = (cl.size * cl.limit_price_c / 100) / remaining if remaining > 0 else 1.0
                            safe = await jev_questions.safe_without_human(cl, view.confidence, hit_rate, round(frac, 3), side_edge)
                            span.set_attribute("jev.safe", safe)
                            doc.update({"safe_p": safe, "safe_prob": safe, "status": "recorded" if safe >= SAFE_THRESHOLD else "needs_human"})
                            if safe >= SAFE_THRESHOLD:
                                used += cl.size * cl.limit_price_c / 100
                        else:
                            doc.update({"status": "skip", "skip_reason": f"clamp: {cl.reason}"})
            await db.decisions().insert_one(doc)
            results.append({k: v for k, v in _jsonable(doc).items() if k != "_id"})
        return {"skip": False, "target_date": target, "decisions": results,
                "needs_human": any(r["status"] == "needs_human" for r in results)}

    @market_view.durable_task(parents=[decide], execution_timeout=timedelta(hours=6), schedule_timeout=timedelta(hours=6))
    async def gate(input: ViewInput, ctx: DurableContext) -> dict:
        """Durable wait for a human verdict (bookie:verdict scoped to this run) when any decision needs one."""
        d = ctx.task_output(decide)
        if d.get("skip") or not d.get("needs_human"):
            return {"waited": False, "n": len(d.get("decisions", []))}
        with tracer.start_as_current_span("bookie.human_gate") as span:
            span.set_attribute("run_id", ctx.workflow_run_id)
            ev = await ctx.aio_wait_for_event(VERDICT_EVENT, scope=ctx.workflow_run_id,
                                              lookback_window=timedelta(hours=1), payload_validator=Verdict)
        v = _find(ev, Verdict)
        await db.decisions().update_many({"run_id": ctx.workflow_run_id, "status": "needs_human"},
                                         {"$set": {"status": "approved" if v.approved else "rejected", "note": v.note, "decided_at": now()}})
        return {"waited": True, "approved": v.approved, "note": v.note}

    @market_view.task(parents=[gate], execution_timeout=timedelta(minutes=3), retries=0)
    async def place(input: ViewInput, ctx: Context) -> dict:
        """CODE place (phase 3): every recorded / approved decision of this run -> execute.place_from_decision.
        Dry-run (ORDERS_ENABLED != true) records the exact payload as would_place and sends nothing."""
        d = ctx.task_output(decide)
        if d.get("skip"):
            return {"skip": True, "placed": 0}
        client = ctx.lifespan.get("exchange")
        if client is None:
            return {"skip": True, "reason": "no exchange client configured", "placed": 0}
        caps = Caps()
        out = []
        with tracer.start_as_current_span("bookie.place") as span:
            span.set_attribute("dry_run", client.dry_run)
            async for dec in db.decisions().find({"run_id": ctx.workflow_run_id, "status": {"$in": list(execute.PLACEABLE)}}):
                r = await execute.place_from_decision(dec, client, caps)
                out.append(_jsonable(r))
            span.set_attribute("n", len(out))
            span.set_attribute("placed", sum(1 for r in out if r.get("placed")))
        return {"skip": False, "dry_run": client.dry_run, "results": out,
                "placed": sum(1 for r in out if r.get("placed")), "would_place": sum(1 for r in out if r.get("status") == "would_place")}

    return market_view


def build_sync_orders(hatchet):
    """Hatchet cron (every 2 min): pull our orders + fills from the exchange into db.orders(). GET only."""
    @hatchet.task(name="sync_orders_cron", on_crons=["*/2 * * * *"], execution_timeout=timedelta(minutes=1), retries=0)
    async def sync_orders_cron(input: EmptyModel, ctx: Context) -> dict:
        client = ctx.lifespan.get("exchange")
        if client is None:
            return {"skip": True, "reason": "no exchange client configured"}
        with tracer.start_as_current_span("bookie.sync_orders"):
            return _jsonable(await execute.sync_orders(client))

    return sync_orders_cron
