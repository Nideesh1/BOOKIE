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
from engine import jev_questions, risk, state as engine_state
from engine.clamps import Caps, clamp
from engine.contracts import Gap, MarketView, OrderProposal, View, WeatherView
from engine.tools import exposure_today
from engine import execute
from streams import OUT_PROPOSAL

tracer = trace.get_tracer("bookie.engine")
ET = nws.ET
VERDICT_EVENT = "bookie:verdict"
RETHINK_THRESHOLD = 0.5
SAFE_THRESHOLD = float(os.environ.get("AUTO_APPROVE_THRESHOLD", str(jev_questions.SAFE_THRESHOLD)))
CLOSE_THRESHOLD = jev_questions.CLOSE_THRESHOLD
EDGE_GONE_C = 2          # our-side edge (view p vs mid) at or below this: "the edge is gone"
VIEW_FLIP_C = -4         # our-side edge at or below this: "the view flipped against us"


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
            # The agent's OWN probabilities are the model; gap_table is only a code baseline. Recompute every gap's
            # numbers from p_by_bucket so Jev and the execution agent act on the agent's view, not the baseline.
            mids = {b: bb.mid for b, bb in tick.buckets.items()} if hasattr(tick, "buckets") else {}
            if isinstance(tick, dict):
                mids = {b: v.get("mid") for b, v in (tick.get("buckets") or {}).items()}
            mids = {b: (float(m) / 100.0 if m is not None and float(m) > 1.0 else (float(m) if m is not None else None)) for b, m in mids.items()}
            seen = set()
            for g in view.gaps:
                seen.add(g.bucket)
                if g.bucket in view.p_by_bucket:
                    g.p_model = round(float(view.p_by_bucket[g.bucket]), 3)
                if mids.get(g.bucket) is not None:
                    g.p_market = round(float(mids[g.bucket]), 3)
                g.edge_c = int(round((g.p_model - g.p_market) * 100))
            for b, p in view.p_by_bucket.items():   # buckets the agent priced but didn't list: add as unbelieved
                if b not in seen and mids.get(b) is not None:
                    e = int(round((float(p) - float(mids[b])) * 100))
                    if abs(e) >= 4:
                        view.gaps.append(Gap(bucket=b, p_model=round(float(p), 3), p_market=round(float(mids[b]), 3), edge_c=e,
                                             believed=False, why="not discussed by the agent; listed for completeness"))
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

    async def manage_positions(ctx: Context, target: str, view: View, tick: engine_state.TickState, hit_rate: float | None,
                               caps: Caps) -> tuple[list[dict], dict | None]:
        """Phase 3b, BEFORE new gaps. 1) CODE risk.evaluate_positions -> forced closes (status "forced": no Jev, no agent, no
        human gate). 2) JEV should_close on the rest -> execution agent proposes reduce/close -> clamp -> JEV safe. Returns
        (decision docs, today's P&L or None when positions could not be read)."""
        client = ctx.lifespan.get("exchange")
        rules = risk.RiskRules(caps=caps)
        marks = risk.marks_from_tick(tick)
        if client is None:
            return [], None
        try:
            positions = await risk.positions_from_exchange(client, target)
        except Exception as e:                                  # exchange read failed: manage nothing this run
            return [{"run_id": ctx.workflow_run_id, "target_date": target, "kind": "positions", "status": "skip",
                     "skip_reason": f"positions unavailable ({type(e).__name__})", "created_at": now()}], None
        pnl_today = await risk.daily_pnl(target, positions, marks)
        if not positions:
            return [], pnl_today
        agent = ctx.lifespan["engine"]["execution"]
        docs: list[dict] = []
        forced = risk.evaluate_positions(positions, marks, rules)
        forced_refs = {fa.position.ref for fa in forced}
        for fa in forced:
            with tracer.start_as_current_span("bookie.decide.forced") as span:
                span.set_attribute("reason", fa.reason)
                span.set_attribute("position", fa.position.ref)
                cl = clamp(fa.proposal, tick, 0.0, caps, position_contracts=fa.position.contracts)
                span.set_attribute("clamp.allowed", cl.allowed)
            docs.append({"run_id": ctx.workflow_run_id, "target_date": target, "kind": "position", "position": fa.position.model_dump(),
                         "pnl": fa.pnl.model_dump(exclude={"position"}), "forced_reason": fa.reason, "forced_detail": fa.detail,
                         "hours_left": view.hours_left, "proposal": fa.proposal.model_dump(), "clamped": cl.model_dump(),
                         "created_at": now(), "status": "forced" if cl.allowed else "skip",
                         **({} if cl.allowed else {"skip_reason": f"clamp: {cl.reason}"})})
        for i, pos in enumerate(p for p in positions if p.ref not in forced_refs):
            pn = risk.position_pnl(pos, marks)
            p_view = view.p_by_bucket.get(pos.bucket)
            mid = marks.mid_c.get(pos.bucket)
            edge = None
            if p_view is not None and mid is not None:
                edge = round(p_view * 100 - mid) if pos.side == "yes" else round(mid - p_view * 100)
            edge_gone = edge is None or edge <= EDGE_GONE_C
            flipped = edge is not None and edge <= VIEW_FLIP_C
            doc = {"run_id": ctx.workflow_run_id, "target_date": target, "kind": "position", "position": pos.model_dump(),
                   "pnl": pn.model_dump(exclude={"position"}), "view_p": p_view, "our_side_edge_c": edge, "edge_gone": edge_gone,
                   "view_flipped": flipped, "hours_left": view.hours_left, "created_at": now(), "status": "hold"}
            with tracer.start_as_current_span("bookie.decide.position") as span:
                span.set_attribute("position", pos.ref)
                p_close = await jev_questions.should_close(pos, pn, p_view, edge_gone, flipped)
                span.set_attribute("jev.close", p_close)
                doc.update({"close_p": p_close})
                if p_close >= CLOSE_THRESHOLD:
                    out = await run_agent(agent,
                        f"View for {target} (hours_left={view.hours_left}, confidence={view.confidence}):\n" + json.dumps(view.model_dump(), indent=1) +
                        f"\n\nOPEN POSITION to manage (position_ref='{pos.ref}'):\n{json.dumps(pos.model_dump(), indent=1)}\n"
                        f"Unrealized: {json.dumps(pn.model_dump(exclude={'position'}))}\nOur-side edge now: {edge}¢ (edge_gone={edge_gone}, view_flipped={flipped}).\n"
                        f"Jev says close with p={p_close:.2f}. Propose an OrderProposal with action='reduce' or 'close' on bucket '{pos.bucket}' "
                        f"(side='{'no' if pos.side == 'yes' else 'yes'}', price in that leg's cents, size <= {pos.contracts}), or tactic='skip' to hold.",
                        thread_id=f"{ctx.workflow_run_id}:pos:{i}", recursion_limit=40)
                    prop: OrderProposal | None = out.get("structured_response")
                    if prop is None:
                        doc.update({"status": "hold", "skip_reason": "execution agent returned no OrderProposal"})
                    elif prop.tactic == "skip" or prop.size <= 0:
                        doc.update({"proposal": prop.model_dump(), "status": "hold"})
                    elif not prop.is_close or prop.side == pos.side or prop.bucket != pos.bucket:
                        doc.update({"proposal": prop.model_dump(), "status": "skip",
                                    "skip_reason": f"proposal is not a close of {pos.ref} (action={prop.action}, side={prop.side}, bucket={prop.bucket})"})
                    else:
                        prop = prop.model_copy(update={"position_ref": pos.ref})
                        cl = clamp(prop, tick, 0.0, caps, position_contracts=pos.contracts)
                        doc.update({"proposal": prop.model_dump(), "clamped": cl.model_dump()})
                        span.set_attribute("clamp.allowed", cl.allowed)
                        if cl.allowed:
                            safe = await jev_questions.safe_without_human(cl, view.confidence, hit_rate, 0.0, None)
                            span.set_attribute("jev.safe", safe)
                            doc.update({"safe_p": safe, "safe_prob": safe, "status": "recorded" if safe >= SAFE_THRESHOLD else "needs_human"})
                        else:
                            doc.update({"status": "skip", "skip_reason": f"clamp: {cl.reason}"})
            docs.append(doc)
        return docs, pnl_today

    @market_view.task(parents=[form_view], execution_timeout=timedelta(minutes=10), retries=0)
    async def decide(input: ViewInput, ctx: Context) -> dict:
        """Positions first (risk rules -> Jev should_close -> execution agent), then per believed gap: JEV act/watch/skip ->
        execution agent -> CODE clamp -> JEV safe-without-human. One `decisions` doc each."""
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
        pos_docs, pnl_today = await manage_positions(ctx, target, view, tick, hit_rate, caps)
        for doc in pos_docs:
            await db.decisions().insert_one(doc)
            results.append({k: v for k, v in _jsonable(doc).items() if k != "_id"})
        rules = risk.RiskRules(caps=caps)
        loss_capped = pnl_today is not None and rules.daily_loss_cap_hit(pnl_today["total_usd"])
        for i, gap in enumerate(g for g in view.gaps if g.believed):
            if loss_capped:
                doc = {"run_id": ctx.workflow_run_id, "target_date": target, "gap": gap.model_dump(), "hours_left": view.hours_left,
                       "created_at": now(), "status": "skip", "pnl_today": pnl_today,
                       "skip_reason": f"daily loss cap: today's P&L ${pnl_today['total_usd']:.2f} <= -${rules.daily_loss_cap_usd:.2f}; no new opens"}
                await db.decisions().insert_one(doc)
                results.append({k: v for k, v in _jsonable(doc).items() if k != "_id"})
                continue
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
