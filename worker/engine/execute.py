"""Phase 3 order path (docs/ENGINE.md "CODE place"): decision doc -> re-clamp -> exchange, plus order sync and cancel.

Code only. The model never reaches this module. Every write is dry-run aware: with ORDERS_ENABLED != "true" the
client returns the exact payload it would have sent and we persist it as status "would_place" / "would_cancel".

orders doc (db.orders()):
  client_order_id  "bookie-<decision _id>"   (unique index -> placing twice for one decision is impossible)
  decision_id, run_id, target_date, bucket, ticker, side (yes|no), action (buy|sell), count, price_c (that side's cents),
  tactic, post_only, payload (V2 request body), status: would_place | resting | executed | canceled | rejected |
  would_cancel | error, order_id (exchange), fill_count, remaining_count, avg_fill_price, fees_usd, fills[],
  created_at, updated_at, exchange_env, dry_run,
  intent (open | add | reduce | close), position_ref, forced_reason   (phase 3b; closes are sent reduce_only).

Phase 3b: a decision with status "forced" (engine.risk decided the close) bypasses the human gate but still goes through
the clamp (close mode) and the kill switch here. New opens are refused once the daily loss cap is hit.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Any

from bson import ObjectId

import db
from exchange import ExchangeError, KalshiClient, event_ticker_for, market_ticker_for

from . import state as engine_state
from . import risk
from .clamps import Caps, clamp
from .contracts import OrderProposal
from .tools import exposure_today

PLACEABLE = ("recorded", "approved", "forced")
OPEN_STATUSES = ("resting", "would_place")   # would_place docs are listed, never mutated, by cancel_all_for


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _cid(decision: dict) -> str:
    return f"bookie-{decision['_id']}"


async def _skip(decision: dict, reason: str) -> dict:
    await db.decisions().update_one({"_id": decision["_id"]},
                                    {"$set": {"order": {"status": "skipped", "skip_reason": reason, "at": now()}}})
    return {"placed": False, "skip_reason": reason, "decision_id": str(decision["_id"])}


async def place_from_decision(decision: dict, client: KalshiClient, caps: Caps | None = None) -> dict:
    """Turn one `decisions` doc into (at most) one exchange order. Idempotent: a decision that already has an
    order_id, or whose client_order_id is already in db.orders(), is never sent again."""
    caps = caps or Caps()
    status = decision.get("status")
    clamped = decision.get("clamped") or {}
    prop_d = decision.get("proposal") or {}
    if status not in PLACEABLE:
        return {"placed": False, "skip_reason": f"status {status!r} not in {PLACEABLE}", "decision_id": str(decision["_id"])}
    if not clamped.get("allowed"):
        return await _skip(decision, "clamp did not allow this order")
    if decision.get("order_id") or (decision.get("order") or {}).get("order_id"):
        return {"placed": False, "skip_reason": "decision already has an order_id", "decision_id": str(decision["_id"])}
    cid = _cid(decision)
    existing = await db.orders().find_one({"client_order_id": cid})
    if existing and existing.get("status") not in ("would_place", "error"):
        return {"placed": False, "skip_reason": f"order already exists ({existing.get('status')})", "decision_id": str(decision["_id"])}
    if caps.kill_switch_on():
        return await _skip(decision, f"kill switch on ({caps.kill_switch_path})")
    tactic = prop_d.get("tactic")
    if tactic == "skip":
        return await _skip(decision, "tactic skip")
    if tactic == "ladder":
        return await _skip(decision, "ladder tactic is not implemented in phase 3; nothing sent")
    if tactic not in ("post_and_wait", "cross_now"):
        return await _skip(decision, f"unknown tactic {tactic!r}")

    # ---- re-clamp against a FRESH tick_state -------------------------------------------------
    target = decision["target_date"]
    prop = OrderProposal.model_validate({**prop_d, "size": int(clamped.get("size") or prop_d.get("size") or 0),
                                         "limit_price_c": int(clamped.get("limit_price_c") or prop_d.get("limit_price_c") or 0)})
    tick = await engine_state.tick_state(target)
    if prop.is_close:
        # closes free risk: no exposure math, no daily-loss gate; size capped at the contracts we hold (as recorded)
        held = int(((decision.get("position") or {}).get("contracts")) or prop.size)
        cl = clamp(prop, tick, 0.0, caps, position_contracts=held)
    else:
        rules = risk.RiskRules(caps=caps)
        gap = decision.get("gap") or {}
        side_edge = None
        if gap.get("edge_c") is not None:
            side_edge = int(gap["edge_c"]) if prop.side == "yes" else -int(gap["edge_c"])
        used = await exposure_today(target)
        used -= _f(clamped.get("size")) * _f(clamped.get("limit_price_c")) / 100     # this decision is already counted
        try:
            positions = await risk.positions_from_exchange(client, target)
            pnl = await risk.daily_pnl(target, positions, risk.marks_from_tick(tick))
        except Exception as e:                       # exchange read failed: fall back to what our own fills say
            realized, fees = await risk.realized_today_usd(target)
            pnl = {"total_usd": realized - fees, "note": f"positions unavailable ({type(e).__name__})"}
        if rules.daily_loss_cap_hit(pnl["total_usd"]):
            return await _skip(decision, f"daily loss cap: today's P&L ${pnl['total_usd']:.2f} <= -${rules.daily_loss_cap_usd:.2f}; no new opens")
        cl = clamp(prop, tick, max(used, 0.0), caps, edge_c=side_edge)
    await db.decisions().update_one({"_id": decision["_id"]}, {"$set": {"reclamped": cl.model_dump(), "reclamped_at": now()}})
    if not cl.allowed:
        return await _skip(decision, f"re-clamp: {cl.reason}")

    try:
        ticker = await market_ticker_for(target, prop.bucket)
    except LookupError as e:
        return await _skip(decision, str(e))

    post_only = tactic == "post_and_wait" and not prop.is_close      # a close must be able to take liquidity
    payload = client.limit_payload(ticker, prop.side, "buy", cl.size, cl.limit_price_c, cid, post_only=post_only, reduce_only=prop.is_close)
    doc: dict[str, Any] = {
        "client_order_id": cid, "decision_id": decision["_id"], "run_id": decision.get("run_id"), "target_date": target,
        "bucket": prop.bucket, "ticker": ticker, "side": prop.side, "action": "buy", "count": cl.size,
        "price_c": cl.limit_price_c, "tactic": tactic, "post_only": post_only, "payload": payload,
        "intent": prop.action, "position_ref": prop.position_ref, "forced_reason": decision.get("forced_reason"),
        "clamps_applied": cl.clamps_applied, "exchange_env": client.env, "dry_run": client.dry_run,
        "created_at": now(), "updated_at": now(), "fill_count": 0, "remaining_count": cl.size, "fills": [],
    }
    try:
        resp = await client.place_limit(ticker, prop.side, "buy", cl.size, cl.limit_price_c, cid, post_only=post_only, reduce_only=prop.is_close)
    except ExchangeError as e:
        doc.update({"status": "error", "error": str(e)})
        await db.orders().replace_one({"client_order_id": cid}, doc, upsert=True)
        await db.decisions().update_one({"_id": decision["_id"]}, {"$set": {"order": {"status": "error", "error": str(e), "client_order_id": cid, "at": now()}}})
        return {"placed": False, "error": str(e), "decision_id": str(decision["_id"]), "payload": payload}

    if resp.get("dry_run"):
        doc.update({"status": "would_place", "response": resp})
        await db.orders().replace_one({"client_order_id": cid}, doc, upsert=True)
        await db.decisions().update_one({"_id": decision["_id"]},
                                        {"$set": {"order": {"status": "would_place", "client_order_id": cid, "ticker": ticker,
                                                            "payload": payload, "at": now()}}})
        return {"placed": False, "dry_run": True, "status": "would_place", "payload": payload, "decision_id": str(decision["_id"])}

    fill = _f(resp.get("fill_count")); remaining = _f(resp.get("remaining_count"))
    st = "executed" if remaining <= 0 and fill > 0 else "resting"
    doc.update({"status": st, "order_id": resp.get("order_id"), "response": resp, "fill_count": fill, "remaining_count": remaining,
                "avg_fill_price": resp.get("average_fill_price"), "placed_at": now()})
    await db.orders().replace_one({"client_order_id": cid}, doc, upsert=True)
    await db.decisions().update_one({"_id": decision["_id"]},
                                    {"$set": {"order_id": resp.get("order_id"),
                                              "order": {"status": st, "order_id": resp.get("order_id"), "client_order_id": cid,
                                                        "ticker": ticker, "payload": payload, "at": now()}}})
    return {"placed": True, "status": st, "order_id": resp.get("order_id"), "payload": payload, "decision_id": str(decision["_id"])}


async def sync_orders(client: KalshiClient, target_date: str | None = None) -> dict:
    """Pull orders + fills from the exchange and refresh db.orders() docs (idempotent; matched by order_id or
    client_order_id). Read-only against the exchange."""
    q: dict = {"status": {"$in": ["resting", "executed", "canceled", "would_place"]}}
    if target_date:
        q["target_date"] = target_date
    ours = [o async for o in db.orders().find(q)]
    if not ours:
        return {"checked": 0, "updated": 0}
    by_cid = {o["client_order_id"]: o for o in ours}
    by_oid = {o["order_id"]: o for o in ours if o.get("order_id")}
    ex_orders: list[dict] = []
    for st in ("resting", "executed", "canceled"):
        ex_orders += await client.orders(status=st)
    updated = 0
    matched: list[tuple[dict, dict]] = []
    for eo in ex_orders:
        mine = by_oid.get(eo.get("order_id")) or by_cid.get(eo.get("client_order_id") or "")
        if mine:
            matched.append((mine, eo))
    for mine, eo in matched:
        fills = await client.fills(order_id=eo["order_id"]) if _f(eo.get("fill_count_fp")) > 0 else []
        fees = _f(eo.get("taker_fees_dollars")) + _f(eo.get("maker_fees_dollars"))
        upd = {"order_id": eo["order_id"], "status": eo.get("status") or mine.get("status"),
               "fill_count": _f(eo.get("fill_count_fp")), "remaining_count": _f(eo.get("remaining_count_fp")),
               "yes_price_dollars": eo.get("yes_price_dollars"), "no_price_dollars": eo.get("no_price_dollars"),
               "fees_usd": round(fees, 4), "exchange_order": eo, "updated_at": now(), "synced_at": now(),
               "fills": [{"fill_id": f.get("fill_id"), "count": _f(f.get("count_fp")), "yes_price_dollars": f.get("yes_price_dollars"),
                          "no_price_dollars": f.get("no_price_dollars"), "is_taker": f.get("is_taker"), "fee_cost": f.get("fee_cost"),
                          "outcome_side": f.get("outcome_side"), "book_side": f.get("book_side"), "action": f.get("action"),
                          "created_time": f.get("created_time")} for f in fills]}
        r = await db.orders().update_one({"_id": mine["_id"]}, {"$set": upd})
        updated += r.modified_count
        if mine.get("decision_id") is not None:
            await db.decisions().update_one({"_id": mine["decision_id"]},
                                            {"$set": {"order_id": eo["order_id"], "order.status": upd["status"], "order.order_id": eo["order_id"],
                                                      "order.fill_count": upd["fill_count"], "order.synced_at": now()}})
    return {"checked": len(ours), "exchange_orders": len(ex_orders), "matched": len(matched), "updated": updated}


async def cancel_all_for(target_date: str, client: KalshiClient) -> dict:
    """Cancel every open order of ours for target_date. Dry-run: lists what it would cancel (our resting docs plus any
    resting exchange orders on that event), sends nothing."""
    ev = event_ticker_for(target_date)
    mine = [o async for o in db.orders().find({"target_date": target_date, "status": {"$in": list(OPEN_STATUSES)}})]
    try:
        resting = await client.orders(status="resting", event_ticker=ev)
    except ExchangeError as e:
        resting = []
        exchange_err = str(e)
    else:
        exchange_err = None
    targets: dict[str, dict] = {}
    for o in mine:
        if o.get("order_id"):
            targets[o["order_id"]] = {"order_id": o["order_id"], "ticker": o.get("ticker"), "source": "db", "client_order_id": o.get("client_order_id")}
    for eo in resting:
        targets.setdefault(eo["order_id"], {"order_id": eo["order_id"], "ticker": eo.get("ticker"), "source": "exchange",
                                            "client_order_id": eo.get("client_order_id")})
    would = [o for o in mine if not o.get("order_id")]      # would_place docs: nothing on the exchange to cancel
    results = []
    for oid, t in targets.items():
        try:
            r = await client.cancel(oid)
        except ExchangeError as e:
            results.append({**t, "error": str(e)}); continue
        results.append({**t, "result": r})
        if not r.get("dry_run"):
            await db.orders().update_one({"order_id": oid}, {"$set": {"status": "canceled", "canceled_at": now(), "updated_at": now(),
                                                                       "cancel_response": r}})
        else:
            await db.orders().update_one({"order_id": oid}, {"$set": {"last_cancel_dry_run_at": now()}})
    return {"target_date": target_date, "event_ticker": ev, "dry_run": client.dry_run, "cancelled" if not client.dry_run else "would_cancel": results,
            "would_place_docs": [str(o["_id"]) for o in would],   # dry-run payloads: nothing on the exchange to cancel
            "exchange_error": exchange_err, "n": len(results)}
