"""Read-only LangChain tools over Atlas, grouped per agent (docs/ENGINE.md "Tools per agent").

Subagents get only their domain's list. The weather tools re-declare brain.py's tools on the same db calls
so brain.py can later import them from here. Nothing in this module writes to Atlas or to the exchange.
"""
from __future__ import annotations

import datetime as dt
from datetime import timedelta

from langchain_core.tools import tool

import db
import intraday
import kalshi
import memory_search

from . import risk
from .state import live_depth, tick_state

_xc = None                    # ctx-less exchange client for the tools (GET only); None until first use, False if unavailable


def _exchange():
    global _xc
    if _xc is None:
        try:
            from exchange import KalshiClient
            _xc = KalshiClient.from_env()
        except (KeyError, OSError, ValueError):
            _xc = False
    return _xc or None


def _iso(v) -> str | None:
    return v.isoformat() if hasattr(v, "isoformat") else v


# ---- weather subagent ---------------------------------------------------------------------
@tool
async def get_forecast(target_date: str) -> dict:
    """Latest NWS hourly forecast for target_date (YYYY-MM-DD, ET): day max and the hourly curve."""
    doc = await db.forecasts().find_one({"target_date": target_date}, sort=[("fetched_at", -1)])
    if not doc:
        return {"error": "no forecast stored for that date"}
    hours = [p for p in doc["periods"] if p["t"].startswith(target_date)]
    return {"fetched_at": _iso(doc["fetched_at"]), "day_max_f": doc["day_max"],
            "hourly": [{"h": p["t"][11:13], "f": p["temp_f"], "pop": p.get("pop"), "sky": p.get("short")} for p in hours]}


@tool
async def get_observations(hours: int = 24) -> list[dict]:
    """Recent Central Park observations, newest first. Timestamps are UTC. Convert to America/New_York (EDT = UTC-4 now, EST = UTC-5 in winter) before attributing a reading to a day."""
    cur = db.observations().find({}, {"_id": 0}).sort("ts", -1).limit(hours)
    return [{"ts": _iso(o["ts"]), "temp_f": o["temp_f"]} async for o in cur]


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


WEATHER_TOOLS = [get_forecast, get_observations, get_recent_actuals, search_past_reasoning]


# ---- market subagent ----------------------------------------------------------------------
async def _latest_snapshot(target_date: str) -> dict | None:
    return await db.market_snapshots().find_one({"target_date": target_date}, sort=[("ts", -1)])


def _c(dollars) -> int:
    return int(round(float(dollars or 0) * 100))


@tool
async def get_book(target_date: str) -> dict:
    """Latest Kalshi book for target_date (YYYY-MM-DD): per bucket yes bid/ask in cents, mid, spread, visible
    depth at the best bid/ask (contracts), volume and open interest, plus the crowd favorite."""
    s = await _latest_snapshot(target_date)
    if not s:
        return {"error": "no market snapshot"}
    depth = await live_depth([m["ticker"] for m in s["markets"] if m.get("ticker")])
    rows = []
    for m in s["markets"]:
        bid, ask = _c(m.get("yes_bid")), _c(m.get("yes_ask"))
        bs, asz = depth.get(m.get("ticker"), (0, 0))
        rows.append({"bucket": m.get("label") or kalshi.bucket_of(m), "bid_c": bid, "ask_c": ask,
                     "mid_c": (bid + ask) / 2 if (bid or ask) else None, "spread_c": max(0, ask - bid),
                     "bid_size": bs, "ask_size": asz, "last_c": _c(m.get("last")),
                     "volume": m.get("volume"), "open_interest": m.get("open_interest")})
    return {"ts": _iso(s["ts"]), "favorite_bucket": s.get("favorite_bucket"), "favorite_mid": s.get("favorite_mid"), "buckets": rows}


@tool
async def get_book_history(target_date: str, minutes: int = 60) -> dict:
    """Mid price (cents) per bucket over the last `minutes` of stored snapshots, oldest first, with the net drift per bucket."""
    since = dt.datetime.now(dt.timezone.utc) - timedelta(minutes=minutes)
    cur = db.market_snapshots().find({"target_date": target_date, "ts": {"$gte": since}}, {"_id": 0, "ts": 1, "markets": 1}).sort("ts", 1)
    series: dict[str, list[dict]] = {}
    n = 0
    async for s in cur:
        n += 1
        for m in s["markets"]:
            bid, ask = _c(m.get("yes_bid")), _c(m.get("yes_ask"))
            if not (bid or ask):
                continue
            series.setdefault(m.get("label") or kalshi.bucket_of(m), []).append({"ts": _iso(s["ts"]), "mid_c": (bid + ask) / 2})
    if not n:
        return {"error": f"no snapshots in the last {minutes} min"}
    drift = {b: int(round(v[-1]["mid_c"] - v[0]["mid_c"])) for b, v in series.items() if len(v) >= 2}
    return {"minutes": minutes, "snapshots": n, "drift_c": drift, "mids": series}


@tool
async def get_depth(target_date: str) -> dict:
    """Visible depth per bucket: contracts resting at the best yes bid and best yes ask (from the live Kalshi orderbook)."""
    s = await _latest_snapshot(target_date)
    if not s:
        return {"error": "no market snapshot"}
    depth = await live_depth([m["ticker"] for m in s["markets"] if m.get("ticker")])
    return {(m.get("label") or kalshi.bucket_of(m)): {"bid_size": depth.get(m.get("ticker"), (0, 0))[0],
                                                       "ask_size": depth.get(m.get("ticker"), (0, 0))[1]} for m in s["markets"]}


@tool
async def get_trades_recent(target_date: str, minutes: int = 60) -> dict:
    """Recent public trades per bucket. Not stored yet: returns an error until the trade feed lands (phase 3)."""
    return {"error": "not available"}


MARKET_TOOLS = [get_book, get_book_history, get_depth, get_trades_recent]


# ---- main agent ---------------------------------------------------------------------------
@tool
async def gap_table(target_date: str) -> dict:
    """Code-computed gap table for target_date: per bucket the model probability (running max + remaining NWS curve),
    the market mid, and the edge in cents (p_model - mid). Positive edge = model thinks yes is cheap."""
    est = await intraday.estimate_today(target_date)
    return {"as_of": _iso(est["as_of"]), "running_max_f": est["running_max"], "remaining_forecast_max_f": est["remaining_forecast_max"],
            "hours_left": est["hours_left"], "sd_f": est["sd_f"], "obs_n": est["obs_n"],
            "favorite_market": est["favorite_bucket_market"], "favorite_model": est["favorite_bucket_model"],
            "buckets": [{"bucket": b["label"], "p_model": b["p_model"], "mid": b["mid"], "p_market": b["mid"],
                         "edge_c": b["edge_cents"]} for b in est["buckets"]]}


@tool
async def get_last_view(target_date: str) -> dict:
    """Your most recent View for target_date (probabilities, gaps, confidence, rationale) and how old it is. Empty if none yet."""
    v = await db.views().find_one({"target_date": target_date}, {"_id": 0, "tick": 0}, sort=[("created_at", -1)])
    if not v:
        return {"error": "no view yet for that date"}
    created = v.get("created_at")
    if isinstance(created, dt.datetime):
        v["age_s"] = round((dt.datetime.now(dt.timezone.utc) - created).total_seconds())
        v["created_at"] = created.isoformat()
    return v


MAIN_TOOLS = [gap_table, get_last_view]


# ---- execution agent ----------------------------------------------------------------------
async def exposure_today(target_date: str) -> float:
    """Dollars at risk from recorded (clamped, allowed) decisions today: sum of size * limit price. 0 if none."""
    total = 0.0
    q = {"target_date": target_date, "clamped.allowed": True, "proposal.action": {"$nin": ["reduce", "close"]}}   # closes free risk
    async for d in db.decisions().find(q, {"_id": 0, "clamped": 1}):
        c = d.get("clamped") or {}
        total += float(c.get("size") or 0) * float(c.get("limit_price_c") or 0) / 100
    return round(total, 2)


@tool
async def get_exposure(target_date: str) -> dict:
    """Dollars already committed today across recorded orders (size x price), and how many orders that is."""
    n = await db.decisions().count_documents({"target_date": target_date, "clamped.allowed": True})
    return {"target_date": target_date, "exposure_usd": await exposure_today(target_date), "orders": n}


@tool
async def get_my_fills(target_date: str) -> list[dict]:
    """Your fills on target_date's event (from the synced orders docs): ticker, bucket, side, count, price, fee, time."""
    out = []
    for ticker, legs in (await risk.our_fills(target_date)).items():
        for t, f, o in legs:
            out.append({"time": _iso(t), "ticker": ticker, "bucket": o.get("bucket"), "side": f.get("outcome_side") or o.get("side"),
                        "action": f.get("action") or o.get("action"), "count": f.get("count"), "yes_price_dollars": f.get("yes_price_dollars"),
                        "no_price_dollars": f.get("no_price_dollars"), "fee_cost": f.get("fee_cost"), "client_order_id": o.get("client_order_id")})
    return out


async def positions_now(target_date: str | None = None) -> list[risk.Position] | dict:
    """Live positions via the module client; {"error": ...} when the exchange is unavailable."""
    client = _exchange()
    if client is None:
        return {"error": "exchange not configured (KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH)"}
    try:
        return await risk.positions_from_exchange(client, target_date)
    except Exception as e:                       # ExchangeError / httpx: the agent gets a reason, never a traceback
        return {"error": f"exchange unavailable ({type(e).__name__}: {str(e)[:160]})"}


@tool
async def get_positions(target_date: str | None = None) -> list[dict] | dict:
    """Your open positions on the exchange (live GET): ticker, bucket, side (yes|no), contracts, average entry price in
    that side's cents. Empty list when flat; {"error": ...} when the exchange cannot be reached."""
    pos = await positions_now(target_date)
    if isinstance(pos, dict):
        return pos
    return [p.model_dump() for p in pos]


async def position_pnl_rows(target_date: str | None = None, positions: list[risk.Position] | None = None) -> list[dict] | dict:
    """Per position: entry, current mid, unrealized cents / % / dollars. Marks come from the latest market snapshot."""
    if positions is None:
        positions = await positions_now(target_date)
        if isinstance(positions, dict):
            return positions
    rows = []
    marks_by_date: dict[str, risk.Marks] = {}
    for pos in positions:
        td = pos.target_date or target_date
        if td not in marks_by_date:
            snap = await _latest_snapshot(td) if td else None
            est = await intraday.estimate_today(td) if (td and snap) else None
            marks_by_date[td] = risk.marks_from_snapshot(snap, est["running_max"], est["hours_left"]) if snap else risk.Marks()
        pn = risk.position_pnl(pos, marks_by_date[td])
        rows.append({"position_ref": pos.ref, "ticker": pos.ticker, "bucket": pos.bucket, "target_date": td, "side": pos.side,
                     "contracts": pos.contracts, "entry_c": pos.entry_c, "mark_c": pn.mark_c, "unrealized_c": pn.unrealized_c,
                     "unrealized_pct": pn.unrealized_pct, "unrealized_usd": pn.unrealized_usd, "entry_source": pos.source})
    return rows


@tool
async def position_pnl(target_date: str | None = None) -> list[dict] | dict:
    """Unrealized P&L per open position: average entry (from your fills), current mid, unrealized cents, % of entry cost,
    dollars, contracts. [] when flat; {"error": ...} when the exchange cannot be reached."""
    return await position_pnl_rows(target_date)


EXEC_TOOLS = [get_book, get_depth, get_exposure, get_my_fills, get_positions, position_pnl]

__all__ = ["WEATHER_TOOLS", "MARKET_TOOLS", "MAIN_TOOLS", "EXEC_TOOLS", "tick_state", "exposure_today", "positions_now", "position_pnl_rows"]
