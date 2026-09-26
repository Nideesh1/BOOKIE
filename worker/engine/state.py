"""tick_state(): the code-owned state block computed every 15 s (docs/ENGINE.md "The loop").

Everything here is deterministic and read from Atlas: the latest market snapshot, today's observations,
the latest NWS forecast, and the last `views` doc (for deltas). Running max / remaining-forecast-max /
hours_left come from intraday.estimate_today (imported, not duplicated). Depth (bid/ask size) is not in
the stored snapshots yet, so it is read from Kalshi's public orderbook endpoint with a short cache; when
that fails sizes are 0 and the clamps treat the book as having no visible depth.
"""
from __future__ import annotations

import datetime as dt
import math
import time
from datetime import timedelta

import httpx

import db
import intraday
import kalshi
import nws

from .contracts import BucketBook, TickState

ET = nws.ET
_DEPTH_TTL_S = 10.0
_depth_cache: dict[str, tuple[float, tuple[int, int]]] = {}


def _c(dollars: float | None) -> int:
    return int(round(float(dollars or 0) * 100))


def bucket_for_temp(temp_f: float | None, markets: list[dict]) -> str | None:
    """Label of the Kalshi bucket a (rounded) temperature settles in, using the tail encoding from intraday._p_bucket."""
    if temp_f is None:
        return None
    t = round(temp_f)
    for m in markets:
        floor, cap = m.get("floor"), m.get("cap")
        if floor is None and cap is not None and t <= cap - 1:
            return m.get("label") or kalshi.bucket_of(m)
        if cap is None and floor is not None and t >= floor + 1:
            return m.get("label") or kalshi.bucket_of(m)
        if floor is not None and cap is not None and floor <= t <= cap:
            return m.get("label") or kalshi.bucket_of(m)
    return None


async def live_depth(tickers: list[str]) -> dict[str, tuple[int, int]]:
    """Best yes-bid size and best yes-ask size per ticker from Kalshi's public orderbook. Yes asks are the
    mirror of no bids (ask = 1 - best no bid). Returns (0, 0) for a ticker it cannot read."""
    now = time.monotonic()
    out: dict[str, tuple[int, int]] = {}
    todo = []
    for t in tickers:
        hit = _depth_cache.get(t)
        if hit and now - hit[0] < _DEPTH_TTL_S:
            out[t] = hit[1]
        else:
            todo.append(t)
    if todo:
        async with httpx.AsyncClient(timeout=8) as c:
            for t in todo:
                try:
                    r = await c.get(f"{kalshi.BASE}/markets/{t}/orderbook", params={"depth": 1})
                    r.raise_for_status()
                    ob = r.json().get("orderbook_fp") or {}
                    yes = ob.get("yes_dollars") or []
                    no = ob.get("no_dollars") or []
                    best_yes = max(yes, key=lambda x: float(x[0])) if yes else None
                    best_no = max(no, key=lambda x: float(x[0])) if no else None
                    d = (int(float(best_yes[1])) if best_yes else 0, int(float(best_no[1])) if best_no else 0)
                except Exception:
                    d = (0, 0)
                out[t] = d
                _depth_cache[t] = (now, d)
    return out


def _curve_at(fc: dict | None, ts: dt.datetime) -> float | None:
    """NWS hourly forecast temperature for the hour containing ts."""
    if not fc:
        return None
    best: tuple[float, float] | None = None       # (distance_s, temp)
    for p in fc.get("periods", []):
        try:
            t = intraday._parse_ts(p["t"])
        except (TypeError, ValueError, KeyError):
            continue
        if p.get("temp_f") is None:
            continue
        if t <= ts < t + timedelta(hours=1):
            return float(p["temp_f"])
        d = abs((t - ts).total_seconds())
        if best is None or d < best[0]:
            best = (d, float(p["temp_f"]))
    # the hourly curve starts at the fetch hour; an obs just before it compares against the nearest period (<= 90 min)
    return best[1] if best and best[0] <= 90 * 60 else None


async def _recent_obs(as_of: dt.datetime, hours: float = 3.0) -> list[tuple[dt.datetime, float]]:
    since = as_of - timedelta(hours=hours)
    q = {"$or": [{"ts": {"$gte": since}}, {"ts": {"$gte": since.isoformat()}}]}
    obs = []
    async for o in db.observations().find(q, {"_id": 0, "ts": 1, "temp_f": 1}).sort("ts", -1).limit(24):
        try:
            ts = intraday._parse_ts(o["ts"])
        except (TypeError, ValueError):
            continue
        if o.get("temp_f") is not None and ts <= as_of:
            obs.append((ts, float(o["temp_f"])))
    return obs


def _slope_per_hr(obs: list[tuple[dt.datetime, float]]) -> float | None:
    if len(obs) < 2:
        return None
    t0 = obs[-1][0]
    xs = [(t - t0).total_seconds() / 3600 for t, _ in obs]
    ys = [y for _, y in obs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den, 2)


async def last_view(target_date: str) -> dict | None:
    return await db.views().find_one({"target_date": target_date}, sort=[("created_at", -1)])


async def tick_state(target_date: str, as_of: dt.datetime | None = None, *, with_depth: bool = True) -> TickState:
    as_of = (as_of or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    est = await intraday.estimate_today(target_date, as_of)
    snap = await db.market_snapshots().find_one({"target_date": target_date}, sort=[("ts", -1)])
    markets = snap["markets"] if snap else []
    fc = await db.forecasts().find_one({"target_date": target_date}, {"periods": 1, "fetched_at": 1}, sort=[("fetched_at", -1)])

    depth = await live_depth([m["ticker"] for m in markets if m.get("ticker")]) if (with_depth and markets) else {}
    buckets: dict[str, BucketBook] = {}
    for m in markets:
        label = m.get("label") or kalshi.bucket_of(m)
        bid, ask = _c(m.get("yes_bid")), _c(m.get("yes_ask"))
        mid = (bid + ask) / 2 if (bid or ask) else None
        bs, asz = depth.get(m.get("ticker"), (0, 0))
        buckets[label] = BucketBook(bid=bid, ask=ask, mid=mid, bid_size=bs, ask_size=asz, spread_c=max(0, ask - bid),
                                    implied_p=round(mid / 100, 3) if mid is not None else None,
                                    floor=m.get("floor"), cap=m.get("cap"), ticker=m.get("ticker"))

    obs = await _recent_obs(as_of)
    last_obs_f = obs[0][1] if obs else None
    last_obs_age_min = int((as_of - obs[0][0]).total_seconds() // 60) if obs else None
    curve = _curve_at(fc, obs[0][0]) if obs else None
    obs_drift_f = round(last_obs_f - curve, 1) if (last_obs_f is not None and curve is not None) else None

    tick = TickState(
        target_date=target_date, as_of=as_of, market_ts=snap.get("ts") if snap else None, buckets=buckets,
        running_max_f=est["running_max"], running_max_bucket=bucket_for_temp(est["running_max"], markets),
        last_obs_f=last_obs_f, last_obs_age_min=last_obs_age_min, trend_f_per_hr=_slope_per_hr(obs),
        remaining_forecast_max_f=est["remaining_forecast_max"], hours_left=float(est["hours_left"]),
        forecast_fetched_at=fc.get("fetched_at") if fc else None, obs_drift_f=obs_drift_f,
    )

    # ---- deltas vs the last view ---------------------------------------------------------
    lv = await last_view(target_date)
    if lv:
        prev = lv.get("tick") or {}
        created = lv.get("created_at")
        if isinstance(created, dt.datetime):
            created = created if created.tzinfo else created.replace(tzinfo=dt.timezone.utc)
            tick.last_view_at = created
            tick.last_view_age_s = round((as_of - created).total_seconds(), 1)
        pb = prev.get("buckets") or {}
        for label, b in buckets.items():
            pm = (pb.get(label) or {}).get("mid")
            if b.mid is not None and pm is not None:
                tick.mid_delta_c[label] = int(round(b.mid - float(pm)))
                pbs, pas = int((pb[label]).get("bid_size") or 0), int((pb[label]).get("ask_size") or 0)
                if (pbs or pas) and (b.bid_size or b.ask_size) and ((pbs > pas) != (b.bid_size > b.ask_size)):
                    tick.depth_flipped.append(label)
        tick.boundary_crossed = bool(prev.get("running_max_bucket")) and prev.get("running_max_bucket") != tick.running_max_bucket
        pf = prev.get("forecast_fetched_at")
        if isinstance(pf, str):
            pf = intraday._parse_ts(pf)
        if isinstance(pf, dt.datetime) and tick.forecast_fetched_at is not None:
            pf = pf if pf.tzinfo else pf.replace(tzinfo=dt.timezone.utc)
            tick.new_forecast = tick.forecast_fetched_at > pf
    return tick


async def record_tick(tick: TickState) -> None:
    """Persist to `ticks` (ENGINE.md: written by code every 15 s)."""
    await db.ticks().insert_one(tick.model_dump(mode="python"))
