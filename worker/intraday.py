"""Intraday gap watch: pure code, no LLM.

Re-estimates TODAY's daily-high bucket probabilities from what has already happened (running max of
observations over the America/New_York calendar day) plus what is left of the day (max NWS hourly
forecast over the hours still ahead), and compares them to the live Kalshi book.

Model: daily max M = max(running_max, X) where X ~ Normal(remaining_forecast_max, sd), sd widening with
hours left. If no hours are left, M = running_max exactly.
"""
from __future__ import annotations

import datetime as dt
import math
from datetime import timedelta

import db
import kalshi
import nws

ET = nws.ET


def _sd(hours_left: int) -> float:
    if hours_left <= 3:
        return 1.5
    if hours_left <= 8:
        return 2.5
    return 3.5


def _norm_cdf(x: float, mu: float, sd: float) -> float:
    return 0.5 * (1.0 + math.erf((x - mu) / (sd * math.sqrt(2.0))))


def _p_bucket(floor: int | None, cap: int | None, running_max: int | None, remaining_max: float | None, sd: float) -> float:
    """P(M in bucket). Interior buckets cover rounded integers floor..cap. Kalshi encodes tails as strict
    bounds: "61 or below" is floor=None, cap=62 (max < 62); "70 or above" is floor=69, cap=None (max > 69).

    The rounded max lands in [a, b] when the continuous max is in [a-0.5, b+0.5).
    """
    if floor is None and cap is None:
        lo, hi = -math.inf, math.inf
    elif floor is None:
        lo, hi = -math.inf, cap - 0.5          # rounded max <= cap-1
    elif cap is None:
        lo, hi = floor + 0.5, math.inf         # rounded max >= floor+1
    else:
        lo, hi = floor - 0.5, cap + 0.5
    if remaining_max is None:  # day is over: M = running_max exactly
        if running_max is None:
            return 0.0
        return 1.0 if lo <= running_max < hi else 0.0
    # M = max(R, X): P(M < t) = 1[R < t] * P(X < t)
    def cdf_m(t: float) -> float:
        if t == math.inf:
            return 1.0
        if t == -math.inf:
            return 0.0
        if running_max is not None and running_max >= t:
            return 0.0
        return _norm_cdf(t, remaining_max, sd)
    return max(0.0, cdf_m(hi) - cdf_m(lo))


def _parse_ts(v) -> dt.datetime:
    if isinstance(v, dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    return dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))


async def _today_obs(start: dt.datetime, end: dt.datetime) -> list[float]:
    """Observations within [start, end). Handles both datetime and legacy ISO-string `ts` values."""
    q = {"$or": [{"ts": {"$gte": start, "$lt": end}},
                 {"ts": {"$gte": start.astimezone(dt.timezone.utc).isoformat(), "$lt": end.astimezone(dt.timezone.utc).isoformat()}}]}
    temps = []
    async for o in db.observations().find(q, {"_id": 0, "ts": 1, "temp_f": 1}):
        try:
            ts = _parse_ts(o["ts"])
        except (TypeError, ValueError):
            continue
        if start <= ts < end and o.get("temp_f") is not None:
            temps.append(float(o["temp_f"]))
    return temps


async def estimate_today(target_date: str, as_of: dt.datetime | None = None) -> dict:
    as_of = (as_of or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    day_start = dt.datetime.fromisoformat(target_date).replace(tzinfo=ET)
    day_end = day_start + timedelta(days=1)

    temps = await _today_obs(day_start, day_end)
    running_max = round(max(temps)) if temps else None

    fc = await db.forecasts().find_one({"target_date": target_date}, sort=[("fetched_at", -1)])
    ahead: list[float] = []
    if fc:
        for p in fc.get("periods", []):
            try:
                t = _parse_ts(p["t"])
            except (TypeError, ValueError, KeyError):
                continue
            # an hourly period is "still ahead" if it has not fully elapsed
            if day_start <= t < day_end and t + timedelta(hours=1) > as_of and p.get("temp_f") is not None:
                ahead.append(float(p["temp_f"]))
    hours_left = 0 if as_of >= day_end else max(0, math.ceil((day_end - max(as_of, day_start)).total_seconds() / 3600))
    remaining_forecast_max = max(ahead) if ahead else None
    if hours_left == 0:
        remaining_forecast_max = None
    sd = _sd(hours_left)

    snap = await db.market_snapshots().find_one({"target_date": target_date}, sort=[("ts", -1)])
    markets = snap["markets"] if snap else []
    buckets = []
    for m in markets:
        floor, cap = m.get("floor"), m.get("cap")
        p = _p_bucket(floor, cap, running_max, remaining_forecast_max, sd)
        bid, ask = float(m.get("yes_bid") or 0), float(m.get("yes_ask") or 0)
        mid = round((bid + ask) / 2, 3) if (bid or ask) else None
        buckets.append({"label": m.get("label") or kalshi.bucket_of(m), "floor": floor, "cap": cap,
                        "p_model": round(p, 3), "yes_bid": bid, "yes_ask": ask, "mid": mid,
                        "edge_cents": round((p - mid) * 100) if mid is not None else None})
    # normalise model probs so they sum to 1 across the book (tails absorb the rest)
    tot = sum(b["p_model"] for b in buckets)
    if tot > 0 and abs(tot - 1) > 1e-6:
        for b in buckets:
            b["p_model"] = round(b["p_model"] / tot, 3)
            b["edge_cents"] = round((b["p_model"] - b["mid"]) * 100) if b["mid"] is not None else None

    fav_market = (snap or {}).get("favorite_bucket")
    if fav_market is None and markets:
        f = kalshi.favorite(markets)
        fav_market = kalshi.bucket_of(f) if f else None
    fav_model = max(buckets, key=lambda b: b["p_model"])["label"] if buckets else None
    return {"target_date": target_date, "as_of": as_of, "running_max": running_max,
            "remaining_forecast_max": remaining_forecast_max, "hours_left": hours_left, "sd_f": sd,
            "obs_n": len(temps), "forecast_fetched_at": fc.get("fetched_at") if fc else None,
            "market_ts": snap.get("ts") if snap else None,
            "buckets": buckets, "favorite_bucket_market": fav_market, "favorite_bucket_model": fav_model}
