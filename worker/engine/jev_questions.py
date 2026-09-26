"""The four Jev questions (docs/ENGINE.md "Jev questions"; the wording there is the prompt, verbatim).

Jev (TypeSafe System One) via OpenRouter's decisions endpoint, same request pattern as jev.py. Every function
is graceful: on any failure it returns the conservative answer (0.0 / "skip" / 0.0 / 0.0) so the loop never
acts because the gate was unreachable.
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx

from .contracts import ClampedOrder, Gap, TickState

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
RETHINK_COOLDOWN_S = 120          # 2 min between views, except a boundary cross always wakes the agent
SAFE_THRESHOLD = 0.8


def _jsonable(v: Any) -> Any:
    return json.loads(json.dumps(v, default=str))


async def ask(state: dict, questions: dict) -> dict:
    """POST one decisions request; returns the `answers` dict. Raises on any failure."""
    body = {"model": MODEL, "state": _jsonable(state), "questions": questions}
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(URL, headers={"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}"}, json=body)
        r.raise_for_status()
        return r.json()["answers"]


# ---- re-think (noul, every tick) ---------------------------------------------------------
RETHINK_Q = {
    "rethink": {
        "type": "noul",
        "instructions": "Has the situation changed enough since the last view that the agent should re-think?",
        "criteria": {
            "true": ("running max crossed a bucket boundary; any bucket mid moved ≥ 4¢ or depth flipped sides; "
                     "observations drifted ≥ 2°F from the curve; a new NWS run landed."),
            "false": "the book and the readings are within noise of the last view.",
        },
    }
}


async def should_rethink(tick: TickState, last_view_age_s: float | None) -> float:
    """P(re-think). Enforces the cooldown in code: within 2 min of the last view only a boundary cross can wake the agent."""
    if last_view_age_s is not None and last_view_age_s < RETHINK_COOLDOWN_S and not tick.boundary_crossed:
        return 0.0
    state = {**tick.deltas(), "last_view_age_s": last_view_age_s, "has_prior_view": last_view_age_s is not None,
             "book": {b: {"mid_c": v.mid, "bid_size": v.bid_size, "ask_size": v.ask_size} for b, v in tick.buckets.items()},
             "last_obs_f": tick.last_obs_f, "trend_f_per_hr": tick.trend_f_per_hr,
             "remaining_forecast_max_f": tick.remaining_forecast_max_f}
    try:
        return float((await ask(state, RETHINK_Q))["rethink"]["noul"])
    except Exception:
        return 0.0


# ---- act / watch / skip (choice, per believed gap) -----------------------------------------
ACT_Q = {
    "action": {
        "type": "choice",
        "instructions": "For this believed gap between the model and the market, should the agent act, watch, or skip?",
        "criteria": {
            "act": "edge ≥ 8¢ and depth ≥ 20 contracts at that price and hours_left > 1.",
            "watch": "edge 4–8¢, or edge is large but the book is thin or stale.",
            "skip": "edge < 4¢, or the gap is on a bucket the running max already killed.",
        },
    }
}


async def act_watch_skip(gap: Gap, depth: tuple[int, int] | dict, spread_c: int, hours_left: float,
                         running_max_f: float | None = None, bucket_killed: bool | None = None) -> tuple[str, dict]:
    """Returns (choice, probabilities). choice is one of act / watch / skip; skip on any failure."""
    if isinstance(depth, dict):
        d = {"bid_size": depth.get("bid_size", 0), "ask_size": depth.get("ask_size", 0)}
    else:
        d = {"bid_size": depth[0], "ask_size": depth[1]}
    side_depth = d["ask_size"] if gap.edge_c > 0 else d["bid_size"]
    state = {"gap": gap.model_dump(), "abs_edge_c": abs(gap.edge_c), "depth": d, "depth_at_that_price": side_depth,
             "spread_c": spread_c, "hours_left": hours_left, "running_max_f": running_max_f,
             "bucket_killed_by_running_max": bucket_killed}
    try:
        a = (await ask(state, ACT_Q))["action"]
        choice = a.get("choice")
        probs = a.get("probabilities") or {}
        if choice not in ("act", "watch", "skip"):
            return "skip", probs
        return choice, probs
    except Exception:
        return "skip", {}


# ---- safe without a human (noul) ---------------------------------------------------------
SAFE_Q = {
    "safe": {
        "type": "noul",
        "instructions": "Is this clamped order safe to record without a human looking at it?",
        "criteria": {
            "true": ("confidence ≥ 0.7, size ≤ 25% of remaining daily cap, edge ≥ 8¢, tactic is post_and_wait or cross_now, "
                     "hit rate over the last 10 not below 40%."),
            "false": "anything else.",
        },
    }
}


async def safe_without_human(clamped: ClampedOrder, view_conf: float, recent_hit_rate: float | None,
                             cap_used_frac: float, edge_c: int | None = None) -> float:
    """P(safe). Compare against SAFE_THRESHOLD (0.8). 0.0 on any failure or when the clamp already rejected it."""
    if not clamped.allowed:
        return 0.0
    p = clamped.limit_price_c / 100
    state = {"order": {"bucket": clamped.proposal.bucket, "side": clamped.proposal.side, "limit_price_c": clamped.limit_price_c,
                       "size": clamped.size, "tactic": clamped.proposal.tactic, "dollars_at_risk": round(clamped.size * p, 2),
                       "clamps_applied": clamped.clamps_applied, "proposed_size": clamped.proposal.size},
             "view_confidence": view_conf, "recent_hit_rate_last_10": recent_hit_rate,
             "daily_cap_used_frac": cap_used_frac, "size_vs_remaining_daily_cap_frac": cap_used_frac,
             "edge_c": edge_c}
    try:
        return float((await ask(state, SAFE_Q))["safe"]["noul"])
    except Exception:
        return 0.0


# ---- enough signal to reflect (noul, nightly) ---------------------------------------------
REFLECT_Q = {
    "reflect": {
        "type": "noul",
        "instructions": "Is there enough new signal in today's scored views and orders to change the rules?",
        "criteria": {
            "true": ("≥ 3 new graded outcomes since the last rulebook version, or one miss ≥ 5°F, "
                     "or a fill got run over by ≥ 10¢."),
            "false": "fewer graded outcomes than that and no large miss or run-over fill; changing rules now would be fitting noise.",
        },
    }
}


async def enough_to_reflect(summary: dict) -> float:
    """P(enough signal). `summary` is code's digest of today's scored views and orders. 0.0 on any failure."""
    try:
        return float((await ask(summary, REFLECT_Q))["reflect"]["noul"])
    except Exception:
        return 0.0
