"""Typed contracts for the phase-2 engine (docs/ENGINE.md "Typed contracts", verbatim) plus TickState.

Numbers in these objects come from tools / code, not from the model's head; the model adds the narrative fields.
Prices are in cents (int) unless a field name says otherwise; probabilities are 0..1.
"""
from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field


# ---- ENGINE.md contracts (verbatim) -----------------------------------------------------
class SimilarDay(BaseModel):
    date: str
    called: str
    actual_f: int
    hit: bool
    lesson: str


class WeatherView(BaseModel):
    running_max_f: float
    last_obs_f: float
    last_obs_age_min: int
    trend_f_per_hr: float
    ceiling_f: float
    ceiling_p10_f: float
    ceiling_p90_f: float
    p_new_high: float                    # chance the running max gets beaten at all
    hours_of_risk: list[str]             # ["14:00-16:00"] windows where a new high is plausible
    regime: str                          # post-frontal, clear-sky heating, convective, marine layer, ...
    forecast_bias_note: str              # what NWS has done vs obs today
    similar_days: list[SimilarDay]
    p_by_bucket: dict[str, float]        # weather-only probabilities
    key_uncertainty: str


class MarketView(BaseModel):
    crowd_p: dict[str, float]            # implied p per bucket from mids
    depth: dict[str, tuple[int, int]]    # (bid size, ask size) per bucket
    spread_c: dict[str, int]
    drift_since_last_view_c: dict[str, int]
    drift_last_hour_c: dict[str, int]
    thin_buckets: list[str]              # a small order moves price
    stale_buckets: list[str]             # no trades in N min
    volume_today: float
    open_interest: float
    where_money_went: str
    crowd_confidence: str


class Gap(BaseModel):
    bucket: str
    p_model: float
    p_market: float
    edge_c: int
    believed: bool
    why: str


class View(BaseModel):
    target_date: str
    as_of: str
    hours_left: float
    p_by_bucket: dict[str, float]
    gaps: list[Gap]
    confidence: float
    what_would_change_my_mind: str
    rationale: str


class OrderProposal(BaseModel):
    bucket: str
    side: Literal["yes", "no"]
    limit_price_c: int
    size: int
    tactic: Literal["post_and_wait", "cross_now", "ladder", "skip"]
    max_slippage_c: int
    reasoning: str
    # phase 3b position management. open/add = buy `side`; reduce/close = flatten an existing position by buying the
    # OPPOSITE leg (`side` is that opposite leg, price in its cents), so a YES long is closed with side="no" -> V2 ask.
    action: Literal["open", "add", "reduce", "close"] = "open"
    position_ref: str | None = None       # "<ticker>:<yes|no>" of the position this reduces / closes

    @property
    def is_close(self) -> bool:
        return self.action in ("reduce", "close")


class ClampedOrder(BaseModel):            # what code actually sends
    proposal: OrderProposal
    size: int
    limit_price_c: int                     # after caps
    clamps_applied: list[str]              # "size cut to 20% of depth", "capped at daily exposure"
    allowed: bool
    reason: str


# ---- TickState: what code computes every 15 s --------------------------------------------
class BucketBook(BaseModel):
    """Book shape of one bucket. Prices in cents; sizes in contracts (0 when depth is unknown)."""
    bid: int
    ask: int
    mid: float | None = None
    bid_size: int = 0
    ask_size: int = 0
    spread_c: int
    implied_p: float | None = None
    floor: int | None = None
    cap: int | None = None
    ticker: str | None = None


class TickState(BaseModel):
    target_date: str
    as_of: dt.datetime
    market_ts: dt.datetime | None = None
    buckets: dict[str, BucketBook] = Field(default_factory=dict)    # keyed by Kalshi label, e.g. "64° to 65°"
    # weather
    running_max_f: float | None = None
    running_max_bucket: str | None = None                            # label of the bucket the running max sits in
    last_obs_f: float | None = None
    last_obs_age_min: int | None = None
    trend_f_per_hr: float | None = None
    remaining_forecast_max_f: float | None = None
    hours_left: float = 0.0
    forecast_fetched_at: dt.datetime | None = None
    # deltas vs last view (all zero / empty / False when there is no prior view)
    last_view_at: dt.datetime | None = None
    last_view_age_s: float | None = None
    mid_delta_c: dict[str, int] = Field(default_factory=dict)
    depth_flipped: list[str] = Field(default_factory=list)
    obs_drift_f: float | None = None                                 # last obs minus the NWS curve at that hour
    boundary_crossed: bool = False
    new_forecast: bool = False

    def depth(self) -> dict[str, tuple[int, int]]:
        return {k: (b.bid_size, b.ask_size) for k, b in self.buckets.items()}

    def deltas(self) -> dict:
        """The compact "what changed" block that Jev sees for the re-think question."""
        return {
            "last_view_age_s": self.last_view_age_s,
            "boundary_crossed": self.boundary_crossed,
            "running_max_f": self.running_max_f,
            "running_max_bucket": self.running_max_bucket,
            "mid_delta_c": self.mid_delta_c,
            "max_abs_mid_delta_c": max((abs(v) for v in self.mid_delta_c.values()), default=0),
            "depth_flipped": self.depth_flipped,
            "obs_drift_f": self.obs_drift_f,
            "new_forecast": self.new_forecast,
            "hours_left": self.hours_left,
        }
