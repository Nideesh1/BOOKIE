"""Code clamps (docs/ENGINE.md "Code clamps"): applied in order, each can only shrink or reject.

1. kill switch file/flag            -> reject everything
2. per-day exposure cap             -> shrink size to what the remaining daily budget buys
3. per-bucket cap                   -> shrink size to what the remaining bucket budget buys
4. size <= 20% of visible depth at the limit price
5. limit price inside the spread, never through it by more than max_slippage_c
6. no orders in the last 30 min before settlement lock
7. fees: taker ~ 7% x p x (1-p) per contract; reject if edge after fees < 2c

Every applied clamp appends a human-readable line to clamps_applied so reflect can learn from it.
Prices are cents; dollars at risk = size x price / 100. A "no" order is priced in the no book
(no_bid = 100 - yes_ask, no_ask = 100 - yes_bid) and takes depth from the yes-bid side.
"""
from __future__ import annotations

import math
import os

from pydantic import BaseModel, Field

from .contracts import BucketBook, ClampedOrder, OrderProposal, TickState


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class Caps(BaseModel):
    kill_switch_path: str = Field(default_factory=lambda: os.environ.get("KILL_SWITCH", "/tmp/bookie.kill"))
    daily_cap_usd: float = Field(default_factory=lambda: _env_float("DAILY_CAP_USD", 50))
    bucket_cap_usd: float = Field(default_factory=lambda: _env_float("BUCKET_CAP_USD", 25))
    depth_frac: float = Field(default_factory=lambda: _env_float("DEPTH_FRAC", 0.2))
    lock_minutes: float = Field(default_factory=lambda: _env_float("LOCK_MINUTES", 30))
    min_edge_after_fees_c: float = Field(default_factory=lambda: _env_float("MIN_EDGE_AFTER_FEES_C", 2))
    fee_rate: float = 0.07

    def kill_switch_on(self) -> bool:
        return os.path.exists(self.kill_switch_path) or os.environ.get("BOOKIE_KILL", "").lower() in ("1", "true", "yes")


def fee_usd(limit_price_c: int, rate: float = 0.07) -> float:
    """Kalshi taker fee per contract in dollars: rate x p x (1-p), p = price/100."""
    p = limit_price_c / 100
    return rate * p * (1 - p)


def _reject(proposal: OrderProposal, price: int, applied: list[str], reason: str) -> ClampedOrder:
    applied.append(reason)
    return ClampedOrder(proposal=proposal, size=0, limit_price_c=price, clamps_applied=applied, allowed=False, reason=reason)


def clamp(proposal: OrderProposal, book: TickState | dict[str, BucketBook] | BucketBook, exposure_today: float,
          caps: Caps | None = None, *, hours_left: float | None = None, bucket_exposure_today: float = 0.0,
          edge_c: int | None = None) -> ClampedOrder:
    """Shrink or reject `proposal` against the book. `book` is the TickState (preferred: carries hours_left), the
    buckets dict, or the one BucketBook for proposal.bucket. `edge_c` is the model edge in favour of the proposal's
    side (yes: p_model - p_market; no: p_market - p_model), used by the fee clamp; when None the fee is only reported.
    """
    caps = caps or Caps()
    applied: list[str] = []
    size, price = int(proposal.size), int(proposal.limit_price_c)

    if isinstance(book, TickState):
        hours_left = book.hours_left if hours_left is None else hours_left
        bb = book.buckets.get(proposal.bucket)
    elif isinstance(book, dict):
        bb = book.get(proposal.bucket)
    else:
        bb = book
    if proposal.tactic == "skip" or size <= 0:
        return _reject(proposal, price, applied, "proposal is a skip / zero size; nothing to send")

    # 1. kill switch
    if caps.kill_switch_on():
        return _reject(proposal, price, applied, f"kill switch on ({caps.kill_switch_path}); rejected")

    # 2. per-day exposure cap
    p = max(price, 1) / 100
    remaining_day = caps.daily_cap_usd - exposure_today
    if remaining_day <= 0:
        return _reject(proposal, price, applied, f"daily cap ${caps.daily_cap_usd:.0f} already used (${exposure_today:.2f}); rejected")
    max_by_day = math.floor(remaining_day / p)
    if size > max_by_day:
        applied.append(f"capped at daily exposure: size {size} -> {max_by_day} (${remaining_day:.2f} of ${caps.daily_cap_usd:.0f} left at {price}¢)")
        size = max_by_day

    # 3. per-bucket cap
    remaining_bucket = caps.bucket_cap_usd - bucket_exposure_today
    if remaining_bucket <= 0:
        return _reject(proposal, price, applied, f"bucket cap ${caps.bucket_cap_usd:.0f} already used on {proposal.bucket}; rejected")
    max_by_bucket = math.floor(remaining_bucket / p)
    if size > max_by_bucket:
        applied.append(f"capped at bucket exposure: size {size} -> {max_by_bucket} (${remaining_bucket:.2f} of ${caps.bucket_cap_usd:.0f} left on {proposal.bucket})")
        size = max_by_bucket

    # 4. size <= 20% of visible depth at the limit price
    if bb is None:
        return _reject(proposal, price, applied, f"no book for bucket {proposal.bucket}; rejected")
    if proposal.side == "yes":
        bid, ask, depth_at = bb.bid, bb.ask, bb.ask_size
        depth_side = "ask"
    else:                                   # buying no = hitting the yes bid
        bid, ask, depth_at = 100 - bb.ask, 100 - bb.bid, bb.bid_size
        depth_side = "yes-bid"
    max_by_depth = math.floor(caps.depth_frac * depth_at)
    if max_by_depth <= 0:
        return _reject(proposal, price, applied, f"no visible depth on the {depth_side} side of {proposal.bucket}; rejected")
    if size > max_by_depth:
        applied.append(f"size cut to {caps.depth_frac:.0%} of depth: {size} -> {max_by_depth} ({depth_at} contracts visible at the {depth_side})")
        size = max_by_depth

    # 5. limit price inside the spread, never through it by more than max_slippage_c
    slip = max(0, int(proposal.max_slippage_c))
    hi = min(99, ask + slip) if ask else 99
    lo = max(1, bid)
    if price > hi:
        applied.append(f"limit {price}¢ through the {proposal.side} ask ({ask}¢) by more than max_slippage {slip}¢; set to {hi}¢")
        price = hi
    elif price < lo:
        applied.append(f"limit {price}¢ below the {proposal.side} bid ({bid}¢); set to {lo}¢ (inside the spread)")
        price = lo
    if price != proposal.limit_price_c:          # re-check the dollar caps at the new price
        p = price / 100
        cap_size = min(math.floor(remaining_day / p), math.floor(remaining_bucket / p))
        if size > cap_size:
            applied.append(f"re-capped after price change: size {size} -> {cap_size}")
            size = cap_size

    # 6. no orders in the last 30 min before settlement lock
    if hours_left is not None and hours_left * 60 <= caps.lock_minutes:
        return _reject(proposal, price, applied, f"{hours_left * 60:.0f} min to settlement lock (< {caps.lock_minutes:.0f}); rejected")

    # 7. fees
    fee_c = fee_usd(price, caps.fee_rate) * 100
    if edge_c is None:
        applied.append(f"fee {fee_c:.2f}¢/contract at {price}¢ (edge not supplied; fee check reported only)")
    else:
        net = edge_c - fee_c
        if net < caps.min_edge_after_fees_c:
            return _reject(proposal, price, applied, f"edge after fees {net:.1f}¢ < {caps.min_edge_after_fees_c:.0f}¢ (edge {edge_c}¢, fee {fee_c:.2f}¢); rejected")
        applied.append(f"fee {fee_c:.2f}¢/contract; edge after fees {net:.1f}¢")

    if size <= 0:
        return _reject(proposal, price, applied, "size shrank to 0; rejected")
    return ClampedOrder(proposal=proposal, size=size, limit_price_c=price, clamps_applied=applied, allowed=True,
                        reason=f"{proposal.side} {size} @ {price}¢ on {proposal.bucket} (${size * price / 100:.2f} at risk)")
