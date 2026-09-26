"""Hard risk rules (phase 3b): the code-side position limits that fire BEFORE the execution agent sees anything.

RiskRules is read from env once per use. The dollar / size / lock / kill-switch limits are the same numbers the
clamps use (engine.clamps.Caps is the single source of truth; RiskRules only adds the loss limits on top):

  MAX_LOSS_PCT_PER_POSITION   close a position whose unrealized loss is >= this % of its entry cost   (20)
  DAILY_LOSS_CAP_USD          no NEW opens once today's realized + unrealized P&L <= -cap              (10)
  BUCKET_CAP_USD / MAX_CONTRACTS_PER_ORDER / LOCK_MINUTES / KILL_SWITCH   -> Caps

Everything here is deterministic. Positions are what the exchange says we hold (net per market, YES-leg signed);
marks are the latest mids from tick_state / market_snapshots; the running max comes from the same tick. A forced
action is a close the code has already decided on: the workflow turns it into a decision that skips Jev and the
agent, and execute.place_from_decision sends it through the clamp + kill switch but not the human gate.

P&L is always computed on the YES leg: buying NO at q is selling YES at 100 - q. `ledger_from_fills` is the one
round-trip matcher (average-cost), shared by the daily loss cap, position_pnl and brain.py's trade scoring.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Literal

from pydantic import BaseModel, Field

import db

from .clamps import Caps, _env_float
from .contracts import OrderProposal, TickState

ForcedReason = Literal["stop_loss", "bucket_killed", "settlement_lock"]


class RiskRules(BaseModel):
    caps: Caps = Field(default_factory=Caps)
    max_loss_pct_per_position: float = Field(default_factory=lambda: _env_float("MAX_LOSS_PCT_PER_POSITION", 20))
    daily_loss_cap_usd: float = Field(default_factory=lambda: _env_float("DAILY_LOSS_CAP_USD", 10))

    # the overlapping limits live in Caps; expose them so callers (UI, workflow) read one object
    @property
    def bucket_cap_usd(self) -> float:
        return self.caps.bucket_cap_usd

    @property
    def max_contracts_per_order(self) -> int:
        return self.caps.max_contracts_per_order

    @property
    def lock_minutes(self) -> float:
        return self.caps.lock_minutes

    @property
    def kill_switch_path(self) -> str:
        return self.caps.kill_switch_path

    def kill_switch_on(self) -> bool:
        return self.caps.kill_switch_on()

    def daily_loss_cap_hit(self, pnl_usd: float) -> bool:
        return pnl_usd <= -abs(self.daily_loss_cap_usd)

    def as_rows(self) -> list[tuple[str, str, str]]:
        """(env key, live value, meaning) for the Risk panel."""
        return [
            ("MAX_LOSS_PCT_PER_POSITION", f"{self.max_loss_pct_per_position:.0f}%", "close when unrealized loss >= this % of entry cost"),
            ("DAILY_LOSS_CAP_USD", f"${self.daily_loss_cap_usd:.2f}", "no new opens once realized + unrealized <= -cap"),
            ("BUCKET_CAP_USD", f"${self.bucket_cap_usd:.2f}", "dollars at risk per bucket (clamp 3)"),
            ("MAX_CONTRACTS_PER_ORDER", str(self.max_contracts_per_order), "hard per-order size cap (clamp 3b)"),
            ("LOCK_MINUTES", f"{self.lock_minutes:.0f} min", "no new orders this close to settlement; losing positions are closed"),
            ("KILL_SWITCH", self.kill_switch_path, "file (or BOOKIE_KILL=1) -> reject everything, closes included"),
        ]


# ---- positions & marks ---------------------------------------------------------------------
class Position(BaseModel):
    """One net position on one market. `contracts` is unsigned; `side` says which leg we hold.
    `entry_c` is the average entry price in that side's cents (yes: yes price; no: 100 - yes price)."""
    ticker: str
    bucket: str
    target_date: str | None = None
    side: Literal["yes", "no"]
    contracts: int
    entry_c: float
    fees_usd: float = 0.0
    realized_usd: float = 0.0          # what the exchange / ledger says is already booked on this market
    source: str = "exchange"

    @property
    def ref(self) -> str:
        return f"{self.ticker}:{self.side}"

    @property
    def cost_usd(self) -> float:
        return self.contracts * self.entry_c / 100


class Marks(BaseModel):
    """Current mids (YES cents) per bucket label plus the tick facts the rules need."""
    mid_c: dict[str, float] = Field(default_factory=dict)
    bounds: dict[str, tuple[int | None, int | None]] = Field(default_factory=dict)   # bucket -> (floor, cap)
    running_max_f: float | None = None
    hours_left: float | None = None
    as_of: dt.datetime | None = None

    def mark_for(self, pos: Position) -> float | None:
        m = self.mid_c.get(pos.bucket)
        if m is None:
            return None
        return float(m) if pos.side == "yes" else 100.0 - float(m)


def marks_from_tick(tick: TickState) -> Marks:
    return Marks(mid_c={b: v.mid for b, v in tick.buckets.items() if v.mid is not None},
                 bounds={b: (v.floor, v.cap) for b, v in tick.buckets.items()},
                 running_max_f=tick.running_max_f, hours_left=tick.hours_left, as_of=tick.as_of)


def marks_from_snapshot(snap: dict, running_max_f: float | None = None, hours_left: float | None = None) -> Marks:
    import kalshi
    mid, bounds = {}, {}
    for m in snap.get("markets", []):
        label = m.get("label") or kalshi.bucket_of(m)
        bid, ask = float(m.get("yes_bid") or 0) * 100, float(m.get("yes_ask") or 0) * 100
        if bid or ask:
            mid[label] = (bid + ask) / 2
        bounds[label] = (m.get("floor"), m.get("cap"))
    return Marks(mid_c=mid, bounds=bounds, running_max_f=running_max_f, hours_left=hours_left, as_of=snap.get("ts"))


class PositionPnl(BaseModel):
    position: Position
    mark_c: float | None
    unrealized_c: float | None            # total cents across the position
    unrealized_pct: float | None          # of entry cost, + = profit
    unrealized_usd: float | None


def position_pnl(pos: Position, marks: Marks) -> PositionPnl:
    mark = marks.mark_for(pos)
    if mark is None:
        return PositionPnl(position=pos, mark_c=None, unrealized_c=None, unrealized_pct=None, unrealized_usd=None)
    per = mark - pos.entry_c
    total_c = per * pos.contracts
    pct = (per / pos.entry_c * 100) if pos.entry_c > 0 else None
    return PositionPnl(position=pos, mark_c=round(mark, 2), unrealized_c=round(total_c, 2),
                       unrealized_pct=round(pct, 1) if pct is not None else None, unrealized_usd=round(total_c / 100, 4))


# ---- bucket geometry (same tail encoding as intraday._p_bucket) -----------------------------
def bucket_killed_for_yes(bounds: tuple[int | None, int | None], running_max_f: float | None) -> bool:
    """The running max already exceeds the bucket: a YES on it can no longer pay."""
    if running_max_f is None:
        return False
    floor, cap = bounds
    r = round(running_max_f)
    if cap is None:
        return False                                   # "70 or above" can never be overshot
    if floor is None:
        return r >= cap                                # "61 or below" = max <= cap-1
    return r > cap


def bucket_locked_in(bounds: tuple[int | None, int | None], running_max_f: float | None, hours_left: float | None) -> bool:
    """The bucket can no longer be avoided: a NO on it can no longer pay."""
    if running_max_f is None:
        return False
    floor, cap = bounds
    r = round(running_max_f)
    if cap is None and floor is not None and r >= floor + 1:
        return True                                    # at or above the open-top tail: it cannot come back down
    if hours_left is not None and hours_left <= 0:
        if floor is None and cap is not None:
            return r <= cap - 1
        if floor is not None and cap is not None:
            return floor <= r <= cap
    return False


# ---- forced actions --------------------------------------------------------------------------
class ForcedAction(BaseModel):
    position: Position
    reason: ForcedReason
    detail: str
    pnl: PositionPnl
    close_side: Literal["bid", "ask"]     # V2 book side of the closing order on the YES leg
    close_price_c: int                    # limit in the OPPOSITE side's cents (what the OrderProposal carries)
    proposal: OrderProposal

    def summary(self) -> str:
        p = self.position
        return (f"{self.reason}: {p.side} {p.contracts} on {p.bucket} @ {p.entry_c:.0f}¢, mark {self.pnl.mark_c}¢ "
                f"({self.pnl.unrealized_pct if self.pnl.unrealized_pct is not None else '-'}%) — {self.detail}")


def close_proposal(pos: Position, marks: Marks, reasoning: str, action: Literal["reduce", "close"] = "close",
                   size: int | None = None, slippage_c: int = 3) -> OrderProposal:
    """The order that flattens `pos`: buy the opposite leg (YES long -> buy NO = ask on the yes leg; NO long -> buy YES
    = bid on the yes leg) at the current mark in that leg's cents. exchange.to_v2 turns it into side/price."""
    opp: Literal["yes", "no"] = "no" if pos.side == "yes" else "yes"
    yes_mid = marks.mid_c.get(pos.bucket)
    if yes_mid is None:
        price = 100 - int(round(pos.entry_c)) if pos.side == "yes" else int(round(pos.entry_c))
    else:
        price = int(round(100 - yes_mid)) if opp == "no" else int(round(yes_mid))
    price = max(1, min(99, price))
    return OrderProposal(bucket=pos.bucket, side=opp, limit_price_c=price, size=int(size or pos.contracts), tactic="cross_now",
                         max_slippage_c=slippage_c, reasoning=reasoning, action=action, position_ref=pos.ref)


def evaluate_positions(positions: list[Position], marks: Marks, rules: RiskRules | None = None) -> list[ForcedAction]:
    """Forced closes from live positions + current marks + the running max. One action per position at most;
    reasons in priority order: bucket_killed (certain loss) > stop_loss > settlement_lock."""
    rules = rules or RiskRules()
    out: list[ForcedAction] = []
    for pos in positions:
        if pos.contracts <= 0:
            continue
        pnl = position_pnl(pos, marks)
        bounds = marks.bounds.get(pos.bucket, (None, None))
        reason: ForcedReason | None = None
        detail = ""
        if pos.side == "yes" and bucket_killed_for_yes(bounds, marks.running_max_f):
            reason, detail = "bucket_killed", f"running max {marks.running_max_f} already exceeds {pos.bucket}; YES cannot pay"
        elif pos.side == "no" and bucket_locked_in(bounds, marks.running_max_f, marks.hours_left):
            reason, detail = "bucket_killed", f"running max {marks.running_max_f} locks {pos.bucket} in; NO cannot pay"
        elif pnl.unrealized_pct is not None and -pnl.unrealized_pct >= rules.max_loss_pct_per_position:
            reason, detail = "stop_loss", (f"unrealized {pnl.unrealized_pct}% <= -{rules.max_loss_pct_per_position:.0f}% "
                                           f"(entry {pos.entry_c:.0f}¢, mark {pnl.mark_c}¢)")
        elif (marks.hours_left is not None and 0 < marks.hours_left * 60 <= rules.lock_minutes
              and pnl.unrealized_c is not None and pnl.unrealized_c < 0):
            reason, detail = "settlement_lock", (f"{marks.hours_left * 60:.0f} min to settlement lock and the position is "
                                                 f"under water ({pnl.unrealized_c:.0f}¢); last chance to exit")
        if reason is None:
            continue
        prop = close_proposal(pos, marks, reasoning=f"forced {reason}: {detail}")
        out.append(ForcedAction(position=pos, reason=reason, detail=detail, pnl=pnl,
                                close_side="ask" if pos.side == "yes" else "bid", close_price_c=prop.limit_price_c, proposal=prop))
    return out


# ---- fills -> ledger (YES leg, average cost) --------------------------------------------------
def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def fill_yes_leg(fill: dict, order: dict | None = None) -> tuple[int, float] | None:
    """(signed contracts on the YES leg, YES price in cents) for one fill. +n = bought YES (or sold NO);
    -n = bought NO (or sold YES). Uses the fill's own side fields, falling back to the parent order doc."""
    n = _f(fill.get("count", fill.get("count_fp")))
    if n <= 0:
        return None
    yes_c = _f(fill.get("yes_price_dollars"), -1) * 100
    if yes_c < 0 and fill.get("no_price_dollars") is not None:
        yes_c = 100 - _f(fill["no_price_dollars"]) * 100
    if yes_c < 0 and order is not None:
        yes_c = _f(order.get("price_c")) if order.get("side") == "yes" else 100 - _f(order.get("price_c"))
    side = fill.get("outcome_side") or (order or {}).get("side")
    action = fill.get("action") or (order or {}).get("action") or "buy"
    book = fill.get("book_side")
    if book in ("bid", "ask"):
        sign = 1 if book == "bid" else -1
    elif side in ("yes", "no"):
        sign = 1 if (side, action) in (("yes", "buy"), ("no", "sell")) else -1
    else:
        return None
    return sign * int(round(n)), yes_c


def ledger_from_fills(legs: list[tuple[int, float]]) -> dict:
    """Average-cost ledger over YES-leg fills in time order. Returns realized_c (closed round trips), the open signed
    position and its average YES price. A NO long shows as a negative position."""
    pos, avg, realized = 0, 0.0, 0.0
    for n, px in legs:
        if pos == 0 or (pos > 0) == (n > 0):              # opening / adding
            avg = (abs(pos) * avg + abs(n) * px) / (abs(pos) + abs(n))
            pos += n
            continue
        closed = min(abs(pos), abs(n))                      # reducing / flipping
        realized += closed * (px - avg) * (1 if pos > 0 else -1)
        pos += n
        if pos != 0 and (pos > 0) == (n > 0):               # flipped through zero: the remainder opens at px
            avg = px
        elif pos == 0:
            avg = 0.0
    return {"realized_c": round(realized, 2), "open": pos, "avg_yes_c": round(avg, 2)}


def settlement_c(open_signed: int, avg_yes_c: float, bucket_hit: bool) -> float:
    """Cents earned by a position held to settlement: YES pays 100 if the bucket hit, NO pays 100 if it missed."""
    settle = 100.0 if bucket_hit else 0.0
    return round(open_signed * (settle - avg_yes_c), 2)


async def our_fills(target_date: str | None = None, ticker: str | None = None) -> dict[str, list[tuple[dt.datetime | str, dict, dict]]]:
    """Fills from db.orders() grouped per ticker, oldest first: (time, fill, order doc)."""
    q: dict = {"fill_count": {"$gt": 0}}
    if target_date:
        q["target_date"] = target_date
    if ticker:
        q["ticker"] = ticker
    by_ticker: dict[str, list] = {}
    async for o in db.orders().find(q):
        for f in o.get("fills") or []:
            by_ticker.setdefault(o["ticker"], []).append((f.get("created_time") or o.get("placed_at") or o.get("created_at") or "", f, o))
    for v in by_ticker.values():
        v.sort(key=lambda x: str(x[0]))
    return by_ticker


async def realized_today_usd(target_date: str) -> tuple[float, float]:
    """(realized P&L, fees) in dollars from our fills on target_date's event, round trips only."""
    realized_c = fees = 0.0
    for legs in (await our_fills(target_date)).values():
        parsed = [fill_yes_leg(f, o) for _, f, o in legs]
        realized_c += ledger_from_fills([p for p in parsed if p])["realized_c"]
        fees += sum(_f(f.get("fee_cost")) for _, f, _ in legs)
    return round(realized_c / 100, 4), round(fees, 4)


# ---- exchange positions -> Position ---------------------------------------------------------
async def _bucket_index(target_date: str | None) -> dict[str, tuple[str, str]]:
    """ticker -> (bucket label, target_date) from the latest snapshot(s)."""
    out: dict[str, tuple[str, str]] = {}
    q = {"target_date": target_date} if target_date else {}
    seen: set[str] = set()
    async for s in db.market_snapshots().find(q, {"target_date": 1, "markets.ticker": 1, "markets.label": 1}).sort("ts", -1).limit(400):
        if s["target_date"] in seen:
            continue
        seen.add(s["target_date"])
        for m in s.get("markets", []):
            if m.get("ticker"):
                out[m["ticker"]] = (m.get("label") or m["ticker"], s["target_date"])
        if target_date or len(seen) >= 3:
            break
    return out


async def positions_from_exchange(client, target_date: str | None = None) -> list[Position]:
    """Live net positions (GET /portfolio/positions) as Position objects. Entry price comes from our own fills when
    we have them, else from the exchange's market_exposure (cost basis) / contracts."""
    raw = await client.positions(event_ticker=None)
    mp = (raw or {}).get("market_positions") or []
    idx = await _bucket_index(target_date)
    fills = await our_fills(target_date)
    out: list[Position] = []
    for p in mp:
        n = _f(p.get("position_fp", p.get("position")))
        if n == 0:
            continue
        ticker = p.get("ticker")
        bucket, tdate = idx.get(ticker, (ticker, None))
        if target_date and tdate not in (None, target_date):
            continue
        side: Literal["yes", "no"] = "yes" if n > 0 else "no"
        contracts = int(round(abs(n)))
        legs = [fill_yes_leg(f, o) for _, f, o in fills.get(ticker, [])]
        led = ledger_from_fills([l for l in legs if l])
        if led["open"] != 0 and (led["open"] > 0) == (n > 0):
            entry = led["avg_yes_c"] if side == "yes" else 100 - led["avg_yes_c"]
            src = "fills"
        else:
            exposure = _f(p.get("market_exposure_dollars"), _f(p.get("market_exposure")) / 100)
            entry = exposure / contracts * 100 if contracts else 0.0
            src = "exchange"
        out.append(Position(ticker=ticker, bucket=bucket, target_date=tdate or target_date, side=side, contracts=contracts,
                            entry_c=round(entry, 2), fees_usd=_f(p.get("fees_paid_dollars")),
                            realized_usd=_f(p.get("realized_pnl_dollars"), _f(p.get("realized_pnl")) / 100), source=src))
    return out


async def daily_pnl(target_date: str, positions: list[Position], marks: Marks) -> dict:
    realized, fees = await realized_today_usd(target_date)
    unreal = 0.0
    for pos in positions:
        if pos.target_date not in (None, target_date):
            continue
        u = position_pnl(pos, marks).unrealized_usd
        unreal += u or 0.0
    total = round(realized - fees + unreal, 4)
    return {"target_date": target_date, "realized_usd": realized, "fees_usd": fees, "unrealized_usd": round(unreal, 4), "total_usd": total}


def marks_now_for(tick: TickState | None, snap: dict | None, running_max_f: float | None = None, hours_left: float | None = None) -> Marks:
    if tick is not None:
        return marks_from_tick(tick)
    if snap is not None:
        return marks_from_snapshot(snap, running_max_f, hours_left)
    return Marks()


__all__ = ["RiskRules", "Position", "Marks", "PositionPnl", "ForcedAction", "evaluate_positions", "position_pnl",
           "close_proposal", "marks_from_tick", "marks_from_snapshot", "ledger_from_fills", "fill_yes_leg", "settlement_c",
           "our_fills", "realized_today_usd", "positions_from_exchange", "daily_pnl", "bucket_killed_for_yes", "bucket_locked_in"]
