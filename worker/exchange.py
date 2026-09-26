"""Kalshi exchange client (phase 3, docs/ENGINE.md). Async httpx + RSA-PSS request signing, copied from
worker/kalshi_auth_smoke.py (the smoke test that is known to work against demo).

Order API, confirmed 2026-09-26 from the official docs (https://docs.kalshi.com/llms.txt index):

  * Create order  POST /portfolio/events/orders          https://docs.kalshi.com/api-reference/orders/create-order-v2
      The legacy `POST /portfolio/orders` (action buy|sell, side yes|no, type limit, yes_price/no_price cents,
      *_dollars variants, expiration_ts) is no longer documented (its page 404s) and was scheduled for deprecation
      "no earlier than May 6, 2026", so this client speaks V2 only.
      Request:  ticker (str, required), side ("bid" | "ask", required; YES-leg only: bid = buy YES, ask = sell YES
                == buy NO at 1 - price), count (fixed-point str, e.g. "1.00", required), price (fixed-point dollars
                str, YES price, e.g. "0.5600", required; whole cents are valid in every price_level_structure),
                time_in_force ("good_till_canceled" | "fill_or_kill" | "immediate_or_cancel", required),
                self_trade_prevention_type ("taker_at_cross" | "maker", required), client_order_id (str, optional),
                expiration_time (unix seconds, optional, only with good_till_canceled), post_only (bool),
                cancel_order_on_pause (bool), reduce_only (bool), subaccount (int), order_group_id, exchange_index.
      Response: order_id, client_order_id, fill_count ("0.00"), remaining_count ("1.00"), ts_ms,
                average_fill_price / average_fee_paid (only when fill_count > 0).
  * Cancel        DELETE /portfolio/events/orders/{order_id}   https://docs.kalshi.com/api-reference/orders/cancel-order-v2
      Response: order_id, client_order_id, reduced_by (contracts cancelled), ts_ms.
  * Get order     GET /portfolio/orders/{order_id}             https://docs.kalshi.com/api-reference/orders/get-order
  * List orders   GET /portfolio/orders?status=resting|canceled|executed&ticker=&event_ticker=&limit=&cursor=
      Order: order_id, client_order_id, ticker, outcome_side (yes|no), book_side (bid|ask), type (limit|market),
             status (resting | canceled | executed), yes_price_dollars, no_price_dollars, fill_count_fp,
             remaining_count_fp, initial_count_fp, taker/maker_fees_dollars, taker/maker_fill_cost_dollars,
             expiration_time, created_time. Legacy `action`/`side` still present but deprecated.
  * Fills         GET /portfolio/fills?ticker=&order_id=&min_ts=&limit=&cursor=   .../portfolio/get-fills
      Fill: fill_id, trade_id, order_id, ticker, outcome_side, book_side, count_fp, yes_price_dollars,
            no_price_dollars, is_taker, fee_cost, created_time.
  * Positions     GET /portfolio/positions?ticker=&event_ticker=&count_filter=position&limit=&cursor=
      -> market_positions[], event_positions[]                                    .../portfolio/get-positions
  * Balance       GET /portfolio/balance -> balance (cents), balance_dollars, portfolio_value, updated_ts
  * Public GET /markets (no auth): price fields are fixed-point dollar strings: yes_bid_dollars, yes_ask_dollars,
      no_bid_dollars, no_ask_dollars, last_price_dollars, yes_bid_size_fp, yes_ask_size_fp, volume_fp,
      open_interest_fp, price_level_structure, price_ranges. No integer-cent price fields are returned any more.
      Demo and prod expose the SAME ticker names (checked: KXHIGHNY-26SEP26-{T62,B62.5,...} on both hosts);
      demo prices/liquidity differ.
  * Auth (https://docs.kalshi.com/getting_started/api_environments): sign timestamp_ms + METHOD + full path from the
      API root without query string, e.g. "1712...POST/trade-api/v2/portfolio/events/orders". Headers
      KALSHI-ACCESS-KEY / KALSHI-ACCESS-TIMESTAMP / KALSHI-ACCESS-SIGNATURE. Demo and prod keys are not shared.

Safety: `dry_run` (default: ORDERS_ENABLED != "true") makes place_limit / cancel return
{"dry_run": True, "payload": ...} without any HTTP call. Key material is never logged or returned.
"""
from __future__ import annotations

import base64
import datetime as dt
import os
import time
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

import db
import kalshi

PREFIX = "/trade-api/v2"
BASES = {
    "demo": "https://demo-api.kalshi.co" + PREFIX,             # same host the smoke test uses
    "prod": "https://api.elections.kalshi.com" + PREFIX,
}
Side = Literal["yes", "no"]
Action = Literal["buy", "sell"]


def orders_enabled() -> bool:
    return os.environ.get("ORDERS_ENABLED", "").strip().lower() == "true"


class ExchangeError(RuntimeError):
    def __init__(self, status: int, method: str, path: str, text: str):
        self.status, self.method, self.path, self.text = status, method, path, text
        super().__init__(f"{method} {path} -> {status}: {text[:500]}")


def to_v2(side: Side, action: Action, price_c: int) -> tuple[str, str]:
    """(book_side, yes-leg price string) for a legacy (side, action, price in that side's cents).
    buy yes @p  -> bid @p ; sell yes @p -> ask @p ; buy no @q -> ask @(100-q) ; sell no @q -> bid @(100-q)."""
    if side not in ("yes", "no") or action not in ("buy", "sell"):
        raise ValueError(f"bad side/action {side}/{action}")
    yes_c = int(price_c) if side == "yes" else 100 - int(price_c)
    if not 1 <= yes_c <= 99:
        raise ValueError(f"price {price_c}c on {side} maps to yes {yes_c}c, outside 1..99")
    book_side = "bid" if (side, action) in (("yes", "buy"), ("no", "sell")) else "ask"
    return book_side, f"{yes_c / 100:.4f}"


class KalshiClient:
    def __init__(self, key_id: str, private_key_pem: bytes, env: str = "demo", *, dry_run: bool | None = None,
                 timeout: float = 20.0):
        if env not in BASES:
            raise ValueError(f"KALSHI_ENV must be one of {list(BASES)}, got {env!r}")
        self.env = env
        self.base = BASES[env]
        self.dry_run = (not orders_enabled()) if dry_run is None else bool(dry_run)
        self._kid = key_id
        self._key = serialization.load_pem_private_key(private_key_pem, password=None)
        self._http = httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_env(cls, *, dry_run: bool | None = None) -> "KalshiClient":
        """Reads KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH / KALSHI_ENV (+ ORDERS_ENABLED). The edge process does not
        load worker/.env by itself, so load it here (never overriding real environment variables)."""
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)
        kid = os.environ["KALSHI_API_KEY_ID"]
        with open(os.environ["KALSHI_PRIVATE_KEY_PATH"], "rb") as f:
            pem = f.read()
        return cls(kid, pem, os.environ.get("KALSHI_ENV", "demo"), dry_run=dry_run)

    async def aclose(self) -> None:
        await self._http.aclose()

    def __repr__(self) -> str:      # never include the key id or key material
        return f"KalshiClient(env={self.env}, dry_run={self.dry_run})"

    # ---- signing (identical to kalshi_auth_smoke.py) ----------------------------------------
    def _headers(self, method: str, path: str) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        msg = (ts + method + PREFIX + path).encode()
        sig = self._key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                             hashes.SHA256())
        return {"KALSHI-ACCESS-KEY": self._kid, "KALSHI-ACCESS-TIMESTAMP": ts,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(), "Content-Type": "application/json"}

    async def _req(self, method: str, path: str, *, params: dict | None = None, json: dict | None = None) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        r = await self._http.request(method, self.base + path, params=params, json=json, headers=self._headers(method, path))
        if r.status_code // 100 != 2:
            raise ExchangeError(r.status_code, method, path, r.text)
        return r.json() if r.content else {}

    # ---- reads (GET only) ---------------------------------------------------------------------
    async def balance(self) -> dict:
        return await self._req("GET", "/portfolio/balance")

    async def positions(self, ticker: str | None = None, event_ticker: str | None = None, limit: int = 200) -> dict:
        return await self._req("GET", "/portfolio/positions",
                               params={"ticker": ticker, "event_ticker": event_ticker, "limit": limit})

    async def orders(self, status: str | None = None, ticker: str | None = None, event_ticker: str | None = None,
                     limit: int = 200) -> list[dict]:
        out = await self._req("GET", "/portfolio/orders",
                              params={"status": status, "ticker": ticker, "event_ticker": event_ticker, "limit": limit})
        return out.get("orders", [])

    async def get_order(self, order_id: str) -> dict:
        out = await self._req("GET", f"/portfolio/orders/{order_id}")
        return out.get("order", out)

    async def fills(self, ticker: str | None = None, order_id: str | None = None, min_ts: int | None = None,
                    limit: int = 200) -> list[dict]:
        out = await self._req("GET", "/portfolio/fills",
                              params={"ticker": ticker, "order_id": order_id, "min_ts": min_ts, "limit": limit})
        return out.get("fills", [])

    # ---- writes (dry_run aware) ---------------------------------------------------------------
    def limit_payload(self, ticker: str, side: Side, action: Action, count: int, price_c: int, client_order_id: str,
                      post_only: bool = False, expiration_ts: int | None = None, reduce_only: bool = False) -> dict:
        if int(count) <= 0:
            raise ValueError("count must be >= 1")
        book_side, price = to_v2(side, action, price_c)
        p: dict[str, Any] = {"ticker": ticker, "side": book_side, "count": f"{int(count)}.00", "price": price,
                             "time_in_force": "good_till_canceled", "self_trade_prevention_type": "taker_at_cross",
                             "post_only": bool(post_only), "client_order_id": client_order_id}
        if expiration_ts is not None:
            p["expiration_time"] = int(expiration_ts)
        if reduce_only:                      # closes (phase 3b): the exchange refuses to flip us into a new position
            p["reduce_only"] = True
        return p

    async def place_limit(self, ticker: str, side: Side, action: Action, count: int, price_c: int, client_order_id: str,
                          post_only: bool = False, expiration_ts: int | None = None, reduce_only: bool = False) -> dict:
        payload = self.limit_payload(ticker, side, action, count, price_c, client_order_id, post_only, expiration_ts, reduce_only)
        if self.dry_run:
            return {"dry_run": True, "method": "POST", "path": "/portfolio/events/orders", "payload": payload}
        return await self._req("POST", "/portfolio/events/orders", json=payload)

    async def cancel(self, order_id: str) -> dict:
        if self.dry_run:
            return {"dry_run": True, "method": "DELETE", "path": f"/portfolio/events/orders/{order_id}",
                    "payload": {"order_id": order_id}}
        return await self._req("DELETE", f"/portfolio/events/orders/{order_id}")


# ---- ticker resolution: never guess ticker strings -------------------------------------------------
async def market_ticker_for(target_date: str, bucket_label: str) -> str:
    """Ticker of the bucket `bucket_label` (as stored in market_snapshots.markets[].label, e.g. "61° or below") from
    the latest snapshot for target_date. Raises LookupError if there is no snapshot or no such bucket."""
    snap = await db.market_snapshots().find_one({"target_date": target_date}, sort=[("ts", -1)])
    if not snap:
        raise LookupError(f"no market snapshot for {target_date}")
    want = str(bucket_label).strip()
    for m in snap.get("markets", []):
        if str(m.get("label") or "").strip() == want or kalshi.bucket_of(m) == want:
            if m.get("ticker"):
                return m["ticker"]
    labels = [m.get("label") for m in snap.get("markets", [])]
    raise LookupError(f"bucket {bucket_label!r} not in snapshot for {target_date}; have {labels}")


def event_ticker_for(target_date: str) -> str:
    return kalshi.event_ticker(dt.date.fromisoformat(target_date))
