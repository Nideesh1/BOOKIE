"""Kalshi public market data (no auth needed for reads). Series KXHIGHNY = NYC daily high temperature."""
import httpx, datetime as dt
BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXHIGHNY"

def event_ticker(date: dt.date) -> str:     # 2026-09-27 -> KXHIGHNY-26SEP27
    return f"{SERIES}-{date.strftime('%y%b%d').upper()}"

async def markets_for(client: httpx.AsyncClient, date: dt.date) -> list[dict]:
    r = await client.get(f"{BASE}/markets", params={"event_ticker": event_ticker(date), "limit": 50}, timeout=20)
    r.raise_for_status()
    out = []
    for m in r.json().get("markets", []):
        out.append({"ticker": m["ticker"], "label": m.get("yes_sub_title"), "floor": m.get("floor_strike"), "cap": m.get("cap_strike"),
                    "yes_bid": float(m.get("yes_bid_dollars") or 0), "yes_ask": float(m.get("yes_ask_dollars") or 0),
                    "last": float(m.get("last_price_dollars") or 0), "volume": float(m.get("volume_fp") or 0),
                    "open_interest": float(m.get("open_interest_fp") or 0), "status": m.get("status")})
    return sorted(out, key=lambda x: (x["floor"] if x["floor"] is not None else -999))

def favorite(markets: list[dict]) -> dict | None:
    live = [m for m in markets if m["yes_bid"] or m["yes_ask"]]
    return max(live, key=lambda m: (m["yes_bid"] + m["yes_ask"]) / 2) if live else None

def bucket_of(m: dict) -> str:
    if m["floor"] is None: return f"<={m['cap']-1}" if m["cap"] else "?"
    if m["cap"] is None: return f">={m['floor']+1}"
    return f"{m['floor']}-{m['cap']}"
