"""Edge: pollers that fetch public data and publish to the bus, plus a thin REST surface.

`uvicorn main:app --port 8000`
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException

import db
import kalshi
import nws
from bus import broker, ensure_groups, ping_redis, publish
from config import settings
from models import ActualMsg, ForecastMsg, ObsMsg, TickMsg, VerdictBody, VerdictMsg
from streams import CMD_VERDICT, MKT_TICK, WX_ACTUAL, WX_FORECAST, WX_OBS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

_HTTP: httpx.AsyncClient | None = None
_TASKS: dict[str, asyncio.Task[None]] = {}


def _http() -> httpx.AsyncClient:
    assert _HTTP is not None, "http client used before lifespan"
    return _HTTP


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _et_dates() -> list[dt.date]:
    today = dt.datetime.now(ET).date()
    return [today, today + dt.timedelta(days=1)]


# ---- pollers ----------------------------------------------------------------------

async def poll_forecast() -> None:
    periods = await nws.hourly_forecast(_http())
    fetched = _now()
    for d in _et_dates():
        target = d.isoformat()
        msg = ForecastMsg(fetched_at=fetched, target_date=target,
                          day_max=nws.day_max(periods, target), periods=periods)
        ok = await publish(WX_FORECAST, msg)
        logger.info("forecast target=%s day_max=%s published=%s", target, msg.day_max, ok)


async def poll_obs() -> None:
    obs = await nws.observations(_http(), limit=24)
    n = 0
    for o in obs:
        if await publish(WX_OBS, ObsMsg(**o)):
            n += 1
    logger.info("obs published %d/%d", n, len(obs))


async def poll_actuals() -> None:
    end = dt.datetime.now(ET).date()
    start = end - dt.timedelta(days=45)
    rows = await nws.ncei_tmax(_http(), start.isoformat(), end.isoformat())
    n = 0
    for r in rows:
        if await publish(WX_ACTUAL, ActualMsg(**r)):
            n += 1
    logger.info("actuals published %d/%d (%s..%s)", n, len(rows), start, end)


async def poll_ticks() -> None:
    for d in _et_dates():
        markets = await kalshi.markets_for(_http(), d)
        fav = kalshi.favorite(markets)
        msg = TickMsg(
            ts=_now(), event_ticker=kalshi.event_ticker(d), target_date=d.isoformat(),
            markets=markets,
            favorite_bucket=kalshi.bucket_of(fav) if fav else None,
            favorite_mid=round((fav["yes_bid"] + fav["yes_ask"]) / 2, 3) if fav else None,
        )
        ok = await publish(MKT_TICK, msg)
        logger.info("tick %s markets=%d fav=%s mid=%s published=%s",
                    msg.event_ticker, len(markets), msg.favorite_bucket, msg.favorite_mid, ok)


async def _loop(name: str, fn: Callable[[], Awaitable[None]], interval: int) -> None:
    """Run `fn` forever. Any exception is logged and the loop keeps going."""
    while True:
        try:
            await fn()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("poller %s failed; retrying in %ss", name, interval)
        await asyncio.sleep(interval)


def _start(name: str, fn: Callable[[], Awaitable[None]], interval: int) -> None:
    _TASKS[name] = asyncio.create_task(_loop(name, fn, interval), name=f"poll-{name}")


async def _stop_all() -> None:
    for task in list(_TASKS.values()):
        task.cancel()
    for task in list(_TASKS.values()):
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    _TASKS.clear()


# ---- app --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _HTTP
    _HTTP = httpx.AsyncClient(headers={"User-Agent": settings.nws_user_agent})
    await broker.connect()
    await ensure_groups()
    _start("forecast", poll_forecast, settings.forecast_interval)
    _start("obs", poll_obs, settings.obs_interval)
    _start("actuals", poll_actuals, settings.actual_interval)
    _start("ticks", poll_ticks, settings.tick_interval)
    logger.info("edge ready; pollers=%s", list(_TASKS))
    try:
        yield
    finally:
        await _stop_all()
        await broker.stop()
        await _HTTP.aclose()
        _HTTP = None
        logger.info("edge stopped")


app = FastAPI(title="bookie edge", version="0.1.0", lifespan=lifespan)

# NiceGUI judge page at /ui. Mounted last (see bottom of file) so every route above is
# registered first; ui.run_with wraps our lifespan rather than replacing it.


def _clean(doc: dict) -> dict:
    doc = dict(doc)
    doc["_id"] = str(doc["_id"])
    return doc


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "redis": await ping_redis(),
        "pollers": {n: (not t.done()) for n, t in _TASKS.items()},
        "ts": _now().isoformat(),
    }


@app.get("/pending")
async def pending() -> list[dict]:
    cur = db.proposals().find({"status": "pending"}).sort("created_at", -1).limit(50)
    return [_clean(d) async for d in cur]


@app.get("/rules")
async def rules() -> dict:
    doc = await db.rules().find_one(sort=[("version", -1)])
    if doc is None:
        raise HTTPException(404, "no rules yet")
    return _clean(doc)


@app.get("/scores")
async def scores() -> list[dict]:
    cur = db.scores().find().sort("target_date", -1).limit(100)
    return [_clean(d) async for d in cur]


@app.post("/verdict/{run_id}")
async def verdict(run_id: str, body: VerdictBody) -> dict:
    msg = VerdictMsg(run_id=run_id, approved=body.approved, note=body.note)
    if not await publish(CMD_VERDICT, msg):
        raise HTTPException(503, "bus unavailable")
    return {"queued": True}


import ui_pages  # noqa: E402  (needs `app` to exist)

ui_pages.mount(app)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
