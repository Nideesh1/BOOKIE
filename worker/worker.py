"""Brain-side consumer. `faststream run worker:app`.

Every handler is idempotent (upsert on a natural key or insert of an immutable snapshot)
because the consumer group guarantees at-least-once delivery. Handler bodies never
re-raise: a poison message is logged and acked, not retried forever.
"""
from __future__ import annotations

import datetime as dt
import logging

from faststream import FastStream
from faststream.redis import StreamSub

import db
from bus import broker, ensure_groups
from models import ActualMsg, ForecastMsg, ObsMsg, TickMsg, VerdictMsg
from streams import CMD_VERDICT, GROUP, MKT_TICK, WX_ACTUAL, WX_FORECAST, WX_OBS, consumer_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastStream(broker)


def _sub(stream: str, max_records: int = 1) -> StreamSub:
    return StreamSub(stream, group=GROUP, consumer=consumer_name(), max_records=max_records)


@app.on_startup
async def _groups() -> None:
    # Groups must exist at id 0 before FastStream's subscribers create them at `$`.
    await broker.connect()
    await ensure_groups()


@app.after_startup
async def start() -> None:
    await db.ensure_indexes()
    logger.info("Worker ready · group=%s consumer=%s", GROUP, consumer_name())


@app.on_shutdown
async def stop() -> None:
    await db.client().close()


@broker.subscriber(stream=_sub(WX_FORECAST))
async def on_forecast(msg: ForecastMsg) -> None:
    """Forecast snapshots are immutable; dedupe on (target_date, fetched_at)."""
    try:
        await db.forecasts().update_one(
            {"target_date": msg.target_date, "fetched_at": msg.fetched_at},
            {"$setOnInsert": msg.model_dump()},
            upsert=True,
        )
        logger.info("forecast stored target=%s day_max=%s", msg.target_date, msg.day_max)
    except Exception:
        logger.exception("forecast handler failed target=%s", msg.target_date)


@broker.subscriber(stream=_sub(WX_OBS, max_records=10))
async def on_obs(msg: ObsMsg) -> None:
    try:
        await db.observations().update_one(
            {"ts": msg.ts}, {"$set": {"temp_f": msg.temp_f}}, upsert=True
        )
    except Exception:
        logger.exception("obs handler failed ts=%s", msg.ts)


@broker.subscriber(stream=_sub(WX_ACTUAL, max_records=10))
async def on_actual(msg: ActualMsg) -> None:
    try:
        await db.actuals().update_one(
            {"date": msg.date}, {"$set": {"tmax_f": msg.tmax_f}}, upsert=True
        )
    except Exception:
        logger.exception("actual handler failed date=%s", msg.date)


@broker.subscriber(stream=_sub(MKT_TICK))
async def on_tick(msg: TickMsg) -> None:
    """Book snapshots are immutable; dedupe on (target_date, ts)."""
    try:
        await db.market_snapshots().update_one(
            {"target_date": msg.target_date, "ts": msg.ts},
            {"$setOnInsert": msg.model_dump()},
            upsert=True,
        )
        logger.info("tick stored %s fav=%s mid=%s", msg.event_ticker, msg.favorite_bucket, msg.favorite_mid)
    except Exception:
        logger.exception("tick handler failed event=%s", msg.event_ticker)


@broker.subscriber(stream=_sub(CMD_VERDICT))
async def on_verdict(msg: VerdictMsg) -> None:
    try:
        await db.proposals().update_one(
            {"run_id": msg.run_id},
            {"$set": {
                "status": "approved" if msg.approved else "rejected",
                "note": msg.note,
                "decided_at": dt.datetime.now(dt.timezone.utc),
            }},
            upsert=True,
        )
        logger.info("verdict stored run_id=%s approved=%s", msg.run_id, msg.approved)
        # TODO(brain): hatchet.event.push("proposal:verdict", msg.model_dump()) to resume the durable run
    except Exception:
        logger.exception("verdict handler failed run_id=%s", msg.run_id)
