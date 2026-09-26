"""Redis Streams bus — the single FastStream broker shared by the edge and the brain.

`publish` never raises: a degraded bus is an operational problem, not a caller problem.
"""
from __future__ import annotations

import logging
from typing import Any

from faststream.redis import RedisBroker
from pydantic import BaseModel

from config import settings
from streams import ALL, GROUP, MAXLEN

logger = logging.getLogger(__name__)

# One broker per process. main.py connects it in its lifespan; worker.py hands it to
# a FastStream app, which owns its lifecycle there.
broker: RedisBroker = RedisBroker(settings.redis_url)


async def publish(stream: str, payload: BaseModel | dict[str, Any]) -> bool:
    """Publish to a Redis Stream with approximate trimming. Never raises."""
    try:
        body = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        await broker.publish(body, stream=stream, maxlen=MAXLEN)
        return True
    except Exception:
        logger.exception("Could not publish to stream %s", stream)
        return False


async def ping_redis() -> bool:
    try:
        await broker.ping(timeout=2.0)
        return True
    except Exception:
        logger.warning("Redis ping failed", exc_info=True)
        return False


async def ensure_groups() -> None:
    """Create the consumer group on every stream from id 0, so messages published before
    a consumer first starts are still delivered. Safe to call from any process."""
    client = broker._connection  # underlying redis.asyncio client (set after connect())
    if client is None:
        logger.warning("ensure_groups called before broker.connect(); skipped")
        return
    for stream in ALL:
        try:
            await client.xgroup_create(name=stream, groupname=GROUP, id="0", mkstream=True)
        except Exception as e:  # BUSYGROUP = already exists
            if "BUSYGROUP" not in str(e) and "already exists" not in str(e):
                logger.warning("xgroup_create %s failed: %s", stream, e)
