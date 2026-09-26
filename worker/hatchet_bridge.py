"""The only place the FastStream worker talks to Hatchet. Two calls, both over Hatchet's gRPC client.

push_verdict           -> wakes a durable `gate` task parked in aio_wait_for_event
maybe_start_market_day -> starts a durable run when a fresh forecast arrives for a day we haven't called yet
"""
import datetime as dt, logging, os
from hatchet_sdk import Hatchet
from hatchet_sdk.types.trigger import PushEventOptions
import db

log = logging.getLogger("bookie.bridge")
VERDICT_EVENT = "bookie:verdict"
PROPOSE_AFTER_ET_HOUR = int(os.environ.get("PROPOSE_AFTER_ET_HOUR", "0"))   # 0 = any time (demo); set 15 for prod
from zoneinfo import ZoneInfo
ET = ZoneInfo("America/New_York")
_h: Hatchet | None = None


def hatchet() -> Hatchet:
    global _h
    if _h is None:
        _h = Hatchet()
    return _h


async def push_verdict(run_id: str, approved: bool, note: str) -> bool:
    try:
        await hatchet().event.aio_push(VERDICT_EVENT, {"approved": approved, "note": note}, options=PushEventOptions(scope=run_id))
        return True
    except Exception:
        log.exception("verdict push failed run_id=%s", run_id)
        return False


async def maybe_start_market_day(target_date: str) -> str | None:
    """Idempotent: one proposal per target day. Returns the run id if a run was started."""
    if dt.datetime.now(ET).hour < PROPOSE_AFTER_ET_HOUR:
        return None
    if await db.proposals().find_one({"target_date": target_date}):
        return None
    try:
        ref = await hatchet().runs.aio_create(workflow_name="market_day", input={"target_date": target_date})
        rid = ref.run.metadata.id if hasattr(ref, "run") else getattr(ref, "workflow_run_id", str(ref))
        log.info("market_day started target=%s run=%s", target_date, rid)
        return rid
    except Exception:
        log.exception("could not start market_day for %s", target_date)
        return None
