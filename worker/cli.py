"""bookie CLI: trigger a day, approve/reject, score."""
import sys
from dotenv import load_dotenv; load_dotenv(".env")
from hatchet_sdk.types.trigger import PushEventOptions
from app import hatchet, market_day, DayInput, VERDICT_EVENT, score_and_reflect, ScoreInput, poll_nws

cmd = sys.argv[1]
if cmd == "propose":
    ref = market_day.run_no_wait(DayInput(target_date=sys.argv[2] if len(sys.argv) > 2 else None))
    print("run", ref.workflow_run_id); print(f"approve: uv run python cli.py approve {ref.workflow_run_id} 'ok'")
elif cmd in ("approve", "reject"):
    hatchet.event.push(VERDICT_EVENT, {"approved": cmd == "approve", "note": " ".join(sys.argv[3:])},
                       options=PushEventOptions(scope=sys.argv[2]))
    print(cmd, "sent for", sys.argv[2])
elif cmd == "score":
    print(score_and_reflect.run(ScoreInput(target_date=sys.argv[2] if len(sys.argv) > 2 else None)))
elif cmd == "poll":
    print(poll_nws.run({}))
