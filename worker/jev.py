"""Jev (TypeSafe System One) via OpenRouter: fast, typed decisions. Used as the approval gate."""
import os, httpx

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"


async def auto_approve_probability(state: dict) -> float:
    """Probability that this proposal is routine enough to skip the human. Returns 0.0 on any failure."""
    body = {
        "model": MODEL,
        "state": state,
        "questions": {
            "auto_ok": {
                "type": "noul",
                "instructions": (
                    "A forecasting agent proposed a bucket for tomorrow's official daily high temperature. "
                    "Is this call routine enough to approve without a human looking at it?"
                ),
                "criteria": {
                    "true": (
                        "Confidence is at least 0.7, the bucket equals or is adjacent to the market favorite, "
                        "the point estimate is within 2F of the NWS day max, and recent hit rate is not poor."
                    ),
                    "false": (
                        "Low confidence, bucket far from both the market favorite and the NWS max, "
                        "agent has missed most recent calls, or the rationale mentions unusual or uncertain conditions."
                    ),
                },
            }
        },
    }
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(URL, headers={"Authorization": f"Bearer {os.environ['MODEL_API_KEY']}"}, json=body)
            r.raise_for_status()
            return float(r.json()["answers"]["auto_ok"]["noul"])
    except Exception:
        return 0.0
