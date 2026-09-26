"""Phase-2 agents (docs/ENGINE.md): weather + market subagents, the main View agent, the execution agent.

build_engine_agents(saver, store) is called ONCE from brain.lifespan(); tasks reach the compiled graphs via
ctx.lifespan["engine"]. Subagents return typed objects through `response_format`; deepagents serialises the
structured_response to JSON in the `task` tool result, so the main agent sees both views verbatim.
"""
from __future__ import annotations

import datetime as dt
import os

import httpx
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI


MEMORY_NS = ("bookie", "rules")          # same namespace as phase 1: /memories/AGENTS.md is the live view rulebook
EXEC_MEMORY_KEY = "/EXECUTION.md"

from engine.contracts import ClampedOrder, Gap, MarketView, OrderProposal, SimilarDay, View, WeatherView  # noqa: F401


from engine.tools import EXEC_TOOLS, MAIN_TOOLS, MARKET_TOOLS, WEATHER_TOOLS


# ---- prompts ---------------------------------------------------------------------------
TZ_NOTE = ("Observations are UTC; the local day is America/New_York (EDT, UTC-4). The official daily high is the max over the "
           "local calendar day. Use the exact bucket labels the tools return (e.g. '62° to 63°').")

WEATHER_PROMPT = (
    "You are bookie's weather subagent. Turn the NWS curve, today's observations, recent actuals and similar past days into a "
    "ceiling estimate for the official daily high at Central Park. Use your tools; never invent numbers. " + TZ_NOTE +
    " Return every WeatherView field: running max so far, last obs and its age, trend, ceiling with p10/p90, p_new_high, hours "
    "of risk, regime, how NWS has run vs obs today, similar days (from search_past_reasoning; empty list if none), weather-only "
    "p_by_bucket summing to ~1 over the book's buckets, and the key uncertainty."
)

MARKET_PROMPT = (
    "You are bookie's market subagent. Read the Kalshi book for the target date with your tools and describe where the crowd "
    "is and how sure it is. Never invent numbers. Return every MarketView field: crowd_p (mid/100 per bucket), depth as "
    "(bid size, ask size) per bucket, spread in cents, drift since the last View and over the last hour in cents, thin and "
    "stale buckets, volume and open interest, where money went, and crowd confidence in words."
)

MAIN_PROMPT = (
    "You are bookie, the main View agent for the official daily high at Central Park (Kalshi KXHIGHNY). " + TZ_NOTE +
    " Workflow: (1) delegate to the 'weather' subagent and the 'market' subagent via the task tool, passing the target date; "
    "(2) read your rulebook at /memories/AGENTS.md; (3) form p_by_bucket (must sum to ~1 across the book's buckets) and call "
    "gap_table with it; (4) return a View: list every bucket with a non-trivial gap, mark believed=true only where you trust "
    "the edge and say why, mark the rest believed=false with why you reject them. Numbers come from tools; you add judgment. "
    "You are the only context that sees both views; the execution agent will act on your believed gaps."
)

EXEC_PROMPT = (
    "You are bookie's execution agent. Given a View and ONE believed Gap, propose an OrderProposal for that bucket: side, "
    "limit price in cents, size in contracts, tactic, max slippage, and reasoning. You never see the weather; use get_book, "
    "get_depth, get_exposure and get_my_fills, and follow your execution rulebook at /memories/EXECUTION.md. Code will clamp "
    "you afterwards, so propose what you actually want. If the book is too thin or stale, tactic='skip' with size 0."
)

DEFAULT_EXEC_RULES = """# EXECUTION.md — bookie execution rulebook (v1, seeded)

## Tactic
- edge < 12c: post_and_wait at or inside the spread; do not cross.
- edge >= 12c and depth >= 50 at the price: cross_now.
- otherwise ladder only when depth is spread over several prices; else post_and_wait.

## Size
- never exceed 20% of visible depth at the limit price.
- never exceed 25% of the remaining daily exposure cap.

## Skip
- skip stale buckets (no trades in the window) and buckets the running max already killed.
- skip when hours_left <= 1 or the edge after ~7% x p x (1-p) taker fee is under 2c.

## Known behaviors
(reflect fills this in from graded fills)
"""


def make_model() -> ChatOpenAI:
    return ChatOpenAI(model=os.environ["MODEL_NAME"], base_url=os.environ["MODEL_BASE_URL"],
                      api_key=os.environ["MODEL_API_KEY"], temperature=0, max_tokens=4096,
                      timeout=httpx.Timeout(180.0, connect=10.0), max_retries=1)


async def seed_execution_rules(store) -> None:
    """Seed /memories/EXECUTION.md into the store if missing. Idempotent."""
    if await store.aget(MEMORY_NS, EXEC_MEMORY_KEY) is None:
        ts = dt.datetime.now(dt.timezone.utc).isoformat()
        await store.aput(MEMORY_NS, EXEC_MEMORY_KEY, {"content": DEFAULT_EXEC_RULES, "encoding": "utf-8",
                                                      "created_at": ts, "modified_at": ts, "version": 1})


def _backend(store) -> CompositeBackend:
    return CompositeBackend(default=StateBackend(), routes={"/memories/": StoreBackend(namespace=lambda rt: MEMORY_NS, store=store)})


def build_engine_agents(saver, store) -> dict:
    """Compile every phase-2 agent once. Returns {"weather", "market", "main", "execution"}."""
    model = make_model()
    weather = {"name": "weather", "description": "Weather subagent: forecast curve + observations + similar days -> WeatherView (ceiling, p_new_high, weather-only p_by_bucket).",
               "system_prompt": WEATHER_PROMPT, "tools": WEATHER_TOOLS, "model": model, "response_format": ToolStrategy(WeatherView),
               "middleware": [ModelCallLimitMiddleware(run_limit=12, exit_behavior="end")]}
    market = {"name": "market", "description": "Market subagent: book, depth, drift, trades -> MarketView (crowd_p, depth, spread, thin/stale buckets).",
              "system_prompt": MARKET_PROMPT, "tools": MARKET_TOOLS, "model": model, "response_format": ToolStrategy(MarketView),
              "middleware": [ModelCallLimitMiddleware(run_limit=12, exit_behavior="end")]}
    main = create_deep_agent(
        model=model, tools=MAIN_TOOLS, system_prompt=MAIN_PROMPT, subagents=[weather, market],
        memory=["/memories/AGENTS.md"], backend=_backend(store), store=store, checkpointer=saver,
        response_format=ToolStrategy(View), middleware=[ModelCallLimitMiddleware(run_limit=20, exit_behavior="end")],
    )
    execution = create_deep_agent(
        model=model, tools=EXEC_TOOLS, system_prompt=EXEC_PROMPT,
        memory=["/memories/EXECUTION.md"], backend=_backend(store), store=store, checkpointer=saver,
        response_format=ToolStrategy(OrderProposal), middleware=[ModelCallLimitMiddleware(run_limit=10, exit_behavior="end")],
    )
    return {"weather": weather, "market": market, "main": main, "execution": execution}
