# The engine: how bookie forms a view and acts on it

This is the design for phase 2. Phase 1 (merged) proved the harness: data flows in, an
agent proposes once a day, Jev gates, code grades, the agent rewrites its rulebook.
Phase 2 makes the agent think **continuously** as the book and the weather move, and
adds an execution layer so a view can become an order. Phase 3 turns on real orders.

## One sentence

**Code** keeps state and enforces limits. **Jev** makes cheap, typed decisions about
*whether* to think or act. **deepagents** forms views and decides *how* to trade them.
The agent never sees a path around the limits.

## The loop

```
tick (15 s)
  └─ CODE tick_state()            book shape, running max, obs trend, curve, deltas
       └─ JEV noul "re-think?"    p ≥ 0.5, or a bucket boundary was crossed
            └─ DEEPAGENTS market_view
                 ├─ weather subagent  → WeatherView
                 ├─ market subagent   → MarketView
                 └─ main agent        → View (p per bucket, gaps believed / rejected)
                      └─ JEV choice per gap: act / watch / skip
                           └─ DEEPAGENTS execution agent → OrderProposal (side, price, size, tactic)
                                └─ CODE clamp             caps, depth, spread, kill switch
                                     └─ JEV noul "safe without a human?"
                                          ├─ yes → CODE place (phase 3) / record (phase 2)
                                          └─ no  → Hatchet wait_for_event ← you

nightly
  └─ CODE score every View and Order against the official high
       └─ JEV noul "enough signal to change rules?"
            └─ DEEPAGENTS reflect → rulebook vN+1  (already built)
```

Day-ahead is the same loop with `hours_left = 24` and no running max. There is no
separate code path for "tomorrow".

## Roles

| Role | Owner | Why |
|---|---|---|
| State, gap math, scoring, persistence | code | deterministic, testable, never hallucinates a number |
| Re-think trigger | Jev | 0.2 s, $0.04/M tokens; runs every tick without spending LLM money |
| Act / watch / skip per gap | Jev | typed choice with probabilities; auditable |
| Human gate | Jev | one number, one threshold, one span in Langfuse |
| Reflect trigger | Jev | stops the nightly job from rewriting rules on noise |
| Weather view | subagent | turns curve + obs + similar days into a ceiling estimate |
| Market view | subagent | turns book + depth + drift into "where the crowd is and how sure" |
| View | main agent | only context that sees both views plus the rulebook |
| Order proposal | execution agent | tactic is judgment: post vs cross, size vs depth, skip thin books |
| Caps, kill switch, placement | code | the model cannot talk its way past a cap |
| Rulebook | reflect agent | already built; grows two rulebooks now (view, execution) |

## Typed contracts

Subagents and the main agent return Pydantic objects via `ToolStrategy`. Numbers in
these objects come from tools, not from the model's head; the model adds the narrative
fields. The main agent receives both views verbatim (~400 tokens each), not summaries.

```python
class SimilarDay(BaseModel):
    date: str; called: str; actual_f: int; hit: bool; lesson: str

class WeatherView(BaseModel):
    running_max_f: float
    last_obs_f: float; last_obs_age_min: int; trend_f_per_hr: float
    ceiling_f: float; ceiling_p10_f: float; ceiling_p90_f: float
    p_new_high: float                    # chance the running max gets beaten at all
    hours_of_risk: list[str]             # ["14:00-16:00"] windows where a new high is plausible
    regime: str                          # post-frontal, clear-sky heating, convective, marine layer, ...
    forecast_bias_note: str              # what NWS has done vs obs today
    similar_days: list[SimilarDay]
    p_by_bucket: dict[str, float]        # weather-only probabilities
    key_uncertainty: str

class MarketView(BaseModel):
    crowd_p: dict[str, float]            # implied p per bucket from mids
    depth: dict[str, tuple[int, int]]    # (bid size, ask size) per bucket
    spread_c: dict[str, int]
    drift_since_last_view_c: dict[str, int]
    drift_last_hour_c: dict[str, int]
    thin_buckets: list[str]              # a small order moves price
    stale_buckets: list[str]             # no trades in N min
    volume_today: float; open_interest: float
    where_money_went: str
    crowd_confidence: str

class Gap(BaseModel):
    bucket: str; p_model: float; p_market: float; edge_c: int
    believed: bool; why: str

class View(BaseModel):
    target_date: str; as_of: str; hours_left: float
    p_by_bucket: dict[str, float]
    gaps: list[Gap]
    confidence: float
    what_would_change_my_mind: str
    rationale: str

class OrderProposal(BaseModel):
    bucket: str; side: Literal["yes", "no"]
    limit_price_c: int; size: int
    tactic: Literal["post_and_wait", "cross_now", "ladder", "skip"]
    max_slippage_c: int
    reasoning: str

class ClampedOrder(BaseModel):            # what code actually sends
    proposal: OrderProposal
    size: int; limit_price_c: int          # after caps
    clamps_applied: list[str]              # "size cut to 20% of depth", "capped at daily exposure"
    allowed: bool; reason: str
```

## Jev questions (exact wording matters; these are the prompts)

**re-think** (noul, every tick) — state: tick deltas.
"Has the situation changed enough since the last view that the agent should re-think?"
true when: running max crossed a bucket boundary; any bucket mid moved ≥ 4¢ or depth
flipped sides; observations drifted ≥ 2°F from the curve; a new NWS run landed.
false when: the book and the readings are within noise of the last view.
Cooldown: 2 min between views, except a boundary cross always wakes the agent.

**act / watch / skip** (choice, per believed gap) — state: the Gap, depth, spread, hours left.
act: edge ≥ 8¢ and depth ≥ 20 contracts at that price and hours_left > 1.
watch: edge 4–8¢, or edge is large but the book is thin or stale.
skip: edge < 4¢, or the gap is on a bucket the running max already killed.

**safe without a human** (noul) — state: ClampedOrder, View.confidence, recent hit rate,
size vs daily exposure used.
true when: confidence ≥ 0.7, size ≤ 25% of remaining daily cap, edge ≥ 8¢, tactic is
post_and_wait or cross_now, hit rate over the last 10 not below 40%.
false when: anything else. Threshold 0.8.

**enough signal to reflect** (noul, nightly) — state: today's scored views and orders.
true when: ≥ 3 new graded outcomes since the last rulebook version, or one miss ≥ 5°F,
or a fill got run over by ≥ 10¢.

## Tools per agent

| Agent | Tools (all read-only over Atlas) |
|---|---|
| weather subagent | get_forecast, get_observations, get_recent_actuals, search_past_reasoning |
| market subagent | get_book, get_book_history, get_depth, get_trades_recent |
| main agent | gap_table (code), get_last_view, rulebook via /memories/AGENTS.md |
| execution agent | get_book, get_depth, get_exposure, get_my_fills, rulebook via /memories/EXECUTION.md |
| reflect agent | get_my_scores, get_my_fills, edit_file on both rulebooks |

Subagents get only their domain's tools. The main agent is the only context that sees
both views. The execution agent never sees the weather; it sees the View and the book.

## Code clamps (phase 3, but designed now)

Applied in order, each can only shrink or reject:
1. kill switch file/flag → reject everything
2. per-day exposure cap (dollars at risk)
3. per-bucket cap
4. size ≤ 20% of visible depth at the limit price
5. limit price inside the spread, never through it by more than max_slippage_c
6. no orders in the last 30 min before settlement lock
7. fees: Kalshi taker ≈ 7% × p × (1−p) per contract; reject if edge after fees < 2¢

Every clamp writes a line into `clamps_applied`, so reflect can learn "I keep getting
cut to 20% of depth on tails, propose smaller".

## Persistence

| Collection | Written by |
|---|---|
| ticks | code, every 15 s (book shape + obs) |
| views | main agent, per re-think |
| decisions | Jev act/watch/skip + execution proposal + clamp result |
| orders | code (phase 3): what was sent, fills, cancels |
| scores | code, nightly: views and orders graded |
| rules | reflect agent: view rulebook, execution rulebook, versioned |
| reasoning | embedded views, for search_past_reasoning |

## Hatchet shape

- `market_view` workflow: `tick_and_gate` (Jev) → `form_view` (subagents + main) → `decide` (Jev + execution + clamp) → `gate` (durable wait if unsafe) → `record`.
- Started by the FastStream worker on `mkt:tick`; the 15 s cadence is the poller's.
- All agents compiled once in the Hatchet lifespan and reached via `ctx.lifespan`.
- `score_and_reflect` nightly, unchanged, plus the Jev reflect trigger.

## Phases

- **Phase 2 (next):** views + subagents + execution agent + clamps + Jev questions. Orders are
  recorded, not sent. Judge page shows the View, the believed gaps, and what would have
  been sent.
- **Phase 3:** exchange client (REST, then WebSocket for the book), kill switch, paper
  account first, then real with the caps above. Nothing else changes.
