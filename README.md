# bookie

Bookie is a self-improving forecasting harness. Every day it predicts tomorrow's official
daily high temperature at Central Park, picks one 2-degree bucket, and grades itself against
the NCEI number once it lands. Kalshi's `KXHIGHNY` market is only the scoreboard: bookie polls
the order book to see what the crowd believes and measures itself against it. **No orders are
ever placed.** What makes it self-improving is the last move of the loop: the agent rewrites
its own rulebook from its graded history, and only code can write a score.

## The 4-move loop

1. **propose** - a deepagents forecaster reads the NWS hourly forecast, recent observations,
   its past scores and similar past days (vector search), then emits a bucket, point estimate,
   confidence and biggest risk.
2. **gate** - Jev auto-approves confident proposals; otherwise the durable run parks until a
   human posts a verdict at `/ui` (or `POST /verdict/{run_id}`).
3. **score** - a nightly cron grades every settled day against the NCEI actual and the
   market's favorite bucket. Scores are written by code only.
4. **reflect** - the agent reads its scorecard and proposes a new version of its rules;
   each accepted edit becomes a new document in the `rules` collection.

## Stack

- `worker/main.py` - FastAPI edge: NWS / NCEI / Kalshi pollers, REST, NiceGUI judge page at `/ui`
- `worker/worker.py` - FastStream consumer over Redis Streams (`bus`), writes to MongoDB
- `worker/brain.py` - Hatchet durable workflows (`market_day`, `score_and_reflect`, `intraday_watch`)
  around a deepagents + LangChain forecaster (OpenAI-compatible model via OpenRouter)
- MongoDB - the only store (forecasts, observations, actuals, snapshots, proposals, scores, rules)
- Voyage AI - embeddings for `memory_search.py` (search over past reasoning)
- Langfuse - LLM tracing via plain OTLP; Hatchet dashboard for runs
- One Docker image, three entrypoints (`worker/Dockerfile`)

## Run it

```sh
scripts/gen-env.sh            # or: cp .env.example .env && cp worker/.env.example worker/.env
$EDITOR worker/.env           # fill MONGODB_URI, MODEL_API_KEY (OpenRouter), VOYAGE_API_KEY
docker compose up -d --build
curl localhost:8000/health    # {"ok":true,"redis":true,"pollers":{...all true}}
docker compose logs -f brain worker api
```

MongoDB is the one thing not in the compose file: point `MONGODB_URI` at Atlas or a local
instance. Seed the first rulebook with `cd worker && uv run python seed.py` (reads `RULES.md`).

## URLs

| What | URL | Login |
| --- | --- | --- |
| API | http://localhost:8000 | - (`/health`, `/pending`, `/rules`, `/scores`) |
| Judge page | http://localhost:8000/ui | - |
| Hatchet | http://localhost:8080 | `admin@example.com` / `Admin123!!` |
| Langfuse | http://localhost:3000 | `you@example.com` / `LANGFUSE_PASSWORD` from `.env` |

All infra ports (postgres, redis, minio, engine gRPC) are bound to `127.0.0.1` only.

Architecture walkthrough: [docs/architecture.html](docs/architecture.html).

## How it learns

- The rulebook lives in the `rules` collection as immutable versions (`version` increases;
  the agent reads the latest). `worker/RULES.md` is v1, written by a human.
- `score_and_reflect` runs nightly: code computes the score for each settled day and inserts
  it into `scores`; the agent never writes a score, it only reads them.
- Reflection then asks the agent for a `RulesEdit` (new rulebook text + a change summary that
  must cite the scores justifying it). The edit is stored as the next `rules` version.
- Every proposal is a durable Hatchet run, so a verdict that arrives hours later resumes the
  same run. `intraday_watch` is code-only and records forecast-vs-market gaps every 5 minutes.

## Hatchet token bootstrap

`hatchet_token` is a one-shot compose service that runs
`hatchet-admin token create --config /hatchet/config` after `hatchet_setup_config` and writes
the token to the `hatchet_token` volume. `worker/entrypoint.sh` exports it as
`HATCHET_CLIENT_TOKEN` when that variable is empty, so `worker/.env` needs no token under
compose. Running on the host? Copy the value out with
`docker compose run --rm --no-deps api cat /hatchet-token/token` into `worker/.env`.
