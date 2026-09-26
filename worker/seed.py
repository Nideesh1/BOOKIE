"""Seed Atlas with public history: NCEI official daily highs + current NWS forecast/obs."""
import asyncio, datetime as dt, httpx
from dotenv import load_dotenv; load_dotenv()
import db, nws

async def main():
    await db.ensure_indexes()
    async with httpx.AsyncClient() as c:
        end = dt.date.today(); start = end - dt.timedelta(days=45)
        rows = await nws.ncei_tmax(c, start.isoformat(), end.isoformat())
        for r in rows:
            await db.actuals().update_one({"date": r["date"]}, {"$set": r}, upsert=True)
        print("actuals:", len(rows), rows[-1] if rows else None)
        fc = await nws.hourly_forecast(c); now = dt.datetime.now(dt.timezone.utc)
        tomorrow = (dt.datetime.now(nws.ET).date() + dt.timedelta(days=1)).isoformat()
        await db.forecasts().insert_one({"fetched_at": now, "target_date": tomorrow, "source": "nws",
                                         "periods": fc, "day_max": nws.day_max(fc, tomorrow)})
        obs = await nws.observations(c, 48)
        for o in obs:
            await db.observations().update_one({"ts": o["ts"]}, {"$set": o}, upsert=True)
        print("forecast day_max", tomorrow, nws.day_max(fc, tomorrow), "| obs:", len(obs), obs[0])
    await db.rules().update_one({"version": 1}, {"$setOnInsert": {"version": 1, "created_at": dt.datetime.now(dt.timezone.utc),
        "author": "human", "text": open("RULES.md").read()}}, upsert=True)
    print("rules v1 seeded")

asyncio.run(main())
