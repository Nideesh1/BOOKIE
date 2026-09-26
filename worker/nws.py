"""Public NWS + NCEI data for Central Park (KNYC). No auth. User-Agent required by NWS."""
import os, datetime as dt
import httpx

GRID = "https://api.weather.gov/gridpoints/OKX/34,45/forecast/hourly"
OBS  = "https://api.weather.gov/stations/KNYC/observations"
NCEI = ("https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries"
        "&stations=USW00094728&dataTypes=TMAX&format=json&units=standard")
ET = dt.timezone(dt.timedelta(hours=-4))

def _h(): return {"User-Agent": os.environ.get("NWS_USER_AGENT", "bookie (contact@example.com)")}

async def hourly_forecast(client: httpx.AsyncClient) -> list[dict]:
    r = await client.get(GRID, headers=_h(), timeout=20); r.raise_for_status()
    return [{"t": p["startTime"], "temp_f": p["temperature"], "pop": (p.get("probabilityOfPrecipitation") or {}).get("value"),
             "short": p["shortForecast"]} for p in r.json()["properties"]["periods"]]

async def observations(client: httpx.AsyncClient, limit=24) -> list[dict]:
    r = await client.get(OBS, params={"limit": limit}, headers=_h(), timeout=20); r.raise_for_status()
    out = []
    for f in r.json()["features"]:
        p = f["properties"]; c = p["temperature"]["value"]
        if c is None: continue
        out.append({"ts": p["timestamp"], "temp_f": round(c * 9 / 5 + 32, 1)})
    return out

async def ncei_tmax(client: httpx.AsyncClient, start: str, end: str) -> list[dict]:
    r = await client.get(f"{NCEI}&startDate={start}&endDate={end}", timeout=30); r.raise_for_status()
    return [{"date": x["DATE"], "tmax_f": int(x["TMAX"])} for x in r.json() if x.get("TMAX")]

def day_max(periods: list[dict], target_date: str) -> int | None:
    temps = [p["temp_f"] for p in periods if p["t"].startswith(target_date)]
    return max(temps) if temps else None

def bucket(temp: float) -> str:
    """Prediction-market style 2-degree buckets, e.g. 62-63. Tails are open."""
    t = round(temp)
    lo = t - (t % 2)
    return f"{lo}-{lo+1}"
