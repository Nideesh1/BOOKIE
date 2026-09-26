"""Message schemas carried over the bus. Edge builds them, brain validates them."""
from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field


class ForecastMsg(BaseModel):
    fetched_at: dt.datetime
    target_date: str                     # YYYY-MM-DD (ET)
    day_max: int | None = None
    periods: list[dict[str, Any]]


class ObsMsg(BaseModel):
    ts: dt.datetime
    temp_f: float


class ActualMsg(BaseModel):
    date: str                            # YYYY-MM-DD
    tmax_f: int


class TickMsg(BaseModel):
    ts: dt.datetime
    event_ticker: str
    target_date: str
    markets: list[dict[str, Any]]
    favorite_bucket: str | None = None
    favorite_mid: float | None = None


class VerdictMsg(BaseModel):
    run_id: str
    approved: bool
    note: str = ""


class VerdictBody(BaseModel):
    approved: bool
    note: str = Field(default="")
