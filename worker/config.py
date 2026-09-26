"""Typed settings loaded from worker/.env. One cached object per process."""
from __future__ import annotations

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- store / bus ---
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "bookie"
    redis_url: str = "redis://localhost:6380/0"

    # --- upstream data ---
    nws_user_agent: str = "bookie (contact@example.com)"

    # --- poller cadence (seconds) ---
    forecast_interval: int = 15 * 60
    obs_interval: int = 15 * 60
    actual_interval: int = 24 * 60 * 60
    tick_interval: int = 60


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    # db.py / nws.py read os.environ directly; keep them in sync with the .env values.
    os.environ.setdefault("MONGODB_URI", s.mongodb_uri)
    os.environ.setdefault("MONGODB_DB", s.mongodb_db)
    os.environ.setdefault("NWS_USER_AGENT", s.nws_user_agent)
    return s


settings = get_settings()
