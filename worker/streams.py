"""Redis Streams contract. Imported by BOTH the edge (main.py) and the brain (worker.py)."""
import os
import socket

# Streams: <domain>:<event>
WX_FORECAST = "wx:forecast"   # NWS hourly forecast snapshot
WX_OBS = "wx:obs"             # KNYC hourly observation
WX_ACTUAL = "wx:actual"       # NCEI official daily TMAX
MKT_TICK = "mkt:tick"         # Kalshi KXHIGHNY book snapshot
CMD_VERDICT = "cmd:verdict"   # human approve/reject of a proposal
OUT_PROPOSAL = "out:proposal" # brain -> edge: a new proposal

ALL = [WX_FORECAST, WX_OBS, WX_ACTUAL, MKT_TICK, CMD_VERDICT, OUT_PROPOSAL]

# Consumer group shared by every brain replica
GROUP = "brain"

# Approximate trimming applied on every publish so the streams self-cap
MAXLEN = 10_000


def consumer_name() -> str:
    """Unique per replica, so scaled processes don't collide on group claims."""
    return f"{socket.gethostname()}-{os.getpid()}"
