"""NiceGUI pages mounted on the FastAPI edge at /ui.

Judge page (`/ui`): explains the loop, shows live state from Atlas, and offers exactly one
control: approve / reject a pending call. Architecture page (`/ui/architecture`) renders
docs/architecture.html. Reads Atlas only to render; writes go through the same bus publish
as POST /verdict/{run_id}.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import difflib
import html
import os
import re
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from nicegui import app as nicegui_app
from nicegui import ui

import db
import exchange
import intraday
import nws
from bus import publish
from engine import execute, risk
from engine.agents import EXEC_MEMORY_KEY, MEMORY_NS
from models import VerdictMsg
from streams import CMD_VERDICT

ET = nws.ET
ARCH_HTML = Path(__file__).resolve().parent.parent / "docs" / "architecture.html"
REFRESH_S = 10

STEPS = [
    ("WATCH", "Public weather (NWS forecast + KNYC observations, every 15 min) and the Kalshi "
              "KXHIGHNY book (every 60 s) flow through Redis Streams into Atlas."),
    ("THINK", "Weather + market subagents form a View: a probability per 2-degree bucket for the "
              "Central Park high, the gaps vs the book it believes, a confidence and a rationale. "
              "The daily proposer still runs each afternoon."),
    ("GATE", "Jev, a fast decision model, answers three typed questions: re-think? act/watch/skip? "
             "safe without a human? Below 0.8 on the last one the run waits for you to Approve or Reject."),
    ("LEARN", "A nightly scorer grades the call against the official NCEI high. The agent then reads "
              "its scores and rewrites its own rulebook; every version is kept."),
]


# ---- helpers -----------------------------------------------------------------------

def _fmt_ts(v) -> str:
    if isinstance(v, dt.datetime):
        return v.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return str(v) if v else "-"


def _decider(p: dict) -> str:
    return "jev" if str(p.get("note", "")).startswith("auto") else "human"


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return "-"


async def _send_verdict(run_id: str, approved: bool, note: str) -> None:
    """Same path as POST /verdict/{run_id}: one message on cmd:verdict."""
    ok = await publish(CMD_VERDICT, VerdictMsg(run_id=run_id, approved=approved, note=note))
    if ok:
        ui.notify(f"{'Approved' if approved else 'Rejected'} run {run_id[:8]}… (queued on the bus)",
                  type="positive" if approved else "warning")
    else:
        ui.notify("Bus unavailable; verdict not queued", type="negative")
    pending_panel.refresh()


def _line_diff(old: str, new: str) -> str:
    """Line diff rendered as escaped HTML inside a monospace block."""
    out: list[str] = []
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=2):
        if line.startswith(("---", "+++")):
            continue
        esc = html.escape(line) or " "
        if line.startswith("+"):
            out.append(f'<span style="color:#1a7f37;background:rgba(46,160,67,.15);display:block">{esc}</span>')
        elif line.startswith("-"):
            out.append(f'<span style="color:#cf222e;background:rgba(248,81,73,.15);display:block">{esc}</span>')
        elif line.startswith("@@"):
            out.append(f'<span style="opacity:.6;display:block">{esc}</span>')
        else:
            out.append(f'<span style="display:block">{esc}</span>')
    if not out:
        out.append("<span>(no line changes)</span>")
    return ('<pre style="font-family:ui-monospace,Menlo,monospace;font-size:12.5px;line-height:1.45;'
            'white-space:pre-wrap;margin:0;padding:12px;border:1px solid rgba(128,128,128,.35);'
            'border-radius:4px;overflow-x:auto">' + "\n".join(out) + "</pre>")


# ---- panels ------------------------------------------------------------------------

@ui.refreshable
async def pending_panel() -> None:
    pending = [d async for d in db.proposals().find({"status": "pending"}).sort("created_at", -1).limit(20)]
    decided = [d async for d in db.proposals().find({"status": {"$in": ["approved", "rejected"]}})
               .sort("decided_at", -1).limit(5)]

    if not pending:
        ui.label("Nothing waiting. Jev auto-approved the recent calls, or no proposal has been made yet today.") \
            .classes("text-sm opacity-70")
    for p in pending:
        with ui.card().classes("w-full"):
            with ui.row().classes("items-baseline gap-4 w-full"):
                ui.label(p.get("target_date", "-")).classes("text-sm opacity-70")
                ui.label(str(p.get("bucket", "-"))).classes("text-2xl font-semibold")
                ui.label(f"point {p.get('point_f', '-')} F").classes("text-sm")
                ui.label(f"confidence {_pct(p.get('confidence'))}").classes("text-sm")
                jev = p.get("jev_auto_ok")
                ui.label(f"jev routine-score {jev:.2f}" if isinstance(jev, (int, float)) else "jev: not scored yet") \
                    .classes("text-sm opacity-70")
            ui.label(f"Biggest risk: {p.get('biggest_risk', '-')}").classes("text-sm")
            with ui.expansion("Rationale").classes("w-full text-sm"):
                ui.markdown(str(p.get("rationale", "")))
            note = ui.input("Note (optional)").classes("w-full").props("dense")
            with ui.row().classes("gap-2"):
                rid = p["run_id"]
                ui.button("Approve", on_click=lambda _, r=rid, n=note: _send_verdict(r, True, n.value or "")) \
                    .props("color=positive unelevated")
                ui.button("Reject", on_click=lambda _, r=rid, n=note: _send_verdict(r, False, n.value or "")) \
                    .props("color=negative outline")

    ui.label("Last 5 decided").classes("text-sm font-medium mt-4")
    if not decided:
        ui.label("No decisions yet.").classes("text-sm opacity-70")
    else:
        rows = [{
            "target_date": d.get("target_date", "-"), "bucket": d.get("bucket", "-"),
            "status": d.get("status", "-"), "by": _decider(d), "note": d.get("note", "") or "",
            "decided_at": _fmt_ts(d.get("decided_at")),
        } for d in decided]
        cols = [{"name": k, "label": k.replace("_", " "), "field": k, "align": "left"} for k in rows[0]]
        ui.table(columns=cols, rows=rows).classes("w-full").props("dense flat bordered")


EDGE_STRONG_CENTS = 8


@ui.refreshable
async def gaps_panel() -> None:
    today = dt.datetime.now(ET).date().isoformat()
    g = await db.gaps().find_one({"target_date": today}, sort=[("as_of", -1)])
    if not g:
        ui.label("No gap estimate yet today. The intraday_watch task runs every 5 min and on every market tick.") \
            .classes("text-sm opacity-70")
        return
    as_of = g.get("as_of")
    as_of_s = as_of.astimezone(ET).strftime("%Y-%m-%d %H:%M ET") if isinstance(as_of, dt.datetime) else str(as_of)
    rm, fm, hl = g.get("running_max"), g.get("remaining_forecast_max"), g.get("hours_left")
    with ui.row().classes("items-baseline gap-6 w-full"):
        ui.label(f"as of {as_of_s}").classes("text-sm opacity-70")
        ui.label(f"running max {rm if rm is not None else '-'} F").classes("text-base font-semibold")
        ui.label(f"forecast max, rest of day {fm if fm is not None else '-'} F").classes("text-base font-semibold")
        ui.label(f"{hl if hl is not None else '-'} h left").classes("text-base font-semibold")
    rows = sorted(g.get("buckets", []), key=lambda b: abs(b.get("edge_cents") or 0), reverse=True)
    rows = [{"label": b.get("label", "-"), "p_model": f"{float(b.get('p_model') or 0):.3f}",
             "mid": f"{float(b['mid']):.3f}" if b.get("mid") is not None else "-",
             "edge": b.get("edge_cents") if b.get("edge_cents") is not None else 0} for b in rows]
    cols = [{"name": "label", "label": "bucket", "field": "label", "align": "left"},
            {"name": "p_model", "label": "model p", "field": "p_model", "align": "right"},
            {"name": "mid", "label": "market mid", "field": "mid", "align": "right"},
            {"name": "edge", "label": "edge ¢", "field": "edge", "align": "right"}]
    t = ui.table(columns=cols, rows=rows).classes("w-full").props("dense flat bordered")
    t.add_slot("body-cell-edge", f"""
        <q-td :props="props" :style="props.value >= {EDGE_STRONG_CENTS} ? 'color:#1a7f37;font-weight:600'
                                     : (props.value <= -{EDGE_STRONG_CENTS} ? 'color:#cf222e;font-weight:600' : '')">
            {{{{ props.value > 0 ? '+' + props.value : props.value }}}}
        </q-td>""")
    ui.label("The watch is code, not the LLM: running max + remaining-day forecast vs the book. "
             "Gaps are where the crowd hasn't caught up.").classes("text-sm opacity-70")


def _et(v) -> str:
    if isinstance(v, dt.datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=dt.timezone.utc)
        return v.astimezone(ET).strftime("%Y-%m-%d %H:%M ET")
    if isinstance(v, str) and v:
        try:
            return _et(dt.datetime.fromisoformat(v.replace("Z", "+00:00")))
        except ValueError:
            return v
    return "-"


def _num(v, nd: int = 2) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def _trunc(v, n: int = 80) -> str:
    t = str(v or "")
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def _bucket_key(label: str) -> float:
    """Sort buckets numerically by the first number in the label (e.g. '72-73', '<70', '80+')."""
    digits = "".join(ch if ch.isdigit() or ch in ".-" else " " for ch in str(label)).split()
    for d in digits:
        try:
            return float(d)
        except ValueError:
            continue
    return -1e9 if "<" in str(label) else 1e9


@ui.refreshable
async def view_panel() -> None:
    v = await db.views().find_one({}, sort=[("as_of", -1)])
    if not v:
        ui.label("No view formed yet. The engine wakes when Jev says the picture changed.") \
            .classes("text-sm opacity-70")
        return
    hl = v.get("hours_left")
    with ui.row().classes("items-baseline gap-6 w-full"):
        ui.label(f"target {v.get('target_date', '-')}").classes("text-sm opacity-70")
        ui.label(f"as of {_et(v.get('as_of'))}").classes("text-sm opacity-70")
        ui.label(f"{_num(hl, 1) if hl is not None else '-'} h left").classes("text-base font-semibold")
        ui.label(f"confidence {_pct(v.get('confidence'))}").classes("text-base font-semibold")

    pbb = v.get("p_by_bucket") or {}
    if pbb:
        ui.label("p by bucket").classes("text-sm font-medium mt-2")
        with ui.column().classes("w-full gap-1"):
            for label in sorted(pbb, key=_bucket_key):
                try:
                    p = max(0.0, min(1.0, float(pbb[label])))
                except (TypeError, ValueError):
                    p = 0.0
                with ui.row().classes("items-center w-full gap-3 no-wrap"):
                    ui.label(str(label)).classes("font-mono text-xs w-16 shrink-0 text-right")
                    with ui.element("div").classes("flex-1").style(
                            "height:10px;background:rgba(128,128,128,.18);border-radius:3px;overflow:hidden"):
                        ui.element("div").style(
                            f"height:100%;width:{p * 100:.1f}%;background:#3b82f6;border-radius:3px")
                    ui.label(_pct(p)).classes("font-mono text-xs w-10 shrink-0")

    gaps = v.get("gaps") or []
    if gaps:
        ui.label("gaps").classes("text-sm font-medium mt-2")
        rows = [{
            "bucket": g.get("bucket", "-"), "p_model": _num(g.get("p_model"), 3),
            "p_market": _num(g.get("p_market"), 3),
            "edge": g.get("edge_c") if g.get("edge_c") is not None else 0,
            "believed": "✓" if g.get("believed") else "✗",
            "why": _trunc(g.get("why")), "why_full": str(g.get("why") or ""),
        } for g in sorted(gaps, key=lambda g: abs(g.get("edge_c") or 0), reverse=True)]
        cols = [{"name": "bucket", "label": "bucket", "field": "bucket", "align": "left"},
                {"name": "p_model", "label": "p model", "field": "p_model", "align": "right"},
                {"name": "p_market", "label": "p market", "field": "p_market", "align": "right"},
                {"name": "edge", "label": "edge ¢", "field": "edge", "align": "right"},
                {"name": "believed", "label": "believed", "field": "believed", "align": "center"},
                {"name": "why", "label": "why", "field": "why", "align": "left"}]
        t = ui.table(columns=cols, rows=rows).classes("w-full").props("dense flat bordered")
        t.add_slot("body-cell-edge", f"""
            <q-td :props="props" :style="props.value >= {EDGE_STRONG_CENTS} ? 'color:#1a7f37;font-weight:600'
                                         : (props.value <= -{EDGE_STRONG_CENTS} ? 'color:#cf222e;font-weight:600' : '')">
                {{{{ props.value > 0 ? '+' + props.value : props.value }}}}
            </q-td>""")
        t.add_slot("body-cell-believed", """
            <q-td :props="props" :style="props.value === '✓' ? 'color:#1a7f37;font-weight:600' : 'opacity:.55'">
                {{ props.value }}
            </q-td>""")
        t.add_slot("body-cell-why", """
            <q-td :props="props" style="max-width:28rem;white-space:normal">
                {{ props.value }}
                <q-tooltip v-if="props.row.why_full.length > props.value.length" max-width="32rem">
                    {{ props.row.why_full }}
                </q-tooltip>
            </q-td>""")
    else:
        ui.label("No gaps in this view.").classes("text-sm opacity-70")

    with ui.expansion("Rationale").classes("w-full text-sm"):
        ui.markdown(str(v.get("rationale") or "(none)"))
    ui.label(f"What would change my mind: {v.get('what_would_change_my_mind') or '-'}") \
        .classes("text-sm").style("white-space:nowrap;overflow:hidden;text-overflow:ellipsis") \
        .tooltip(str(v.get("what_would_change_my_mind") or ""))


CHOICE_COLOR = {"act": "positive", "watch": "warning", "skip": "grey"}
STATUS_COLOR = {"recorded": "primary", "needs_human": "warning", "approved": "positive",
                "rejected": "negative", "watch": "warning", "skip": "grey"}


def _fmt_proposal(p: dict | None) -> str:
    if not p:
        return "-"
    if p.get("tactic") == "skip" and not p.get("size"):
        return "skip"
    return f"{p.get('side', '?')} · {p.get('size', '?')} @ {p.get('limit_price_c', '?')}¢ · {p.get('tactic', '?')}"


def _fmt_clamped(c: dict | None) -> str:
    if not c:
        return "-"
    if c.get("allowed") is False:
        return f"rejected: {c.get('reason') or '-'}"
    s = f"{c.get('size', '?')} @ {c.get('limit_price_c', '?')}¢"
    applied = c.get("clamps_applied") or []
    return f"{s} · {len(applied)} clamp{'s' if len(applied) != 1 else ''}" if applied else s


async def _send_decision_verdict(run_id: str, approved: bool, note: str) -> None:
    await _send_verdict(run_id, approved, note)
    decisions_panel.refresh()


@ui.refreshable
async def decisions_panel() -> None:
    docs = [d async for d in db.decisions().find({}).sort([("created_at", -1), ("_id", -1)]).limit(10)]
    if not docs:
        ui.label("No decisions yet. Each believed gap gets a Jev act/watch/skip, then a proposal and a clamp.") \
            .classes("text-sm opacity-70")
    else:
        rows = []
        for d in docs:
            prop, clamped = d.get("proposal") or {}, d.get("clamped") or {}
            probs = d.get("jev_probs") or {}
            choice = str(d.get("jev_choice") or "-")
            rows.append({
                "run_id": str(d.get("run_id") or ""),
                "time": _et(d.get("created_at") or d.get("ts") or d.get("as_of")),
                "bucket": (d.get("gap") or {}).get("bucket") or prop.get("bucket") or "-",
                "choice": choice, "choice_color": CHOICE_COLOR.get(choice, "grey"),
                "choice_tip": " · ".join(f"{k} {_num(v)}" for k, v in probs.items()) if probs else "",
                "proposal": _fmt_proposal(prop), "proposal_tip": str(prop.get("reasoning") or ""),
                "clamped": _fmt_clamped(clamped), "clamped_tip": "; ".join(clamped.get("clamps_applied") or []),
                "safe": _num(d.get("safe_prob")),
                "status": str(d.get("status") or "-"),
                "status_color": STATUS_COLOR.get(str(d.get("status") or ""), "grey"),
            })
        cols = [{"name": k, "label": lbl, "field": k, "align": al} for k, lbl, al in [
            ("time", "time", "left"), ("bucket", "bucket", "left"), ("choice", "Jev", "left"),
            ("proposal", "proposal", "left"), ("clamped", "clamped", "left"),
            ("safe", "safe", "right"), ("status", "status", "left")]]
        t = ui.table(columns=cols, rows=rows, row_key="run_id").classes("w-full").props("dense flat bordered")
        t.add_slot("body-cell-choice", """
            <q-td :props="props">
                <q-chip dense size="sm" text-color="white" :color="props.row.choice_color">{{ props.value }}</q-chip>
                <q-tooltip v-if="props.row.choice_tip">{{ props.row.choice_tip }}</q-tooltip>
            </q-td>""")
        t.add_slot("body-cell-status", """
            <q-td :props="props">
                <q-chip dense size="sm" text-color="white" :color="props.row.status_color">{{ props.value }}</q-chip>
            </q-td>""")
        t.add_slot("body-cell-proposal", """
            <q-td :props="props">{{ props.value }}
                <q-tooltip v-if="props.row.proposal_tip" max-width="32rem">{{ props.row.proposal_tip }}</q-tooltip>
            </q-td>""")
        t.add_slot("body-cell-clamped", """
            <q-td :props="props">{{ props.value }}
                <q-tooltip v-if="props.row.clamped_tip" max-width="32rem">{{ props.row.clamped_tip }}</q-tooltip>
            </q-td>""")

        waiting = [d for d in docs if d.get("status") == "needs_human" and d.get("run_id")]
        for d in waiting:
            prop, rid = d.get("proposal") or {}, str(d["run_id"])
            with ui.card().classes("w-full mt-2"):
                with ui.row().classes("items-baseline gap-4 w-full"):
                    ui.label("Needs you").classes("text-sm font-medium")
                    ui.label(f"{(d.get('gap') or {}).get('bucket') or prop.get('bucket') or '-'} · "
                             f"{_fmt_proposal(prop)} → {_fmt_clamped(d.get('clamped'))} · safe {_num(d.get('safe_prob'))}") \
                        .classes("text-sm")
                    ui.label(f"run {rid[:8]}…").classes("text-xs opacity-60 font-mono")
                if prop.get("reasoning"):
                    ui.label(str(prop["reasoning"])).classes("text-sm opacity-80")
                note = ui.input("Note (optional)").classes("w-full").props("dense")
                with ui.row().classes("gap-2"):
                    ui.button("Approve", on_click=lambda _, r=rid, n=note: _send_decision_verdict(r, True, n.value or "")) \
                        .props("color=positive unelevated")
                    ui.button("Reject", on_click=lambda _, r=rid, n=note: _send_decision_verdict(r, False, n.value or "")) \
                        .props("color=negative outline")

    ui.label("Recorded / approved rows feed the Orders panel below (phase 3), behind the same clamps plus a fresh re-clamp.") \
        .classes("text-sm opacity-70 mt-2")


# ---- phase 3: orders ------------------------------------------------------------------
ORDER_STATUS_COLOR = {"would_place": "primary", "resting": "warning", "executed": "positive", "canceled": "grey",
                      "would_cancel": "grey", "rejected": "negative", "error": "negative"}
_EXCHANGE_TTL_S = 30.0
_exchange_client: exchange.KalshiClient | None = None
_exchange_err: str | None = None
_acct_cache: dict = {"at": 0.0, "data": None, "err": None}


def _exchange() -> exchange.KalshiClient | None:
    """One client per edge process; dry_run follows ORDERS_ENABLED. None when the env is not configured."""
    global _exchange_client, _exchange_err
    if _exchange_client is None and _exchange_err is None:
        try:
            _exchange_client = exchange.KalshiClient.from_env()
        except (KeyError, OSError, ValueError) as e:
            _exchange_err = f"exchange not configured ({type(e).__name__})"
    return _exchange_client


async def _account() -> tuple[dict | None, str | None]:
    """Balance + positions, cached 30 s. (None, reason) when the exchange is unavailable."""
    if time.monotonic() - _acct_cache["at"] < _EXCHANGE_TTL_S:
        return _acct_cache["data"], _acct_cache["err"]
    client = _exchange()
    data, err = None, _exchange_err
    if client is not None:
        try:
            bal, pos = await asyncio.gather(client.balance(), client.positions())
            data = {"balance": bal, "positions": pos}
        except Exception as e:      # httpx / ExchangeError / timeouts: degrade, never crash the page
            err = f"exchange unavailable ({type(e).__name__})"
    _acct_cache.update({"at": time.monotonic(), "data": data, "err": err})
    return data, err


async def _cancel_all_today() -> None:
    client = _exchange()
    if client is None:
        ui.notify(_exchange_err or "exchange not configured", type="negative")
        return
    today = dt.datetime.now(ET).date().isoformat()
    try:
        r = await execute.cancel_all_for(today, client)
    except Exception as e:
        ui.notify(f"cancel failed: {type(e).__name__}: {e}", type="negative")
        return
    key = "would_cancel" if r.get("dry_run") else "cancelled"
    items = r.get(key) or []
    if r.get("dry_run"):
        msg = (f"DRY RUN (ORDERS_ENABLED is not true): would cancel {len(items)} order(s) on {r['event_ticker']}"
               + (": " + ", ".join(str(i.get("order_id"))[:8] for i in items) if items else "; nothing resting"))
        ui.notify(msg, type="info", multi_line=True, timeout=8000)
    else:
        ui.notify(f"Cancelled {len(items)} order(s) on {r['event_ticker']}", type="warning", multi_line=True, timeout=8000)
    _acct_cache["at"] = 0.0
    orders_panel.refresh()


@ui.refreshable
async def orders_panel() -> None:
    client = _exchange()
    dry = client.dry_run if client is not None else True
    env = client.env if client is not None else "-"
    with ui.row().classes("items-center gap-3 w-full"):
        ui.badge("DRY RUN" if dry else "LIVE", color="primary" if dry else "negative").props("outline" if dry else "")
        ui.label(f"env {env} · ORDERS_ENABLED={'true' if not dry else 'false'} · MAX_CONTRACTS_PER_ORDER="
                 f"{os.environ.get('MAX_CONTRACTS_PER_ORDER', '5')}").classes("text-xs opacity-70")
        ui.space()
        ui.button("Cancel all today", on_click=_cancel_all_today, icon="block").props("color=negative unelevated dense") \
            .tooltip("Cancels every resting bookie order on today's event. In dry run it only shows what it would cancel.")

    # balance / positions readout (30 s cache; degrades to a one-liner)
    acct, err = await _account()
    if acct is None:
        ui.label(err or "exchange unavailable").classes("text-sm opacity-70")
    else:
        b = acct["balance"] or {}
        bal = b.get("balance_dollars") or (f"{_f(b.get('balance')) / 100:.2f}" if b.get("balance") is not None else "-")
        pv = b.get("portfolio_value")
        pos = (acct["positions"] or {}).get("market_positions") or []
        live = [p for p in pos if _f(p.get("position_fp", p.get("position"))) != 0]
        with ui.row().classes("items-baseline gap-6 w-full"):
            ui.label(f"balance ${bal}").classes("text-base font-semibold")
            ui.label(f"portfolio value ${_f(pv) / 100:.2f}" if pv is not None else "portfolio value -").classes("text-sm")
            ui.label(f"{len(live)} open position{'s' if len(live) != 1 else ''}").classes("text-sm opacity-70")
        if live:
            prows = await _position_rows()
            if prows:
                rows = [{"ticker": r["ticker"], "bucket": r["bucket"], "side": r["side"], "contracts": r["contracts"],
                         "entry": f"{_num(r['entry_c'], 0)}¢", "mark": f"{_num(r['mark_c'], 0)}¢" if r["mark_c"] is not None else "-",
                         "unrealized": (f"{r['unrealized_c']:+.0f}¢ ({r['unrealized_pct']:+.0f}%)" if r["unrealized_c"] is not None else "-"),
                         "unrealized_v": r["unrealized_c"] or 0, "source": r["entry_source"]} for r in prows[:10]]
                cols = [{"name": k, "label": lbl, "field": k, "align": al} for k, lbl, al in [
                    ("ticker", "ticker", "left"), ("bucket", "bucket", "left"), ("side", "side", "left"), ("contracts", "contracts", "right"),
                    ("entry", "entry", "right"), ("mark", "mark (mid)", "right"), ("unrealized", "unrealized", "right"), ("source", "entry from", "left")]]
                t = ui.table(columns=cols, rows=rows, row_key="ticker").classes("w-full").props("dense flat bordered")
                t.add_slot("body-cell-unrealized", """
                    <q-td :props="props" :style="props.row.unrealized_v > 0 ? 'color:#1a7f37;font-weight:600'
                                                 : (props.row.unrealized_v < 0 ? 'color:#cf222e;font-weight:600' : '')">{{ props.value }}</q-td>""")
            else:
                rows = [{"ticker": p.get("ticker", "-"), "position": p.get("position_fp", p.get("position", "-")),
                         "exposure": p.get("market_exposure_dollars", p.get("market_exposure", "-")),
                         "pnl": p.get("realized_pnl_dollars", p.get("realized_pnl", "-"))} for p in live[:10]]
                cols = [{"name": k, "label": k, "field": k, "align": "left"} for k in rows[0]]
                ui.table(columns=cols, rows=rows).classes("w-full").props("dense flat bordered")

    docs = [d async for d in db.orders().find({}).sort([("created_at", -1), ("_id", -1)]).limit(10)]
    if not docs:
        ui.label("No orders yet. The place task runs after gate for every recorded / approved decision; in dry run it "
                 "records the payload it would send.").classes("text-sm opacity-70")
        return
    rows = []
    for o in docs:
        st = str(o.get("status") or "-")
        oid = str(o.get("order_id") or "")
        rows.append({
            "key": str(o.get("_id")), "time": _et(o.get("created_at")), "ticker": o.get("ticker", "-"),
            "side": f"{o.get('action', 'buy')} {o.get('side', '?')}",
            "size": f"{o.get('count', '?')} @ {o.get('price_c', '?')}¢", "tactic": o.get("tactic", "-"),
            "status": st, "status_color": ORDER_STATUS_COLOR.get(st, "grey"),
            "filled": f"{_num(o.get('fill_count'), 0)}/{o.get('count', '?')}",
            "order_id": (oid[:8] + "…") if oid else ("(dry run)" if o.get("dry_run") else "-"),
            "payload": str(o.get("payload") or ""), "error": str(o.get("error") or ""),
        })
    cols = [{"name": k, "label": lbl, "field": k, "align": al} for k, lbl, al in [
        ("time", "time", "left"), ("ticker", "ticker", "left"), ("side", "side", "left"), ("size", "count @ price", "left"),
        ("tactic", "tactic", "left"), ("status", "status", "left"), ("filled", "filled", "right"), ("order_id", "order id", "left")]]
    t = ui.table(columns=cols, rows=rows, row_key="key").classes("w-full").props("dense flat bordered")
    t.add_slot("body-cell-status", """
        <q-td :props="props">
            <q-chip dense size="sm" text-color="white" :color="props.row.status_color">{{ props.value }}</q-chip>
            <q-tooltip v-if="props.row.error" max-width="32rem">{{ props.row.error }}</q-tooltip>
        </q-td>""")
    t.add_slot("body-cell-size", """
        <q-td :props="props">{{ props.value }}
            <q-tooltip v-if="props.row.payload" max-width="36rem"><span style="font-family:monospace">{{ props.row.payload }}</span></q-tooltip>
        </q-td>""")
    ui.label("Prices are in the order's own side (no @ 44¢ is sent to Kalshi as ask @ 0.56 on the yes leg). "
             "Hover a row for the exact payload.").classes("text-xs opacity-60 mt-1")


def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---- phase 3b: positions + risk rules --------------------------------------------------
_pos_cache: dict = {"at": 0.0, "rows": None, "positions": None, "err": None}


async def _positions_today() -> tuple[list[risk.Position], str | None]:
    """Live positions as risk.Position (30 s cache, shares the exchange client). ([], reason) when unavailable."""
    if time.monotonic() - _pos_cache["at"] < _EXCHANGE_TTL_S and _pos_cache["positions"] is not None:
        return _pos_cache["positions"], _pos_cache["err"]
    client = _exchange()
    positions, err = [], _exchange_err
    if client is not None:
        try:
            positions = await risk.positions_from_exchange(client)
        except Exception as e:
            err = f"exchange unavailable ({type(e).__name__})"
    _pos_cache.update({"at": time.monotonic(), "positions": positions, "err": err})
    return positions, err


async def _marks_for(target_date: str) -> risk.Marks:
    snap = await db.market_snapshots().find_one({"target_date": target_date}, sort=[("ts", -1)])
    if not snap:
        return risk.Marks()
    try:
        est = await intraday.estimate_today(target_date)
        return risk.marks_from_snapshot(snap, est["running_max"], est["hours_left"])
    except Exception:
        return risk.marks_from_snapshot(snap)


async def _position_rows() -> list[dict]:
    positions, _ = await _positions_today()
    rows, marks = [], {}
    for pos in positions:
        td = pos.target_date or dt.datetime.now(ET).date().isoformat()
        if td not in marks:
            marks[td] = await _marks_for(td)
        pn = risk.position_pnl(pos, marks[td])
        rows.append({"ticker": pos.ticker, "bucket": pos.bucket, "target_date": td, "side": pos.side, "contracts": pos.contracts,
                     "entry_c": pos.entry_c, "mark_c": pn.mark_c, "unrealized_c": pn.unrealized_c, "unrealized_pct": pn.unrealized_pct,
                     "unrealized_usd": pn.unrealized_usd, "entry_source": pos.source})
    return rows


_THRESHOLD_WORDS = re.compile(r"take.?profit|cut|hold|forced|close|reduce|stop", re.I)


async def _exec_rulebook() -> tuple[str, str]:
    """(text, label) of the latest execution rulebook: versioned `rules` doc, else the live store copy."""
    doc = await db.rules().find_one({"kind": "execution"}, sort=[("version", -1)])
    if doc:
        return doc.get("text", ""), f"v{doc['version']} · {doc.get('author', '?')} · {_fmt_ts(doc.get('created_at'))}"
    item = await db.lg_store().find_one({"namespace": list(MEMORY_NS), "key": EXEC_MEMORY_KEY})
    if item:
        return (item.get("value") or {}).get("content", ""), "live store copy (not versioned yet)"
    return "", "no execution rulebook yet"


def _threshold_lines(text: str) -> list[str]:
    """The take-profit / cut / hold lines of the '## Position management' section (fallback: any line with those words)."""
    sec = text.split("## Position management", 1)
    body = sec[1] if len(sec) == 2 else text
    body = body.split("\n## ", 1)[0]
    lines = [l.strip().lstrip("-• ").strip() for l in body.splitlines() if l.strip().startswith(("-", "•", "*"))]
    lines = [l for l in lines if _THRESHOLD_WORDS.search(l)]
    return lines[:8]


@ui.refreshable
async def risk_panel() -> None:
    rules = risk.RiskRules()
    today = dt.datetime.now(ET).date().isoformat()
    positions, err = await _positions_today()
    marks = await _marks_for(today)
    try:
        pnl = await risk.daily_pnl(today, positions, marks)
    except Exception as e:
        pnl = {"realized_usd": 0.0, "fees_usd": 0.0, "unrealized_usd": 0.0, "total_usd": 0.0, "note": f"{type(e).__name__}"}
    forced = risk.evaluate_positions([p for p in positions if p.target_date in (None, today)], marks, rules)
    kill = rules.kill_switch_on()
    with ui.grid(columns=2).classes("w-full gap-4"):
        with ui.column().classes("w-full gap-1"):
            with ui.row().classes("items-center gap-3"):
                ui.label("Hard limits (code)").classes("text-sm font-medium")
                ui.badge("KILL SWITCH ON" if kill else "kill switch OFF", color="negative" if kill else "positive").props("" if kill else "outline")
            rows = [{"key": k, "value": v, "meaning": m} for k, v, m in rules.as_rows()]
            cols = [{"name": "key", "label": "env", "field": "key", "align": "left"},
                    {"name": "value", "label": "live value", "field": "value", "align": "right"},
                    {"name": "meaning", "label": "meaning", "field": "meaning", "align": "left"}]
            ui.table(columns=cols, rows=rows, row_key="key").classes("w-full").props("dense flat bordered")
            capped = rules.daily_loss_cap_hit(pnl["total_usd"])
            color = "#cf222e" if capped or pnl["total_usd"] < 0 else "#1a7f37"
            with ui.row().classes("items-baseline gap-4 mt-1"):
                ui.label("Today's P&L").classes("text-sm font-medium")
                ui.label(f"realized {pnl['realized_usd']:+.2f} · fees {pnl['fees_usd']:.2f} · unrealized {pnl['unrealized_usd']:+.2f}") \
                    .classes("text-sm opacity-80")
                ui.label(f"= ${pnl['total_usd']:+.2f} vs cap -${rules.daily_loss_cap_usd:.2f}").classes("text-sm font-semibold").style(f"color:{color}")
                if capped:
                    ui.badge("no new opens", color="negative")
            if err:
                ui.label(f"positions: {err}").classes("text-xs opacity-60")
            if forced:
                ui.label("Forced now (the next decide run sends these):").classes("text-sm font-medium mt-1")
                for fa in forced:
                    ui.label(fa.summary()).classes("text-xs font-mono").style("color:#cf222e")
            else:
                ui.label(f"No forced action on {len(positions)} open position{'s' if len(positions) != 1 else ''}.").classes("text-xs opacity-60")
        with ui.column().classes("w-full gap-1"):
            text, label = await _exec_rulebook()
            ui.label("Agent thresholds (execution rulebook)").classes("text-sm font-medium")
            ui.label(label).classes("text-xs opacity-60")
            lines = _threshold_lines(text)
            if lines:
                with ui.list().props("dense").classes("w-full"):
                    for l in lines:
                        with ui.item():
                            with ui.item_section():
                                ui.item_label(l).classes("text-sm")
            elif text:
                ui.markdown(text[:1200]).classes("text-sm")
            else:
                ui.label("The execution rulebook is seeded when the brain starts.").classes("text-sm opacity-70")
            with ui.expansion("Full execution rulebook").classes("w-full text-sm"):
                ui.markdown(text or "(none)")
    ui.label("Hard limits fire before the agent sees anything. The agent's thresholds are what reflect tunes.") \
        .classes("text-sm opacity-70 mt-1")


@ui.refreshable
async def rules_panel() -> None:
    versions = [d async for d in db.rules().find({"kind": {"$ne": "execution"}}).sort("version", -1)]
    if not versions:
        ui.label("No rulebook yet (seed.py inserts v1).").classes("text-sm opacity-70")
        return
    by_version = {v["version"]: v for v in versions}
    detail = ui.column().classes("w-full")

    def show(v: dict) -> None:
        detail.clear()
        with detail:
            ui.separator()
            ui.label(f"v{v['version']} · {v.get('author', '?')} · {_fmt_ts(v.get('created_at'))}") \
                .classes("text-sm font-medium")
            prev = by_version.get(v["version"] - 1)
            if v.get("author") == "agent" and prev is not None:
                with ui.expansion(f"Line diff vs v{prev['version']}", value=True).classes("w-full text-sm"):
                    ui.html(_line_diff(prev.get("text", ""), v.get("text", "")))
            with ui.expansion("Full rulebook", value=v.get("author") != "agent").classes("w-full text-sm"):
                ui.markdown(v.get("text", ""))

    with ui.list().props("dense separator").classes("w-full"):
        for v in versions:
            with ui.item(on_click=lambda _, vv=v: show(vv)):
                with ui.item_section().props("side"):
                    ui.label(f"v{v['version']}").classes("font-mono text-sm")
                with ui.item_section():
                    ui.item_label(f"{v.get('author', '?')} · {_fmt_ts(v.get('created_at'))}").classes("text-sm")
                    ui.item_label(v.get("change_summary") or "initial rulebook written by a human") \
                        .props("caption")
    show(versions[0])


@ui.refreshable
async def scores_panel() -> None:
    scores = [d async for d in db.scores().find().sort("target_date", -1).limit(100)]
    if not scores:
        ui.label("No graded calls yet. The nightly scorer writes one row per target date.") \
            .classes("text-sm opacity-70")
        return
    hits = sum(1 for s in scores if s.get("hit"))
    errs = [abs(float(s["error_f"])) for s in scores if s.get("error_f") is not None]
    mae = sum(errs) / len(errs) if errs else 0.0
    ui.label(f"Hit rate {hits}/{len(scores)} ({hits / len(scores) * 100:.0f}%) · mean abs error {mae:.1f} F") \
        .classes("text-sm font-medium")
    rows = [{
        "target_date": s.get("target_date", "-"), "bucket": s.get("bucket", "-"),
        "point_f": s.get("point_f", "-"), "actual_f": s.get("actual_f", "-"),
        "error_f": s.get("error_f", "-"), "hit": "✓" if s.get("hit") else "✗",
        "provisional": "yes" if s.get("provisional") else "", "rules_version": s.get("rules_version", "-"),
    } for s in scores]
    cols = [{"name": k, "label": k.replace("_", " "), "field": k, "align": "left"} for k in rows[0]]
    ui.table(columns=cols, rows=rows).classes("w-full").props("dense flat bordered")


# ---- pages -------------------------------------------------------------------------

def _nav_bar(active: str) -> None:
    """Fixed, elevated top bar shared by every page. `active` is 'judge' or 'architecture'."""
    with ui.header(elevated=True).classes("items-center justify-between px-4 py-2 gap-4"):
        ui.link("bookie", judge_page).classes("text-lg font-bold text-white no-underline")
        with ui.row().classes("items-center gap-5"):
            def _link(text: str, target, key: str = "", new_tab: bool = False) -> None:
                # NiceGUI prefixes string targets with the mount path; page objects resolve correctly.
                if isinstance(target, str):
                    cls = "nicegui-link text-white " + ("font-semibold underline" if key == active else "opacity-80 no-underline")
                    ui.html(f'<a href="{target}" target="{"_blank" if new_tab else "_self"}" '
                            f'class="{cls}">{html.escape(text)}</a>')
                    return
                link = ui.link(text, target, new_tab=new_tab).classes("text-white")
                link.classes("font-semibold underline" if key == active else "opacity-80 no-underline")
            _link("Judge page", judge_page, "judge")
            _link("Architecture", architecture_page, "architecture")
            _link("Hatchet", "http://localhost:8080", new_tab=True)
            _link("Langfuse", "http://localhost:3000", new_tab=True)
            _link("Health", "/health", new_tab=True)


def _header(sub: str) -> None:
    with ui.column().classes("gap-0"):
        ui.label("bookie").classes("text-3xl font-semibold")
        ui.label(sub).classes("opacity-70")


@ui.page("/", title="bookie · judge page", response_timeout=20)   # first render waits on the exchange (30 s cache after)
async def judge_page() -> None:
    _nav_bar("judge")
    with ui.column().classes("max-w-6xl mx-auto w-full p-4 gap-6"):
        _header("A self-improving agent that calls tomorrow's NYC Central Park daily high, "
                "is gated by a fast model, is graded nightly, and rewrites its own rulebook.")

        with ui.grid(columns=4).classes("w-full gap-3"):
            for i, (name, desc) in enumerate(STEPS):
                with ui.card().classes("h-full"):
                    ui.label(f"{i + 1} · {name}").classes("text-xl font-semibold tracking-wide")
                    ui.label(desc).classes("text-sm opacity-80")
        ui.label("Why this matters: the reward is a hard external number (the official high), the memory "
                 "is a rulebook that persists and is versioned across days, and a human is in the loop only "
                 "when the gate says the call is not routine.").classes("text-sm")

        with ui.card().classes("w-full"):
            ui.label("Calls awaiting you").classes("text-lg font-semibold")
            ui.label("The one control on this page. Approve or Reject publishes to cmd:verdict, the same "
                     "path as POST /verdict/{run_id}; Hatchet resumes the waiting run.").classes("text-sm opacity-70")
            await pending_panel()

        with ui.card().classes("w-full"):
            ui.label("Gaps right now (today)").classes("text-lg font-semibold")
            ui.label("Model probability per bucket for today's high vs the live Kalshi mid. Sorted by |edge|; "
                     f"green at +{EDGE_STRONG_CENTS}¢ or more, red at -{EDGE_STRONG_CENTS}¢ or less. No orders are placed.") \
                .classes("text-sm opacity-70")
            await gaps_panel()

        with ui.card().classes("w-full"):
            ui.label("Latest view").classes("text-lg font-semibold")
            ui.label("Formed by the market_view workflow: weather and market subagents feed the main agent, which "
                     "returns a typed View. Numbers come from tools; the model adds the narrative.") \
                .classes("text-sm opacity-70")
            await view_panel()

        with ui.card().classes("w-full"):
            ui.label("Decisions").classes("text-lg font-semibold")
            ui.label("Per believed gap: Jev picks act / watch / skip, the execution agent proposes an order, code "
                     "clamps it, Jev scores whether it is safe without you. Last 10, newest first.") \
                .classes("text-sm opacity-70")
            await decisions_panel()

        with ui.card().classes("w-full"):
            ui.label("Orders").classes("text-lg font-semibold")
            ui.label("Phase 3: recorded / approved decisions are re-clamped against a fresh tick and sent as limit orders "
                     "(post-only for post_and_wait). With ORDERS_ENABLED=false nothing is sent; the exact payload is kept as "
                     "would_place. Last 10, newest first.").classes("text-sm opacity-70")
            await orders_panel()

        with ui.card().classes("w-full"):
            ui.label("Risk rules").classes("text-lg font-semibold")
            ui.label("Two layers. Left: engine/risk.py, read from env, deterministic, applied to live positions before any agent runs "
                     "(forced closes skip the human gate but not the clamp or the kill switch). Right: the thresholds the execution "
                     "agent follows from /memories/EXECUTION.md, which the nightly reflect rewrites from graded trades.") \
                .classes("text-sm opacity-70")
            await risk_panel()

        with ui.grid(columns=2).classes("w-full gap-4"):
            with ui.card().classes("w-full"):
                ui.label("Rulebook versions").classes("text-lg font-semibold")
                ui.label("v1 was written by a human. Later versions are written by the agent after it "
                         "reads its own scores.").classes("text-sm opacity-70")
                await rules_panel()
            with ui.card().classes("w-full"):
                ui.label("Scorecard").classes("text-lg font-semibold")
                ui.label("Graded by code, never by the agent. Provisional means NCEI has not published "
                         "yet and the observation max was used.").classes("text-sm opacity-70")
                await scores_panel()

        ui.label(f"Panels refresh from Atlas every {REFRESH_S} s.").classes("text-xs opacity-60")

    async def _tick() -> None:
        pending_panel.refresh()
        gaps_panel.refresh()
        view_panel.refresh()
        decisions_panel.refresh()
        orders_panel.refresh()
        risk_panel.refresh()
        rules_panel.refresh()
        scores_panel.refresh()
    ui.timer(REFRESH_S, _tick)


@nicegui_app.get("/architecture.html", include_in_schema=False)
async def architecture_raw() -> HTMLResponse:
    return HTMLResponse(ARCH_HTML.read_text(encoding="utf-8") if ARCH_HTML.exists() else "<p>docs/architecture.html not found</p>")


@ui.page("/architecture", title="bookie · architecture")
async def architecture_page() -> None:
    _nav_bar("architecture")
    with ui.column().classes("max-w-6xl mx-auto w-full p-4 gap-4"):
        _header("Architecture")
        ui.label("For judges: this is the full data path behind the judge page. Public sources enter through "
                 "a thin FastAPI edge, Redis Streams is the only coupling, a FastStream worker writes to Atlas "
                 "and starts Hatchet runs, and inside Hatchet the deepagents proposer, the Jev gate, and the "
                 "nightly score-and-reflect job run as durable workflows. Atlas is the single source of truth; "
                 "the amber path is the only place a human is involved.").classes("text-sm")
        ui.element("iframe").props('src="/ui/architecture.html"').classes("w-full") \
            .style("border:0; min-height: 1800px") \
            .on("load", js_handler="(e) => { const f = e.target; "
                                   "f.style.height = (f.contentDocument.documentElement.scrollHeight + 40) + 'px'; }")


def mount(app: FastAPI) -> None:
    """Mount NiceGUI at /ui. Wraps (does not replace) the edge lifespan."""
    ui.run_with(app, mount_path="/ui", title="bookie",
                storage_secret=os.environ.get("UI_STORAGE_SECRET", "bookie-dev-secret"),
                show_welcome_message=False)
