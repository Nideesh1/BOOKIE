"""NiceGUI pages mounted on the FastAPI edge at /ui.

Judge page (`/ui`): explains the loop, shows live state from Atlas, and offers exactly one
control: approve / reject a pending call. Architecture page (`/ui/architecture`) renders
docs/architecture.html. Reads Atlas only to render; writes go through the same bus publish
as POST /verdict/{run_id}.
"""
from __future__ import annotations

import datetime as dt
import difflib
import html
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from nicegui import app as nicegui_app
from nicegui import ui

import db
from bus import publish
from models import VerdictMsg
from streams import CMD_VERDICT

ARCH_HTML = Path(__file__).resolve().parent.parent / "docs" / "architecture.html"
REFRESH_S = 10

STEPS = [
    ("WATCH", "Public weather (NWS forecast + KNYC observations, every 15 min) and the Kalshi "
              "KXHIGHNY book (every 60 s) flow through Redis Streams into Atlas."),
    ("THINK", "Each afternoon a deepagents run reads the data and its own rulebook, then proposes "
              "a 2-degree bucket for tomorrow's Central Park high, with a confidence and a rationale."),
    ("GATE", "Jev, a fast decision model, scores how routine the call is. At 0.8 or above it "
             "auto-approves; below that the run waits until a human clicks Approve or Reject."),
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


@ui.refreshable
async def rules_panel() -> None:
    versions = [d async for d in db.rules().find().sort("version", -1)]
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

def _header(sub: str) -> None:
    with ui.row().classes("items-baseline justify-between w-full"):
        with ui.column().classes("gap-0"):
            ui.label("bookie").classes("text-3xl font-semibold")
            ui.label(sub).classes("opacity-70")
        with ui.row().classes("gap-4"):
            # NiceGUI prefixes string targets with the mount path; page objects resolve correctly.
            ui.link("judge page", judge_page)
            ui.link("architecture", architecture_page)
            ui.html('<a href="/health" target="_blank" class="nicegui-link">/health</a>')


@ui.page("/", title="bookie · judge page")
async def judge_page() -> None:
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
        rules_panel.refresh()
        scores_panel.refresh()
    ui.timer(REFRESH_S, _tick)


@nicegui_app.get("/architecture.html", include_in_schema=False)
async def architecture_raw() -> HTMLResponse:
    return HTMLResponse(ARCH_HTML.read_text(encoding="utf-8") if ARCH_HTML.exists() else "<p>docs/architecture.html not found</p>")


@ui.page("/architecture", title="bookie · architecture")
async def architecture_page() -> None:
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
