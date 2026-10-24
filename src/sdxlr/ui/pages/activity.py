"""Activity: everything that ran (loads, images, jobs) and the logs."""

from __future__ import annotations

import time

from nicegui import ui

from ...activity import LOGS, act
from ..core import OPEN_JOB, jm, job_error, logs_zip
from ..theme import button, title
from ...service.jobs import fmt_secs

STATUS = {"running": ("Running", "work"), "queued": ("Waiting", "work"), "done": ("Done", "ok"),
          "failed": ("Failed", "bad"), "cancelled": ("Cancelled", "muted")}


def page(S):
    job = OPEN_JOB.pop(ui.context.client.id, None)
    sel = {"job": job if job in jm.jobs else None}
    with ui.row().classes("w-full items-center"):
        title("Activity")
        ui.space()
        button("Download all logs", lambda: ui.download.file(logs_zip()), icon="download")
    rows = ui.column().classes("w-full gap-0 pt-2")
    with ui.row().classes("w-full items-center pt-4"):
        log_title = ui.label("App log").classes("t-head")
        ui.space()
        back = button("Show the app log", lambda: pick(None, "App log"))
        cancel = button("Cancel this job", lambda: jm.cancel(sel["job"]), kind="stop")
    logv = ui.log(max_lines=500).classes("w-full").style("height:380px; background:var(--panel); font-size:13px; border-radius:10px")
    pos = {"file": None, "off": 0}

    def entries():
        out = []
        for t in act.tasks:
            note = t.stage if t.status == "running" else (t.error or t.detail)
            out.append((t.started, None, t.title or t.kind, t.status, note, f"{t.done}/{t.total}" if t.total else "",
                        fmt_secs(t.elapsed)))
        for j in jm.jobs.values():
            note = job_error(j) if j.status == "failed" else (f"{fmt_secs(j.eta)} left" if j.status == "running" and j.eta else "")
            took = fmt_secs((j.ended or time.time()) - j.started) if j.started else ""
            out.append((j.created, j.id, j.title, j.status, note, f"{j.done}/{j.total}" if j.total else "", took))
        return sorted(out, key=lambda r: r[0], reverse=True)[:40]

    sig = {"v": None}

    def draw():
        es = entries()
        v = [(e[1], e[3], e[4], e[5]) for e in es]
        if v == sig["v"]:
            return
        sig["v"] = v
        rows.clear()
        with rows:
            if not es:
                ui.label("Nothing has run yet.").classes("muted")
            for created, jid, ttl, status, note, prog, took in es:
                word, cls = STATUS.get(status, (status, ""))
                r = ui.row().classes("w-full items-center no-wrap gap-3 py-2").style(
                    "border-bottom:1px solid var(--line);" + ("cursor:pointer;" if jid else "")
                    + ("background:var(--raised);" if jid and jid == sel["job"] else ""))
                with r:
                    ui.label(time.strftime("%H:%M", time.localtime(created))).classes("t-small muted tnum").style("width:44px")
                    ui.label(ttl).classes("grow")
                    ui.label(note or "").classes("t-small " + ("bad" if status == "failed" else "muted")) \
                        .style("max-width:420px; overflow-wrap:anywhere")
                    ui.label(prog).classes("t-small tnum text-right").style("width:70px")
                    ui.label(took).classes("t-small muted tnum text-right").style("width:60px")
                    ui.label(word).classes(f"t-small {cls}").style("width:76px")
                if jid:
                    r.on("click", lambda jid=jid, ttl=ttl: pick(jid, ttl))
                    r.tooltip("Show this job's log")

    def pick(jid, ttl):
        sel["job"] = jid
        log_title.text = ttl
        pos.update(file=None, off=0)
        logv.clear()
        sig["v"] = None
        draw()

    def tail():
        j = jm.jobs.get(sel["job"]) if sel["job"] else None
        f = LOGS / "jobs" / f"{sel['job']}.log" if j else LOGS / "app.log"
        cancel.set_visibility(bool(j) and j.status in ("running", "queued"))
        back.set_visibility(bool(j))
        if pos["file"] != f:
            pos.update(file=f, off=max(f.stat().st_size - 60000, 0) if f.exists() else 0)
        if f.exists() and f.stat().st_size > pos["off"]:
            with open(f, "rb") as fh:
                fh.seek(pos["off"])
                data = fh.read()
            pos["off"] += len(data)
            for line in data.decode("utf-8", "replace").splitlines():
                logv.push(line)

    if sel["job"]:
        log_title.text = jm.jobs[sel["job"]].title
    draw()
    tail()
    ui.timer(2.0, draw)
    ui.timer(1.0, tail)
