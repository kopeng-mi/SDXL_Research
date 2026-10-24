"""Look and shared widgets.

Darkroom palette: graphite surfaces; amber ("safelight") only for work in
progress; green only for "ready"; red only for failures. One typeface (IBM
Plex Sans) at 13/15/18/24 px. No motion. Field labels sit above fields (13 px)
instead of Quasar's small floating labels.
"""

from __future__ import annotations

import time

from nicegui import ui

from ..jobs import fmt_secs
from .core import OPEN_JOB, jm, job_error

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {--bg:#1C1F24; --panel:#23272E; --raised:#2B3038; --line:#363C46; --text:#E6E8EB; --muted:#9AA3AE;
       --work:#E8A33D; --ok:#5FB98A; --bad:#E07A6B;}
*, *::before, *::after {transition:none !important; animation:none !important;}
body, .q-layout, .q-page-container {background:var(--bg) !important; color:var(--text);
  font-family:'IBM Plex Sans', system-ui, sans-serif; font-size:15px; line-height:1.5;}
.q-field, .q-btn, .q-item, .q-toggle, .q-menu, .q-notification, .q-uploader {font-family:inherit;}
.q-btn {text-transform:none; letter-spacing:0; font-weight:500; white-space:nowrap;}
.q-drawer {background:var(--panel) !important;}
.q-drawer--left {border-right:1px solid var(--line);}
.q-field--outlined .q-field__control {border-radius:8px;}
.q-field--outlined .q-field__control:before {border-color:var(--line);}
.q-field__native, .q-field__input {color:var(--text);}
.q-field__native::placeholder, .q-field__input::placeholder {color:#69717D !important; opacity:1; font-style:italic;}
.q-notification {font-size:14px;}
h1.t-title {font-size:24px; font-weight:500; line-height:1.25; margin:0;}
.t-head {font-size:18px; font-weight:500;}
.t-small {font-size:13px;}
.lbl {font-size:13px; color:var(--muted); margin-bottom:2px;}
.hint {font-size:13px; color:var(--muted);}
.muted {color:var(--muted);} .work {color:var(--work);} .bad {color:var(--bad);} .ok {color:var(--ok);}
.tnum {font-variant-numeric:tabular-nums;}
.panel {background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px;}
.section {border-top:1px solid var(--line); padding-top:12px;}
/* segmented toggle */
.seg {border:1px solid var(--line); border-radius:8px; overflow:hidden; flex-shrink:0;}
.seg .q-btn {color:var(--muted); padding:4px 14px; min-height:34px;}
.seg .q-btn.bg-selected {box-shadow:inset 0 -2px 0 var(--text);}
/* buttons */
.btn-main {height:46px; font-size:16px; font-weight:600; border-radius:10px;}
.btn-main.disabled, .btn-main[disabled] {opacity:1 !important; background:var(--raised) !important; color:var(--work) !important;}
.btn-2 {border:1px solid var(--line); color:var(--text); border-radius:8px;}
.btn-2:hover {border-color:var(--muted);}
.btn-stop {border:1px solid var(--work); color:var(--work); border-radius:8px;}
.btn-bad {border:1px solid var(--bad); color:var(--bad); border-radius:8px;}
/* sidebar */
.nav {width:100%; justify-content:flex-start; color:var(--muted); padding:6px 12px; border-radius:8px;}
.nav.active {color:var(--text); background:var(--raised);}
.bar {display:flex; height:6px; border-radius:3px; overflow:hidden; background:var(--line);}
.bar > span {height:100%; display:block;}
.bar .ghost {background:repeating-linear-gradient(45deg,#6B7480 0 4px,#4A515C 4px 8px);}
.bar .over {background:repeating-linear-gradient(45deg,#C0604F 0 4px,#7A3B31 4px 8px);}
.dot {display:inline-block; width:8px; height:8px; border-radius:4px; margin-right:6px; background:var(--line);}
.dot.ok {background:var(--ok);} .dot.busy {background:var(--work);} .dot.bad {background:var(--bad);}
.progress {height:4px; background:var(--line); border-radius:2px; overflow:hidden;}
.progress > span {display:block; height:100%; background:var(--work);}
/* image area */
.canvas {position:relative; background:var(--panel); border:1px solid var(--line); border-radius:12px; min-height:560px;
  display:flex; align-items:center; justify-content:center; overflow:hidden;}
.canvas img {max-height:78vh;}
.overlay {position:absolute; inset:auto 16px 16px 16px; background:rgba(28,31,36,.92); border:1px solid var(--line);
  border-radius:10px; padding:12px 14px;}
.banner {border:1px solid var(--work); border-radius:10px; padding:10px 12px; background:rgba(232,163,61,.07);}
.thumb {width:72px; height:72px; object-fit:cover; border-radius:8px; cursor:pointer; border:2px solid transparent;}
.thumb.sel {border-color:var(--text);}
.tile {width:168px;}
.tile img {width:168px; height:168px; object-fit:cover; border-radius:8px; cursor:pointer;}
/* drop zone */
.drop .q-uploader__header {background:transparent !important; color:var(--muted);}
.drop {border:1px dashed var(--line) !important; background:transparent !important; box-shadow:none !important; width:100%;}
.drop .q-uploader__list {min-height:0 !important; padding:0 8px;}
.drop .q-uploader__subtitle {display:none;}
.menu-btn {display:none;}
@media (max-width: 900px) {.menu-btn {display:inline-flex; position:fixed; top:8px; left:8px; z-index:3000;}
  .page {padding-top:48px !important;} .two-col {flex-wrap:wrap !important;} .two-col > * {width:100% !important;}}
</style>
"""

SEG_COLORS = ["#7C8DA6", "#8FA89A", "#A8987C", "#9A8FA8", "#6F7B8A", "#A87C86", "#7CA3A8"]


def title(text: str) -> None:
    ui.html(f'<h1 class="t-title">{text}</h1>', sanitize=False)


def labeled(text: str, el, hint: str | None = None):
    """A 13 px label above a field (Quasar's floating labels are too small)."""
    with ui.column().classes("w-full gap-0") as col:
        ui.label(text).classes("lbl")
        el.move(col)
        el.props("outlined dense").classes("w-full")
        if hint:
            ui.label(hint).classes("hint pt-1")
    col.bind_visibility_from(el, "visible")  # hiding the field hides its label and hint too
    return el


def seg(options, S, key, on_change=None):
    t = ui.toggle(options, on_change=on_change).bind_value(S, key)
    return t.props("no-caps unelevated color=transparent text-color=grey-5 toggle-color=selected "
                   "toggle-text-color=white").classes("seg")


def main_button(text: str, on_click):
    return ui.button(text, on_click=on_click).props("unelevated no-caps color=grey-3 text-color=grey-10")         .classes("btn-main w-full")


def button(text: str, on_click=None, icon: str | None = None, kind: str = "2"):
    b = ui.button(text, on_click=on_click, icon=icon).props("flat no-caps dense padding='6px 12px'")
    return b.classes(f"btn-{kind}")


def confirm(question: str, yes: str, on_yes) -> None:
    with ui.dialog() as d, ui.card().classes("panel").style("min-width:360px"):
        ui.label(question)
        with ui.row().classes("w-full justify-end gap-2 pt-2"):
            button("Keep it", d.close)
            button(yes, lambda: (d.close(), on_yes()), kind="bad")
    d.open()


def bar_html(total: float, blocks, preview=None) -> str:
    used = sum(g for _, g in blocks)
    segs = "".join(f'<span style="width:{100 * g / total:.2f}%;background:{SEG_COLORS[i % len(SEG_COLORS)]}" '
                   f'title="{n}: {g:.1f} GB"></span>' for i, (n, g) in enumerate(blocks) if g >= 0.05)
    if preview:
        cls = "over" if preview[1] > total - used else "ghost"
        segs += (f'<span class="{cls}" style="width:{min(100 * preview[1] / total, 100 - 100 * used / total):.2f}%" '
                 f'title="{preview[0]}: about {preview[1]:.1f} GB"></span>')
    return f'<div class="bar">{segs}</div>'


class JobLine:
    """Status of the last job started from this page: progress, failure reason, or done."""

    def __init__(self, kinds: tuple[str, ...]):
        self.kinds, self.job_id = kinds, None
        with ui.column().classes("w-full gap-1") as self.box:
            self.text = ui.label().classes("t-small")
            self.bar = ui.html("", sanitize=False).classes("w-full")
            with ui.row().classes("gap-2"):
                self.cancel = button("Cancel", lambda: jm.cancel(self.job_id), kind="stop")
                self.log = button("Show the log", self.open_log)
        self.box.set_visibility(False)
        ui.timer(1.0, self.tick)

    def open_log(self) -> None:
        OPEN_JOB[ui.context.client.id] = self.job_id
        ui.navigate.to("/activity")

    def track(self, job) -> None:
        self.job_id = job.id
        self.tick()

    def tick(self) -> None:
        if self.job_id is None:  # pick up the latest job of these kinds (e.g. after a reload)
            js = [j for j in jm.recent(30) if j.kind in self.kinds]
            if not js:
                return
            self.job_id = js[0].id
        j = jm.jobs.get(self.job_id)
        if j is None:
            return
        self.box.set_visibility(True)
        running = j.status in ("queued", "running")
        if j.status == "queued":
            msg, cls = f"{j.title}: waiting for the GPU", "work"
        elif j.status == "running":
            prog = f" {j.done}/{j.total}" if j.total else ""
            eta = f", {fmt_secs(j.eta)} left" if j.eta else ""
            msg, cls = f"{j.title}:{prog}{eta}" if prog else f"{j.title}: working", "work"
        elif j.status == "failed":
            msg, cls = f"{j.title} failed: {job_error(j)}", "bad"
        elif j.status == "cancelled":
            msg, cls = f"{j.title}: cancelled", "muted"
        else:
            took = fmt_secs((j.ended or time.time()) - (j.started or j.created))
            msg, cls = f"{j.title}: done in {took}", "ok"
        if self.text.text != msg:
            self.text.text = msg
            self.text.classes(replace=f"t-small {cls}")
        pct = 100 * j.done / j.total if j.total and running else 0
        html = f'<div class="progress"><span style="width:{pct:.0f}%"></span></div>' if running else ""
        if self.bar.content != html:
            self.bar.content = html
        self.cancel.set_visibility(running)
        self.log.set_visibility(j.status in ("failed", "running"))
