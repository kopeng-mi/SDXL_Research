"""The frame: sidebar (pages + live status) around one page, plus app startup.

The event loop must stay free (it answers the browser's keep-alive pings):
memory/disk numbers come from core.sample_usage_loop, and timers only push
changes to the page.
"""

from __future__ import annotations

import argparse
import secrets
import threading
from pathlib import Path

from nicegui import app, ui

from .. import presets, session, settings
from ..activity import log, setup_logging
from ..paths import LORAS, MODELS, THUMBS
from ..service.jobs import fmt_secs
from ..service import state
from . import core
from .core import (USAGE, busy_task, engc, estimate_preview, jm, loaded_label, running_job, state_for,
                   task_is_loading)
from .pages import activity, generate, history, library
from .theme import HEAD, bar_html, button, confirm

# dataset, training and compare live in the notebooks now (notebooks/*.ipynb)
PAGES = [("Generate", "/", generate.page, "auto_awesome"), ("Library", "/library", library.page, "inventory_2"),
         ("History", "/history", history.page, "history"), ("Activity", "/activity", activity.page, "list_alt")]


def set_if(el, attr: str, value) -> None:
    if getattr(el, attr) != value:
        setattr(el, attr, value)


def root():
    ui.add_head_html(HEAD)
    ui.dark_mode(True)
    ui.colors(primary="#9AA3AE", selected="#424A57", positive="#3F7F5F", negative="#9A4A3F", warning="#8A6A2E")
    code = app.storage.browser.get("code")
    if not session.valid(code) or session.load(code) is None:
        # a new browser (or a new VM) has no cookie: one user, so continue the latest session
        code = session.latest() or session.new_code()
        app.storage.browser["code"] = code
    S = state_for(code)

    navs = {}

    async def go(path):
        ui.navigate.to(path)
        if await ui.run_javascript("window.innerWidth", timeout=3) < 900:
            drawer.hide()  # on a narrow screen the menu covers the page

    drawer = ui.left_drawer(value=True, fixed=True).props("width=252 breakpoint=900 show-if-above").classes("p-4")
    ui.button(icon="menu", on_click=drawer.toggle).props("flat round").classes("menu-btn")
    with drawer:
        with ui.column().classes("w-full h-full gap-1 no-wrap"):
            ui.label("SDXL Research").classes("t-head pb-3")
            for name, path, _, icon in PAGES:
                navs[path] = ui.button(name, icon=icon, on_click=lambda path=path: go(path)) \
                    .props("flat no-caps align=left").classes("nav")
            ui.space()
            with ui.column().classes("w-full gap-2 section"):
                model_l = ui.html("", sanitize=False).classes("t-small")
                mem = ui.html("", sanitize=False).classes("w-full")
                link_l = ui.label().classes("t-small muted").tooltip(
                    "How your device reaches this VM. Relayed connections add lag; direct is best")
            with ui.column().classes("w-full gap-1 section"):
                task_l = ui.label("Nothing running").classes("t-small muted")
                task_d = ui.label().classes("t-small muted tnum")
                task_p = ui.html("", sanitize=False).classes("w-full")

                def cancel_current():
                    t = busy_task()
                    if t:
                        t.cancel()  # a load or an image: cheap to redo, no question
                    elif (j := running_job()) is not None:
                        confirm(f"Stop \"{j.title}\"? Its progress so far is lost.", "Stop it", lambda: jm.cancel(j.id))

                cancel_b = button("Cancel", cancel_current, kind="stop").props("size=sm")
            with ui.row().classes("w-full items-center section"):
                ui.label(f"Session {code}").classes("t-small muted tnum")
                ui.space()
                ui.button(icon="more_horiz", on_click=lambda: session_dialog(code)).props("flat dense round size=sm") \
                    .tooltip("About sessions, or open another one")

    def page(builder, path):
        def build():
            for p, b in navs.items():
                b.classes(remove="active")
            navs[path].classes(add="active")
            builder(S)
        return build

    with ui.column().classes("w-full p-6 page").style("max-width:1440px"):
        ui.sub_pages({path: page(b, path) for _, path, b, _ in PAGES}).classes("w-full")

    seen = {s: set() for s in ("failed", "done")}
    seen["failed"].update(j.id for j in jm.jobs.values() if j.status == "failed")
    seen["done"].update(j.id for j in jm.jobs.values() if j.status == "done")
    prev = {"loaded": loaded_label()}
    from ..service import guard
    shown_note = [guard.EVENTS[-1][0] if guard.EVENTS else 0]  # this tab shows only notes newer than its start

    def side_tick():
        for seq, msg in list(guard.EVENTS):  # watchdog notes become toasts in every open tab
            if seq > shown_note[0]:
                shown_note[0] = seq
                ui.notify(msg, type="warning", timeout=10000, close_button="Close")
        t = busy_task()
        job = running_job()
        # model
        name = loaded_label()
        if task_is_loading(t):
            what = t.title if t.kind == "load" else "the model"
            ms = f'<div><span class="dot busy"></span>Loading {what}</div><div class="muted">{t.stage}</div>'
        elif name:
            ms = (f'<div><span class="dot {"busy" if t else "ok"}"></span>{name}</div>'
                  f'<div class="muted">{"Making an image" if t else "Ready"}</div>')
        else:
            why = f"Unloaded while {job.title.lower()} runs" if job and job.gpu else "Loads when you press Generate"
            ms = f'<div><span class="dot"></span>No model loaded</div><div class="muted">{why}</div>'
        set_if(model_l, "content", ms)
        if prev["loaded"] and not name and job and job.gpu:
            ui.notify(f"The image model was unloaded to free the GPU for \"{job.title}\"")
        prev["loaded"] = name
        # memory
        pv = estimate_preview(S)
        parts = []
        for nm, (total, blocks) in dict(USAGE).items():
            if nm == "VRAM" and pv and engc.loaded:
                blocks = [b for b in blocks if not b[0].startswith(("SDXL", "Z-Image", "working"))]
            used = sum(g for _, g in blocks)
            parts.append(f'<div class="t-small" style="display:flex;justify-content:space-between">'
                         f'<span class="muted">{nm}</span><span class="tnum">{used:.1f} / {total:.0f} GB</span></div>'
                         + bar_html(total, blocks, pv if nm == "VRAM" else None))
        if pv and "VRAM" in USAGE:
            parts.append(f'<div class="hint">Striped: about {pv[1]:.1f} GB for the picked model</div>')
        from .. import tailnet
        link = tailnet.LINK["text"]
        set_if(link_l, "text", link)
        link_l.set_visibility(bool(link))
        set_if(mem, "content", '<div style="display:flex;flex-direction:column;gap:6px">' + "".join(parts) + "</div>")
        # current work
        queued = sum(j.status == "queued" for j in jm.jobs.values())
        if t:
            step = f" {t.done}/{t.total}" if t.total else ""
            label, detail, pct = f"{t.stage}{step}", t.detail or t.title, 100 * t.done / t.total if t.total else 0
        elif job:
            label = job.title
            detail = f"{job.done}/{job.total}" + (f", {fmt_secs(job.eta)} left" if job.eta else "") if job.total else "Working"
            pct = 100 * job.done / job.total if job.total else 0
        else:
            label, detail, pct = "Nothing running", "", 0
        if queued:
            detail += f" ({queued} waiting)"
        set_if(task_l, "text", label)
        task_l.classes(replace="t-small " + ("work" if t or job else "muted"))
        set_if(task_d, "text", detail)
        set_if(task_p, "content", f'<div class="progress"><span style="width:{pct:.0f}%"></span></div>' if (t or job) else "")
        cancel_b.set_visibility(bool(t or job))
        # job results, wherever you are
        for j in list(jm.jobs.values()):
            if j.status == "failed" and j.id not in seen["failed"]:
                seen["failed"].add(j.id)
                ui.notify(f"{j.title} failed. The reason is on its page and in Activity", type="negative",
                          timeout=8000, close_button="Close")
            elif j.status == "done" and j.id not in seen["done"]:
                seen["done"].add(j.id)
                ui.notify(f"{j.title}: done", type="positive")

    side_tick()
    ui.timer(1.0, side_tick)


def session_dialog(code: str) -> None:
    with ui.dialog() as d, ui.card().classes("panel").style("width:420px"):
        ui.label(f"Session {code}").classes("t-head")
        ui.label("Your form, images and settings are saved on this VM under this code. Opening the app again "
                 "continues it, even after the connection drops. To open another saved session, type its code.") \
            .classes("t-small")
        other = ui.input(placeholder="4 digits").props("outlined dense maxlength=4").classes("w-full")

        def go():
            c = (other.value or "").strip()
            if session.load(c) is None:
                ui.notify(f"No session {c} on this VM", type="warning")
                return
            app.storage.browser["code"] = c
            ui.navigate.reload()

        with ui.row().classes("w-full justify-end gap-2"):
            button("Close", d.close)
            button("Open that session", go)
    d.open()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--access", choices=["tailscale", "none"], default="tailscale")
    ap.add_argument("--sync", action="store_true", help="pull presets + LoRAs from the sync repo at start")
    ap.add_argument("--host-name", default="sdxlr", help="Tailscale name: http://NAME:PORT")
    ap.add_argument("--drive", type=Path, help="Drive SDXLR folder: its LoRAs and Z-Image models are listed too")
    ap.add_argument("--zimage-only", action="store_true", help="hide SDXL (the sdxl notebook runs Fooocus)")
    a = ap.parse_args()
    import os
    if a.drive:
        os.environ["SDXLR_DRIVE"] = str(a.drive)
    if a.zimage_only:
        os.environ["SDXLR_ZIMAGE_ONLY"] = "1"
    setup_logging()
    from ..service import guard
    guard.start(core.engc, core.jm)
    for d in (MODELS, LORAS, core.DATASETS, core.OUTPUTS, core.UPLOADS, THUMBS, presets.PRESETS):
        d.mkdir(parents=True, exist_ok=True)
    if a.sync and settings.get()["sync_repo"]:
        try:
            log.info(presets.pull(settings.get()["sync_repo"]))
        except Exception as e:
            log.info(f"sync pull failed: {e}")
    threading.Thread(target=state.autosave_loop, daemon=True).start()
    threading.Thread(target=core.sample_usage_loop, daemon=True).start()
    from ..service import files as _files
    _files.start_refresh()  # cached listings for every timer (first scan synchronous)
    from ..service import masks
    masks.register(app)  # POST /api/mask/<session>: the painter's mask
    for d in _files.SERVED:  # only what the pages show: not sessions, logs, datasets, models
        (core.WORK / d).mkdir(parents=True, exist_ok=True)
        app.add_media_files(f"/work/{d}", core.WORK / d)
    secret_f = Path.home() / ".sdxlr_ui_secret"  # outside WORK, which is browsable and gets shredded/copied
    (core.WORK / ".ui_secret").unlink(missing_ok=True)  # the old location
    if not secret_f.exists():
        secret_f.write_text(secrets.token_hex(16))

    def announce(u):
        log.info("=" * 60)
        log.info(f"Open the app: {u}")
        log.info("=" * 60)

    if a.access == "tailscale":
        from .. import tailnet

        def ts_up():
            try:
                announce(tailnet.up(a.port, hostname=a.host_name) + "  (on a device signed in to your Tailscale)")
                threading.Thread(target=tailnet.watch_links, daemon=True).start()
            except Exception as e:
                log.info(f"Tailscale failed: {e}")
                log.info(f"Add TS_AUTHKEY in Colab Secrets, run Setup again, then Launch. Local address: http://127.0.0.1:{a.port}")

        app.on_startup(lambda: threading.Thread(target=ts_up, daemon=True).start())
        app.on_shutdown(tailnet.down)
    else:
        app.on_startup(lambda: announce(f"http://127.0.0.1:{a.port}"))
    log.info(f"app starting on port {a.port}")
    ui.run(root, host="0.0.0.0", port=a.port, title="SDXL Research", show=False, reload=False, dark=True,
           storage_secret=secret_f.read_text(), reconnect_timeout=600, favicon="◐")
