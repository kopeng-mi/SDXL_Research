"""History: every image with the settings that made it; make any of them again."""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from ... import history
from ..core import err, reuse, thumb, url
from ..theme import button, title


def page(S):
    with ui.row().classes("w-full items-center gap-4"):
        title("History")
        ui.space()

        async def up(e):
            try:
                missing = reuse(S, history.read(await e.file.read(), e.file.name))
                ui.notify("Settings loaded into Generate", type="positive")
                if missing:
                    ui.notify(f"Missing LoRAs (add them in Library): {', '.join(missing)}", type="warning")
                ui.navigate.to("/")
            except Exception as ex:
                ui.notify(err(ex), type="negative")

        ui.upload(label="Load settings from a .json or a PNG made here", auto_upload=True, on_upload=up) \
            .props("flat bordered accept=.json,.png hide-upload-btn").classes("drop").style("width:360px")
    grid = ui.row().classes("w-full gap-3 pt-2")

    def show(p: Path, m: dict):
        with ui.dialog() as d, ui.card().classes("panel").style("width:960px; max-width:95vw"):
            with ui.row().classes("w-full no-wrap gap-5 two-col"):
                ui.image(url(p)).props("fit=contain no-transition no-spinner").style("width:55%; max-height:75vh")
                with ui.column().classes("gap-2").style("flex:1 1 0"):
                    ui.label(m.get("prompt_raw") or m.get("prompt", "")).classes("t-head").style("line-height:1.35")
                    model = m.get("z_model") if m.get("kind") == "Z-Image" else m.get("sdxl_model")
                    rows = [("Model", f"{m.get('kind')} {model}"), ("Mode", m.get("mode")),
                            ("Steps", m.get("steps")), ("CFG", m.get("cfg")), ("Seed", m.get("seed")),
                            ("Size", "x".join(map(str, m.get("size", [])))),
                            ("LoRAs", ", ".join(f"{n} at {w:.2f}" for n, w in m.get("loras", [])) or "None"),
                            ("Made", f"{m.get('made', '')}, took {m.get('seconds', '?')} s")]
                    if m.get("inpaint"):
                        rows.append(("Area change", f"strength {m.get('strength')}"))
                    for k, v in rows:
                        with ui.row().classes("no-wrap gap-2 t-small"):
                            ui.label(k).classes("muted").style("width:90px")
                            ui.label(str(v))
                    with ui.row().classes("gap-2 pt-3"):
                        ui.button("Make it again", on_click=lambda: make_again(m, d)) \
                            .props("unelevated no-caps color=grey-3 text-color=grey-10").classes("btn-main").style("width:auto; padding:0 18px")
                        button("Download the image", lambda: ui.download.file(p), icon="download")
                        button("Download the settings", lambda: ui.download.file(p.with_suffix(".json")), icon="data_object")
        d.open()

    def make_again(m, d):
        missing = reuse(S, m)
        d.close()
        ui.notify("Settings loaded. Press Generate to make it again (same seed, same image)", type="positive")
        if missing:
            ui.notify(f"Missing LoRAs (add them in Library): {', '.join(missing)}", type="warning")
        ui.navigate.to("/")

    seen = {"top": "x"}

    def draw():
        items = history.items(150)
        seen["top"] = items[0][0].name if items else None
        grid.clear()
        with grid:
            if not items:
                ui.label("Images you generate show up here, with the settings to make them again.").classes("muted")
            for p, m in items:
                with ui.column().classes("tile gap-1"):
                    ui.image(thumb(p)).props("no-transition no-spinner").on("click", lambda p=p, m=m: show(p, m))
                    ui.label((m.get("prompt_raw") or m.get("prompt", ""))[:70]).classes("t-small muted") \
                        .style("line-height:1.3; height:34px; overflow:hidden")

    def tick():
        items = history.items(1)
        if (items[0][0].name if items else None) != seen["top"]:
            draw()

    draw()
    ui.timer(3.0, tick)
