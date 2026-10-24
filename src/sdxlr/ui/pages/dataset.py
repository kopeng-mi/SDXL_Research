"""Dataset: add photos, write captions, and edit each caption in place."""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from ...service import files
from ..core import UPLOADS, datasets, err, images, submit_caption, submit_import, thumb
from ..theme import JobLine, button, labeled, main_button, title

CAPTIONERS = {"qwen": "Qwen3-VL 4B: accurate sentences", "qwen-2b": "Qwen3-VL 2B: faster",
              "joy": "JoyCaption: uncensored", "wd": "WD tagger: tags, for Pony or anime models"}


def page(S):
    if not S["ds_pick"] and datasets():
        S["ds_pick"] = datasets()[-1]
    staged: list[str] = []

    with ui.row().classes("w-full gap-8 no-wrap items-start two-col"):
        with ui.column().classes("gap-4").style("width:420px; min-width:340px"):
            title("Dataset")
            with ui.column().classes("panel w-full gap-3"):
                ui.label("Add photos").classes("t-head")
                labeled("Dataset name", ui.input().bind_value(S, "ds_name"), "New name = new dataset; same name adds to it")

                async def on_up(e):
                    d = UPLOADS / "dataset"
                    d.mkdir(parents=True, exist_ok=True)
                    p = d / Path(e.file.name).name
                    await e.file.save(p)
                    staged.append(str(p))
                    staged_l.text = f"{len(staged)} files ready. Press Import."

                ui.upload(label="Drop photos or a zip here, or click to pick (captions: same-name .txt files)",
                          multiple=True, auto_upload=True, on_upload=on_up) \
                    .props("flat bordered accept=.zip,.txt,image/* hide-upload-btn").classes("drop")
                staged_l = ui.label().classes("hint")
                labeled("Or paste links", ui.textarea(placeholder="https://…, org/repo/file.zip, /content/drive/MyDrive/…, or test")
                        .bind_value(S, "ds_link").props("autogrow"), "One per line. 'test' adds 5 sample photos")

                def do_import():
                    try:
                        job = submit_import(S, staged)
                    except ValueError as e:
                        ui.notify(str(e), type="warning")
                        return
                    staged.clear()
                    staged_l.text = ""
                    S["ds_link"] = ""
                    S["ds_pick"] = S["ds_name"].strip()
                    pick.set_options(sorted({*datasets(), S["ds_pick"]}), value=S["ds_pick"])
                    imp_line.track(job)

                main_button("Import", do_import)
                imp_line = JobLine(("import",))

            with ui.column().classes("panel w-full gap-3"):
                ui.label("Write captions").classes("t-head")
                labeled("Captioner", ui.select(CAPTIONERS).bind_value(S, "cap_model"))
                with ui.row().classes("w-full no-wrap gap-2"):
                    labeled("Trigger word", ui.input().bind_value(S, "cap_trigger"), "Starts every caption")
                    labeled("Class word", ui.input().bind_value(S, "cap_class"), "What the subject is, like woman")
                with ui.expansion("More caption options").classes("w-full"):
                    labeled("Your own instruction", ui.textarea().bind_value(S, "cap_prompt").props("autogrow"),
                            "Empty = describe clothes, pose, place and light, not the face. {cls} = class word")
                    labeled("Caption only the first", ui.number(min=0, step=1, format="%d").bind_value(S, "cap_limit"),
                            "0 = all photos. Use a few to preview the style")
                    ui.checkbox("Replace captions that already exist").bind_value(S, "cap_over")

                def do_caption():
                    try:
                        cap_line.track(submit_caption(S))
                        ui.notify("Captioning queued. The image model is unloaded to free the GPU")
                    except ValueError as e:
                        ui.notify(err(e), type="warning")

                main_button("Write captions", do_caption)
                cap_line = JobLine(("caption",))

        with ui.column().classes("gap-3").style("flex:1 1 0; min-width:0"):
            with ui.row().classes("w-full items-end gap-3"):
                pick = labeled("Showing", ui.select(datasets()).bind_value(S, "ds_pick")).style("width:280px")
                counts = ui.label().classes("muted tnum pb-2")
            grid = ui.row().classes("w-full gap-3")

    def save_caption(p: Path, text: str):
        t = p.with_suffix(".txt")
        text = (text or "").strip()
        if text:
            t.write_text(text, encoding="utf-8")
        elif t.exists():
            t.unlink()  # an empty caption is no caption
        files.poke()
        update_counts()

    def big(p: Path):
        with ui.dialog() as d, ui.card().classes("panel").style("max-width:900px"):
            ui.image(thumb(p, 900)).props("fit=contain no-transition no-spinner").style("max-height:75vh; min-width:400px")
            with ui.row().classes("w-full justify-end"):
                button("Close", d.close)
        d.open()

    seen = {"sig": None}

    def update_counts():
        n, caps = files.ds_counts(S["ds_pick"])
        counts.text = f"{n} photos, {caps} with captions" if S["ds_pick"] else ""

    def draw():
        imgs = images(S["ds_pick"])
        seen["sig"] = (S["ds_pick"], [p.name for p in imgs], files.ds_counts(S["ds_pick"])[1])
        update_counts()
        grid.clear()
        with grid:
            if not S["ds_pick"]:
                ui.label("Add photos on the left. They show up here with their captions.").classes("muted")
            elif not imgs:
                ui.label("No photos in this dataset yet.").classes("muted")
            for p in imgs:
                with ui.column().classes("tile gap-1"):
                    ui.image(thumb(p)).props("no-transition no-spinner").on("click", lambda p=p: big(p))
                    ui.textarea(value=files.caption(S["ds_pick"], p.name), placeholder="No caption") \
                        .props("outlined dense autogrow input-style='font-size:13px;line-height:1.35'").classes("w-full") \
                        .on("blur", lambda e, p=p: save_caption(p, e.sender.value))

    def tick():
        names = datasets()
        if list(pick.options) != sorted({*names, *([S["ds_pick"]] if S["ds_pick"] else [])}):
            pick.set_options(sorted({*names, *([S["ds_pick"]] if S["ds_pick"] else [])}), value=S["ds_pick"])
        imgs = images(S["ds_pick"])
        sig = (S["ds_pick"], [p.name for p in imgs], files.ds_counts(S["ds_pick"])[1])
        if seen["sig"] != sig:
            draw()

    draw()
    ui.timer(2.0, tick)
