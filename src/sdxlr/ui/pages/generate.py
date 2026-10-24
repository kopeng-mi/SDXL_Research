"""Generate: model card, prompt and options on the left; the image, its progress and recent images on the right."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from nicegui import run, ui

from ... import history, presets
from ...activity import Cancelled, act
from ..core import (ASPECTS, SAMPLERS, STYLES, TOOL_LABEL, Z_ENCODERS, busy_task, check_generate, check_model, engc,
                    err, generate_blocking, is_loaded, load_blocking, lora_names, model_names, picked_label, reuse,
                    save_upload, task_is_loading, thumb, url, z_names)
from ..theme import button, labeled, main_button, seg

PRESET_KEYS = ("kind", "sdxl_model", "z_model", "mode", "prompt", "negative", "style", "aspect", "count", "seed",
               "use_adv", "steps", "cfg", "sampler", "clip_skip", "loras")


async def do_load(S):
    try:
        check_model(S)
    except ValueError as e:
        ui.notify(str(e), type="warning")
        return
    if busy_task():
        return
    task = act.start("load", picked_label(S))
    try:
        await run.io_bound(load_blocking, copy.deepcopy(S), task)
        task.finish()
        ui.notify(f"{picked_label(S)} is ready", type="positive")
    except Cancelled:
        task.fail("")
        ui.notify("Load cancelled")
    except Exception as e:
        task.fail(err(e))
        ui.notify(err(e), type="negative", multi_line=True)


async def do_generate(S):
    if busy_task():  # the button is disabled while busy; this guards a fast double click
        return
    try:
        check_generate(S)
    except ValueError as e:
        ui.notify(str(e), type="warning")
        return
    tool = S["tool"]
    task = act.start("generate", (S["prompt"] or TOOL_LABEL[tool])[:50])
    try:
        paths, info, seed = await run.io_bound(generate_blocking, copy.deepcopy(S), task)
        task.finish(info)
        S.update(image=paths[0], info=info, last_seed=seed, boxes=[])
        if tool:  # keep working on the new result; the original stays in the recent images
            S["src"] = paths[0]
            ui.notify(f"{TOOL_LABEL[tool]}: done. The original is still in the recent images", type="positive")
    except Cancelled:
        task.fail("")
        ui.notify("Cancelled")
    except Exception as e:
        task.fail(err(e))
        ui.notify(err(e), type="negative", multi_line=True)


def page(S):
    with ui.row().classes("w-full gap-8 no-wrap items-start two-col"):
        left = ui.column().classes("gap-4").style("width:500px; min-width:400px")
        right = ui.column().classes("gap-3").style("flex:1 1 0; min-width:0")

    # ----- model card -----
    with left:
        with ui.column().classes("panel w-full gap-3"):
            with ui.row().classes("w-full items-center no-wrap gap-3"):
                seg(["Z-Image", "SDXL"], S, "kind")
                zsel = ui.select(z_names(), with_input=True, new_value_mode="add-unique").bind_value(S, "z_model") \
                    .props("outlined dense").classes("grow").bind_visibility_from(S, "kind", value="Z-Image")
                ssel = ui.select(model_names(), with_input=True).bind_value(S, "sdxl_model") \
                    .props("outlined dense label='Pick a checkpoint'").classes("grow") \
                    .bind_visibility_from(S, "kind", value="SDXL")
            with ui.row().classes("w-full items-center no-wrap gap-3"):
                mstate = ui.html("", sanitize=False).classes("t-small grow")
                mbtn = button("Load", lambda: model_action())  # async; NiceGUI awaits the returned coroutine

        async def model_action():
            t = busy_task()
            if t:
                t.cancel()
                ui.notify("Cancelling. A download stops now; loading stops after the current step")
            elif is_loaded(S):
                await run.io_bound(engc.unload)
                ui.notify("Model unloaded")
            else:
                await do_load(S)

        # ----- prompt -----
        with ui.column().classes("w-full gap-1"):
            with ui.row().classes("w-full items-center"):
                ui.label("Prompt").classes("lbl")
                ui.space()
                with button("Recent", icon="history").props("size=sm"):
                    with ui.menu() as recent_menu:
                        pass
                with button("Presets", icon="bookmark_border").props("size=sm"):
                    with ui.menu() as preset_menu:
                        pass
            prompt = ui.textarea(placeholder="ohwx_person, a woman in a green silk saree, standing in a garden") \
                .bind_value(S, "prompt").props("outlined autogrow input-style='min-height:72px'").classes("w-full")
            ui.label("Ctrl+Enter to generate").classes("hint")

        def fill_recent():
            recent_menu.clear()
            with recent_menu:
                ps = history.prompts()
                if not ps:
                    ui.menu_item("No prompts yet").props("disable")
                for p in ps:
                    ui.menu_item(p[:90], on_click=lambda p=p: S.__setitem__("prompt", p))

        def fill_presets():
            preset_menu.clear()
            with preset_menu:
                names = presets.list_presets()
                for n in names:
                    ui.menu_item(f"Load preset: {n}", on_click=lambda n=n: (S.update(presets.load(n)), draw_loras(),
                                                                           ui.notify(f"Preset {n} loaded")))
                if names:
                    ui.separator()
                ui.menu_item("Save the current settings…", on_click=save_preset)

        recent_menu.on("before-show", fill_recent)
        preset_menu.on("before-show", fill_presets)

        def save_preset():
            with ui.dialog() as d, ui.card().classes("panel").style("min-width:380px"):
                ui.label("Save these settings as a preset").classes("t-head")
                name = labeled("Name", ui.input())
                existing = presets.list_presets()
                ui.label("Saves the model, mode, size, prompt, seed and LoRAs. "
                         + (f"Existing presets: {', '.join(existing)}. The same name replaces one." if existing else "")) \
                    .classes("hint")

                def ok():
                    n = (name.value or "").strip()
                    if not n:
                        ui.notify("Give the preset a name", type="warning")
                        return
                    presets.save(n, {k: S[k] for k in PRESET_KEYS})
                    d.close()
                    ui.notify(f"Preset {n} saved", type="positive")

                with ui.row().classes("w-full justify-end gap-2"):
                    button("Cancel", d.close)
                    button("Save preset", ok)
            d.open()

        # ----- options -----
        with ui.row().classes("w-full items-end no-wrap gap-3"):
            with ui.column().classes("gap-0"):
                ui.label("Mode").classes("lbl")
                seg(["Quality", "Speed", "Fast"], S, "mode")
            labeled("Size", ui.select(list(ASPECTS)).bind_value(S, "aspect"))
        with ui.row().classes("w-full items-start no-wrap gap-3"):
            labeled("Images", ui.number(min=1, max=4, step=1, format="%d").bind_value(S, "count")).style("width:90px")
            with ui.column().classes("grow gap-0"):
                ui.label("Seed").classes("lbl")
                with ui.row().classes("w-full no-wrap items-center gap-1"):
                    ui.number(step=1, format="%d").bind_value(S, "seed").props("outlined dense").classes("grow")
                    ui.button(icon="casino", on_click=lambda: S.__setitem__("seed", -1)).props("flat dense round") \
                        .tooltip("Random each time")
                    ui.button(icon="replay", on_click=lambda: S.__setitem__("seed", S["last_seed"])
                              if S["last_seed"] is not None else ui.notify("Generate once first")) \
                        .props("flat dense round").tooltip("Use the last image's seed")
                seed_hint = ui.label().classes("hint")

        # ----- LoRAs -----
        with ui.column().classes("w-full gap-1"):
            ui.label("LoRAs").classes("lbl")
            lora_box = ui.column().classes("w-full gap-1")

        def draw_loras():
            lora_box.clear()
            names = lora_names()
            with lora_box:
                for i, row in enumerate(S["loras"]):
                    with ui.row().classes("w-full items-center no-wrap gap-2"):
                        ui.select(names, value=row[0] if row[0] in names else None, with_input=True,
                                  on_change=lambda e, i=i: S["loras"][i].__setitem__(0, e.value)) \
                            .props("outlined dense label='Pick a LoRA'").classes("grow")
                        sl = ui.slider(min=-1, max=2, step=0.01, value=row[1],
                                       on_change=lambda e, i=i: S["loras"][i].__setitem__(1, e.value)) \
                            .classes("w-28").tooltip("Strength")
                        ui.number(format="%.2f", step=0.05, min=-1, max=2).bind_value(sl, "value") \
                            .props("outlined dense").style("width:76px").tooltip("From -1 to 2")
                        ui.button(icon="close", on_click=lambda i=i: (S["loras"].pop(i), draw_loras())) \
                            .props("flat dense round size=sm").tooltip("Remove")
                if not names:
                    ui.label("No LoRAs yet. Add some in Library, or train one.").classes("hint")
                elif any(not r[0] for r in S["loras"]):
                    ui.label("Rows without a LoRA picked are skipped.").classes("hint")
                button("Add a LoRA", lambda: (S["loras"].append([None, 0.8]), draw_loras()), icon="add") \
                    .props("size=sm")

        draw_loras()

        # ----- go -----
        with ui.row().classes("w-full no-wrap gap-2 items-center"):
            gen_b = main_button("Generate", lambda: do_generate(S))
            stop_b = button("Cancel", lambda: (busy_task() and busy_task().cancel()), kind="stop").style("height:46px")
        ui.label().bind_text_from(S, "info", lambda v: f"Last image: {v}" if v else "").classes("hint tnum")
        tool_hint = ui.label().classes("hint")

        # ----- more options, grouped -----
        with ui.expansion("Style and negative prompt").classes("w-full"):
            labeled("Style", ui.select(list(STYLES)).bind_value(S, "style"), "Added to the end of your prompt")
            labeled("Negative prompt", ui.textarea().bind_value(S, "negative").props("autogrow"),
                    "What to avoid. SDXL only; Z-Image ignores it").bind_visibility_from(S, "kind", value="SDXL")
        with ui.expansion("Steps and sampler").classes("w-full"):
            ui.checkbox("Set steps and CFG myself (otherwise the mode decides)").bind_value(S, "use_adv")
            with ui.row().classes("w-full no-wrap gap-2").bind_visibility_from(S, "use_adv"):
                labeled("Steps", ui.number(min=1, max=80, step=1, format="%d").bind_value(S, "steps"))
                labeled("CFG", ui.number(min=0, max=15, step=0.5).bind_value(S, "cfg"))
            with ui.row().classes("w-full no-wrap gap-2").bind_visibility_from(S, "kind", value="SDXL"):
                labeled("Sampler", ui.select(list(SAMPLERS)).bind_value(S, "sampler"))
                labeled("Clip skip", ui.number(min=1, max=3, step=1, format="%d").bind_value(S, "clip_skip"))
        with ui.expansion("Model options").classes("w-full"):
            with ui.column().classes("w-full gap-2").bind_visibility_from(S, "kind", value="Z-Image"):
                labeled("Prompt reader (text encoder)", ui.select(
                    {"official": "Official", "abliterated": "Uncensored (follows more prompts)"},
                    with_input=True, new_value_mode="add-unique").bind_value(S, "z_te"),
                    "The part that reads your prompt. You can also type a Hugging Face repo")
                with ui.column().classes("gap-0"):
                    ui.label("Prompt reader size").classes("lbl")
                    seg({"nf4": "Small, 2.5 GB", "int8": "Larger, 4.4 GB, a bit more accurate"}, S, "z_quant")
            with ui.column().classes("gap-0").bind_visibility_from(S, "kind", value="SDXL"):
                ui.label("Image decoder (VAE)").classes("lbl")
                seg({"fp16fix": "Fixed for this GPU (recommended)", "model": "The checkpoint's own"}, S, "vae")
                ui.label("The checkpoint's own decoder can give black images on a T4").classes("hint")
            ui.label("Changing these needs the model to load again.").classes("hint")

    # ----- image side -----
    TOOL_TEXT = {
        "area": "Change an area: drag on the image to mark what to change. Your prompt describes the new content.",
        "whole": "Change the whole image: your prompt guides it. Low strength keeps the photo, high reimagines it.",
        "upscale": "Upscale: makes the image bigger and adds detail in tiles. The prompt is optional guidance.",
    }
    with right:
        banner = ui.column().classes("banner w-full gap-2")
        with banner:
            tool_l = ui.label().classes("t-small")
            with ui.row().classes("w-full items-center no-wrap gap-3"):
                for key, tip, lo, hi in (("strength", "Low keeps more of the original; 1 repaints the area fully", 0.1, 1.0),
                                         ("whole_strength", "0.3 keeps the photo, 0.8 reimagines it", 0.1, 1.0),
                                         ("up_strength", "0.2 to 0.35 adds detail without changing the image", 0.1, 0.6)):
                    with ui.row().classes("grow items-center no-wrap gap-3") as row:
                        ui.label().bind_text_from(S, key, lambda v: f"Strength {v:.2f}").classes("t-small tnum") \
                            .style("white-space:nowrap").tooltip(tip)
                        ui.slider(min=lo, max=hi, step=0.05).bind_value(S, key).classes("grow")
                    row.bind_visibility_from(S, "tool", value={"strength": "area", "whole_strength": "whole",
                                                               "up_strength": "upscale"}[key])
                with ui.row().classes("items-center no-wrap gap-1") as scale_row:
                    ui.label("Size").classes("t-small")
                    seg({1.5: "1.5x", 2.0: "2x"}, S, "up_scale")
                scale_row.bind_visibility_from(S, "tool", value="upscale")
                clear_marks = button("Clear marks", lambda: (S["boxes"].clear(), draw_canvas()))
                clear_marks.bind_visibility_from(S, "tool", value="area")
                button("Done", lambda: set_tool(None))
        canvas = ui.element("div").classes("canvas w-full")
        with ui.row().classes("w-full items-center gap-2"):
            earlier = ui.label("").classes("hint")
            ui.space()

            async def on_upload(e):
                try:
                    p = save_upload(await e.file.read(), e.file.name)
                except ValueError as ex:
                    ui.notify(str(ex), type="warning")
                    return
                except Exception as ex:
                    ui.notify(err(ex), type="negative")
                    return
                S.update(image=p, src=p, boxes=[])
                draw_canvas()
                ui.notify("Image added. Pick what to do with it below", type="positive")

            up = ui.upload(auto_upload=True, on_upload=on_upload, max_files=1) \
                .props("accept=image/* hide-upload-btn").style("display:none")
            button("Upload an image", lambda: up.run_method("pickFiles"), icon="upload")
        with ui.row().classes("w-full items-center gap-2") as actions:
            reuse_b = button("Reuse its settings", lambda: reuse_shown(), icon="settings_backup_restore")
            area_b = button("Change an area", lambda: set_tool("area"), icon="crop_free")
            whole_b = button("Change the whole image", lambda: set_tool("whole"), icon="auto_fix_high")
            up_b = button("Upscale", lambda: set_tool("upscale"), icon="open_in_full")
            ui.space()
            dl_b = button("Download", lambda: ui.download.file(S["image"]), icon="download")
            clear_b = button("Clear", lambda: (S.update(image=None, tool=None, boxes=[]), draw_canvas()), icon="close") \
                .tooltip("Empty the image area. The image stays in History")
        recent = ui.row().classes("w-full gap-2")

    shown = {"img": None, "tool": "x", "boxes": -1, "recent": None}
    drag = {}

    def svg_boxes():
        return "".join(f'<rect x="{min(a, c)}" y="{min(b, d)}" width="{abs(c - a)}" height="{abs(d - b)}" '
                       f'fill="rgba(232,163,61,.25)" stroke="#E8A33D" stroke-width="4"/>' for a, b, c, d in S["boxes"])

    def on_mouse(e):
        if e.type == "mousedown":
            drag["p"] = (e.image_x, e.image_y)
        elif e.type == "mouseup" and "p" in drag:
            x0, y0 = drag.pop("p")
            if abs(e.image_x - x0) > 8 and abs(e.image_y - y0) > 8:
                S["boxes"].append([x0, y0, e.image_x, e.image_y])
                draw_canvas()

    overlay = {}

    def shown_img():
        return S["src"] if S["tool"] else S["image"]

    def draw_canvas():
        img = shown_img()
        shown.update(img=img, tool=S["tool"], boxes=len(S["boxes"]))
        canvas.clear()
        with canvas:
            if img and Path(img).exists():
                if S["tool"] == "area":
                    ii = ui.interactive_image(url(img), on_mouse=on_mouse, events=["mousedown", "mouseup"]).classes("w-full")
                    ii.content = svg_boxes()
                else:
                    ui.image(url(img)).classes("w-full").props("fit=contain no-transition")
            else:
                with ui.column().classes("items-center gap-1"):
                    ui.label("Your image shows up here").classes("muted")
                    ui.label("Write a prompt and press Generate, or upload an image to change or upscale it.") \
                        .classes("hint")
            with ui.column().classes("overlay gap-1") as ov:
                overlay["title"] = ui.label().classes("work")
                overlay["bar"] = ui.html("", sanitize=False).classes("w-full")
                overlay["detail"] = ui.label().classes("hint tnum")
            overlay["box"] = ov
            ov.set_visibility(False)

    def draw_recent():
        items = history.items(12)
        sig = [str(p) for p, _ in items] + [shown_img()]
        if sig == shown["recent"]:
            return
        shown["recent"] = sig
        recent.clear()
        with recent:
            for p, m in items:
                tip = "Uploaded: " + m.get("name", "") if m.get("uploaded") else (m.get("prompt_raw") or "")[:80]
                ui.image(thumb(p, 160)).classes("thumb" + (" sel" if str(p) == shown_img() else "")) \
                    .props("no-transition no-spinner").tooltip(tip).on("click", lambda p=p: pick(str(p)))

    def pick(p):
        S.update(image=p, boxes=[])
        if S["tool"]:
            S["src"] = p  # the open tool now works on this image
        draw_canvas()

    def set_tool(tool):
        if tool and not S["image"]:
            ui.notify("Generate, pick or upload an image first", type="warning")
            return
        S.update(tool=tool, src=S["image"] if tool else None, boxes=[])
        draw_canvas()

    def reuse_shown():
        j = Path(S["image"]).with_suffix(".json") if S["image"] else None
        if j and j.exists():
            meta = json.loads(j.read_text(encoding="utf-8"))
            if meta.get("uploaded"):
                ui.notify("This image was uploaded, so it has no settings to reuse", type="warning")
                return
            missing = reuse(S, meta)
            draw_loras()
            ui.notify("Settings filled in from this image. The seed is now fixed, so it makes the same image",
                      type="positive")
            if missing:
                ui.notify(f"Missing LoRAs (add them in Library): {', '.join(missing)}", type="warning")

    def tick():
        if shown["img"] != shown_img() or shown["tool"] != S["tool"] or shown["boxes"] != len(S["boxes"]):
            draw_canvas()
        draw_recent()
        t = busy_task()
        # model card
        if task_is_loading(t):
            pct = f" {100 * t.done // t.total}%" if t.total else ""
            ms = f'<span class="dot busy"></span>Loading {picked_label(S)}: {t.stage.lower()}{pct}'
            mb = "Cancel"
        elif is_loaded(S):
            ms, mb = f'<span class="dot ok"></span>{picked_label(S)} is ready', "Unload"
        else:
            other = " (another model is in memory)" if engc.loaded else ""
            ms, mb = f'<span class="dot"></span>{picked_label(S)} is not loaded{other}', "Load"
        if mstate.content != ms:
            mstate.content = ms
        if mbtn.text != mb:
            mbtn.text = mb
            mbtn.classes(replace="btn-stop" if mb == "Cancel" else "btn-2")
        # main button + overlay
        idle_label = TOOL_LABEL[S["tool"]] + (f" {S['up_scale']:g}x" if S["tool"] == "upscale" else "")
        if t:
            unit = " tiles" if t.stage.startswith("Upscaling") else ""
            step = f" {t.done}/{t.total}{unit}" if t.total else ""
            label = f"{t.stage}{step}"
            if gen_b.text != label:
                gen_b.text = label
            gen_b.disable()
            stop_b.set_visibility(True)
            if "box" in overlay:
                overlay["box"].set_visibility(True)
                overlay["title"].text = label
                pct = 100 * t.done / t.total if t.total else 0
                html = f'<div class="progress"><span style="width:{pct:.0f}%"></span></div>'
                if overlay["bar"].content != html:
                    overlay["bar"].content = html
                overlay["detail"].text = t.detail or f"{int(t.elapsed)} s"
        else:
            if gen_b.text != idle_label:
                gen_b.text = idle_label
            gen_b.enable()
            stop_b.set_visibility(False)
            if "box" in overlay:
                overlay["box"].set_visibility(False)
        th = {"area": "Uses your prompt for the marked area.", "whole": "Uses your prompt and the strength above the image.",
              "upscale": "Press Done above the image to go back to making new images."}.get(S["tool"], "")
        if tool_hint.text != th:
            tool_hint.text = th
        # image actions and hints
        banner.set_visibility(bool(S["tool"]))
        if S["tool"] and tool_l.text != TOOL_TEXT[S["tool"]]:
            tool_l.text = TOOL_TEXT[S["tool"]]
        has = bool(S["image"])
        actions.set_visibility(has)
        for b in (area_b, whole_b, up_b):
            b.set_visibility(not S["tool"])
        meta_p = Path(S["image"]).with_suffix(".json") if has else None
        uploaded = bool(meta_p and meta_p.exists() and '"uploaded": true' in meta_p.read_text(encoding="utf-8"))
        reuse_b.set_visibility(has and not S["tool"] and not uploaded)
        latest = history.items(1)
        is_old = has and latest and str(latest[0][0]) != S["image"]
        txt = "An earlier image. Its settings may differ from the form." if is_old and not S["tool"] and not uploaded else ""
        if earlier.text != txt:
            earlier.text = txt
        sh = "Random each time" if int(S["seed"] or -1) < 0 else "Fixed: the same settings make the same image"
        if seed_hint.text != sh:
            seed_hint.text = sh
        for sel, names in ((zsel, z_names()), (ssel, model_names())):
            if list(sel.options) != names:
                sel.set_options(names, value=sel.value if sel.value in names else None)

    async def on_key(e):
        a = e.args or {}
        if a.get("key") == "Enter" and (a.get("ctrlKey") or a.get("metaKey")):
            await do_generate(S)

    prompt.on("keydown", on_key, args=["key", "ctrlKey", "metaKey"])
    draw_canvas()
    tick()
    ui.timer(0.5, tick)
