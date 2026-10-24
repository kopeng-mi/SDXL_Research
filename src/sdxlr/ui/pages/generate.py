"""Generate: model card, prompt and options on the left; the image, its progress and recent images on the right."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from nicegui import app, run, ui

from ... import presets
from ...activity import Cancelled, act
from ... import loraformat
from ...service import files
from ..core import (ASPECTS, MAX_UPLOAD, SAMPLERS, STYLES, TOOL_LABEL, Z_ENCODERS, busy_task, check_generate, check_model, engc,
                    err, generate_blocking, is_loaded, load_blocking, lora_names, model_names, picked_label, reuse,
                    save_upload, task_is_loading, thumb, url, z_names)
from ..components.before_after import BeforeAfter
from ..components.mask_painter import MaskPainter
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


async def do_generate(S, on_done=None):
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
        files.poke()
        S.update(image=paths[0], info=info, last_seed=seed, boxes=[])
        if on_done:
            on_done(paths, tool)
        if tool:
            more = f" ({len(paths)} variations are in the recent images)" if len(paths) > 1 else ""
            ui.notify(f"{TOOL_LABEL[tool]}: done{more}", type="positive")
    except Cancelled:
        task.fail("")
        files.poke()  # variations finished before the cancel are already in History
        ui.notify("Cancelled. Variations already finished are kept in the recent images" if (tool and S["edit_count"] > 1)
                  else "Cancelled")
    except Exception as e:
        task.fail(err(e))
        ui.notify(err(e), type="negative", multi_line=True)


def page(S):
    with ui.row().classes("w-full gap-8 no-wrap items-start two-col"):
        left = ui.column().classes("gap-4").style("width:500px; min-width:400px")
        right = ui.column().classes("gap-3").style("flex:1 1 0; min-width:0; position:sticky; top:12px")

    # ----- model card -----
    with left:
        with ui.column().classes("panel w-full gap-3"):
            with ui.row().classes("w-full items-center no-wrap gap-3"):
                seg(["Z-Image", "SDXL"], S, "kind", on_change=lambda e: on_kind())
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
                ps = files.prompts()
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

        def add_trigger(t):
            p = S["prompt"] or ""
            if t.lower() not in p.lower():
                S["prompt"] = f"{t}, {p}".strip(", ")

        def draw_loras():
            lora_box.clear()
            fam = "zimage" if S["kind"] == "Z-Image" else "sdxl"
            lo, hi = (0.0, 2.5) if fam == "zimage" else (-1.0, 2.0)
            names = lora_names()
            usable, hidden = [], []
            for n in names:
                info = files.lora_info(n)
                ok, why = loraformat.compatible(info, fam) if info else (True, "")
                (usable if ok else hidden).append((n, why))
            ok_names = [n for n, _ in usable]
            with lora_box:
                for i, row in enumerate(S["loras"]):
                    with ui.row().classes("w-full items-center no-wrap gap-2"):
                        ui.select(ok_names, value=row[0] if row[0] in ok_names else None, with_input=True,
                                  on_change=lambda e, i=i: (S["loras"][i].__setitem__(0, e.value), ui.timer(0.05, draw_loras, once=True))) \
                            .props("outlined dense label='Pick a LoRA'").classes("grow")
                        sl = ui.slider(min=lo, max=hi, step=0.01, value=min(max(row[1], lo), hi),
                                       on_change=lambda e, i=i: S["loras"][i].__setitem__(1, e.value)) \
                            .classes("w-28").tooltip("Strength")
                        ui.number(format="%.2f", step=0.05, min=lo, max=hi).bind_value(sl, "value") \
                            .props("outlined dense").style("width:76px").tooltip(f"From {lo:g} to {hi:g}")
                        ui.button(icon="close", on_click=lambda i=i: (S["loras"].pop(i), draw_loras())) \
                            .props("flat dense round size=sm").tooltip("Remove")
                    info = files.lora_info(row[0]) if row[0] else None
                    if row[0] and row[0] not in ok_names:
                        why = next((w for n, w in hidden if n == row[0]), "It is not in the LoRA list.")
                        ui.label(f"{row[0]} can't be used here. {why}").classes("hint").style("color:var(--bad)")
                    elif info:
                        if info.trigger:  # a trigger word from training: one click puts it in the prompt
                            with ui.row().classes("items-center gap-2"):
                                ui.chip(info.trigger, icon="add", on_click=lambda t=info.trigger: add_trigger(t)) \
                                    .props("clickable dense outline").tooltip("Add the trigger word to the prompt")
                                ui.label(f"The trigger word \"{info.trigger}\" is not in your prompt") \
                                    .classes("hint").style("color:var(--work)") \
                                    .bind_visibility_from(S, "prompt", backward=lambda p, t=info.trigger: t.lower() not in (p or "").lower())
                        elif info.tags:  # no sidecar (downloaded LoRA): the tags it was trained on
                            with ui.row().classes("items-center gap-1"):
                                ui.label("Trained on:").classes("hint")
                                for t in info.tags[:4]:
                                    ui.chip(t, on_click=lambda t=t: add_trigger(t)).props("clickable dense outline")
                if hidden:
                    ui.label("Not shown for this model: " + ", ".join(n for n, _ in hidden[:6]) + ("…" if len(hidden) > 6 else "")) \
                        .classes("hint").tooltip("\n".join(f"{n}: {w}" for n, w in hidden))
                if not names:
                    ui.label("No LoRAs yet. Add some in Library, or train one.").classes("hint")
                elif not ok_names:
                    ui.label("No LoRA here fits this model.").classes("hint")
                elif any(not r[0] for r in S["loras"]):
                    ui.label("Rows without a LoRA picked are skipped.").classes("hint")
                button("Add a LoRA", lambda: (S["loras"].append([None, 0.8]), draw_loras()), icon="add") \
                    .props("size=sm")

        draw_loras()

        # ----- go -----
        with ui.row().classes("w-full no-wrap gap-2 items-center"):
            gen_b = main_button("Generate", lambda: go())
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
                    {"official": "Official", "abliterated": "Uncensored (follows more prompts)",
                     "unsloth": "Fast 4-bit (unsloth, being tested)"},
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
        "area": "Change the painted part: paint over what to change. Your prompt describes the new content.",
        "whole": "Change the whole image: your prompt guides it. Low strength keeps the photo, high reimagines it.",
        "extend": "Extend: add space around the image and fill it in. Say how much to add on each side.",
        "upscale": "Upscale: makes the image bigger and adds detail in tiles. The prompt is optional guidance.",
    }
    code = app.storage.browser.get("code")
    painter = {"el": None, "src": None, "empty": True, "device": False}
    pending = {"result": None, "same_aspect": False}  # the edit result being reviewed (Keep / Try again / Undo)
    shown = {"img": None, "tool": "x", "cmp": None, "need": None, "recent": None}

    def in_tools(*tools):
        return lambda v: v in tools

    def painter_set(**props):
        if painter["el"] is not None:
            painter["el"].set(**props)

    def painter_do(action):
        if painter["el"] is not None:
            getattr(painter["el"], action)()

    def set_extend(i, v):
        e = list(S["extend"])
        e[i] = max(0.0, min(100.0, float(v or 0)))
        S["extend"] = e

    def extend_to(ratio):
        """Percent per side so the image reaches this width/height ratio (split evenly)."""
        m = files.meta(S["src"]) if S["src"] else None
        if not m or not m.get("size"):
            ui.notify("Pick the image first", type="warning")
            return
        w, h = m["size"]
        if ratio >= w / h:
            side = (h * ratio / w - 1) * 50
            S["extend"] = [0, round(side, 1), 0, round(side, 1)]
        else:
            side = (w / ratio / h - 1) * 50
            S["extend"] = [round(side, 1), 0, round(side, 1), 0]
        extend_row.clear()
        draw_extend()

    with right:
        toolbar = ui.row().classes("w-full items-center gap-2 no-wrap")
        banner = ui.column().classes("banner w-full gap-2")
        with banner:
            tool_l = ui.label().classes("t-small")
            seg({"area": "Painted part", "whole": "Whole image", "extend": "Extend", "upscale": "Upscale"}, S, "tool",
                on_change=lambda e: on_mode())
            # painted part: tools, then sliders
            with ui.row().classes("w-full items-center gap-2") as area_tools:
                seg({"move": "Move", "brush": "Brush", "eraser": "Eraser", "box": "Box"}, S, "ptool",
                    on_change=lambda e: painter_set(tool=e.value))
                button("Undo", lambda: painter_do("undo"), icon="undo")
                button("Invert", lambda: painter_do("invert"), icon="flip")
                button("Clear", lambda: painter_do("clear"), icon="backspace")
            area_tools.bind_visibility_from(S, "tool", backward=in_tools("area"))
            with ui.row().classes("w-full items-center no-wrap gap-3") as area_sliders:
                for key, label, lo, hi, step in (("brush", "Brush", 8, 200, 4), ("feather", "Feather", 0, 40, 1)):
                    with ui.row().classes("grow items-center no-wrap gap-2"):
                        ui.label().bind_text_from(S, key, lambda v, label=label: f"{label} {v}").classes("t-small tnum") \
                            .style("white-space:nowrap")
                        ui.slider(min=lo, max=hi, step=step,
                                  on_change=(lambda e: painter_set(size=int(e.value))) if key == "brush" else None) \
                            .bind_value(S, key).classes("grow")
            area_sliders.bind_visibility_from(S, "tool", backward=in_tools("area"))
            with ui.row().classes("w-full items-center no-wrap gap-3"):
                for key, tip, lo, hi in (("strength", "Low keeps more of the original; 1 repaints the part fully", 0.1, 1.0),
                                         ("whole_strength", "0.3 keeps the photo, 0.8 reimagines it", 0.1, 1.0),
                                         ("up_strength", "0.2 to 0.35 adds detail without changing the image", 0.1, 0.6)):
                    with ui.row().classes("grow items-center no-wrap gap-3") as row:
                        ui.label().bind_text_from(S, key, lambda v: f"Strength {v:.2f}").classes("t-small tnum") \
                            .style("white-space:nowrap").tooltip(tip)
                        ui.slider(min=lo, max=hi, step=0.05).bind_value(S, key).classes("grow")
                    row.bind_visibility_from(S, "tool", backward=in_tools({"strength": "area", "whole_strength": "whole",
                                                                           "up_strength": "upscale"}[key]))
            with ui.row().classes("items-center no-wrap gap-2") as whole_row:
                ui.switch("Keep the original size").bind_value(S, "keep_size") \
                    .tooltip("The result is made at about 1 MP, then scaled back to the photo's size")
            whole_row.bind_visibility_from(S, "tool", backward=in_tools("whole"))
            with ui.row().classes("items-center no-wrap gap-2") as blend_row:
                ui.switch("Better blend (downloads 2 GB once)").bind_value(S, "better_blend")                     .bind_visibility_from(S, "kind", backward=lambda k: k == "Z-Image")                     .tooltip("Z-Image only: smoother joins where the new part meets the photo. A bit slower")
            blend_row.bind_visibility_from(S, "tool", backward=in_tools("area", "extend"))
            with ui.row().classes("items-center no-wrap gap-2") as tile_row:
                ui.switch("Tile detail model (downloads 2 GB once)").bind_value(S, "tile_cn")                     .bind_visibility_from(S, "kind", backward=lambda k: k == "Z-Image")                     .tooltip("Z-Image only: adds detail that follows the photo more closely. A bit slower")
            tile_row.bind_visibility_from(S, "tool", backward=in_tools("upscale"))
            with ui.column().classes("w-full gap-1") as extend_box:
                extend_row = ui.row().classes("w-full items-end no-wrap gap-2")

                def draw_extend():
                    with extend_row:
                        for i, name in enumerate(("Top %", "Right %", "Bottom %", "Left %")):
                            ui.number(name, value=S["extend"][i], min=0, max=100, step=5, format="%.0f",
                                      on_change=lambda e, i=i: set_extend(i, e.value)).props("outlined dense").classes("grow")

                draw_extend()
                with ui.row().classes("items-center gap-2"):
                    ui.label("Reach").classes("t-small")
                    button("Square", lambda: extend_to(1.0))
                    button("4:5", lambda: extend_to(4 / 5))
                    button("3:2", lambda: extend_to(3 / 2))
                    button("16:9", lambda: extend_to(16 / 9))
                    button("Reset", lambda: (S.__setitem__("extend", [0, 0, 0, 0]), extend_row.clear(), draw_extend()))
            extend_box.bind_visibility_from(S, "tool", backward=in_tools("extend"))
            with ui.row().classes("items-center no-wrap gap-1") as scale_row:
                ui.label("Size").classes("t-small")
                seg({1.5: "1.5x", 2.0: "2x"}, S, "up_scale")
            scale_row.bind_visibility_from(S, "tool", backward=in_tools("upscale"))
            with ui.row().classes("items-center gap-2") as var_row:
                ui.label("Variations").classes("t-small")
                seg({1: "1", 2: "2", 3: "3", 4: "4"}, S, "edit_count")
            var_row.bind_visibility_from(S, "tool", backward=in_tools("area", "whole", "extend"))
            button("Done", lambda: set_tool(None))
        canvas = ui.element("div").classes("canvas w-full").style("flex-direction:column")
        with canvas:
            view_box = ui.element("div").classes("w-full").style("display:flex; justify-content:center; align-items:center")
            painter_box = ui.element("div").classes("w-full")
            with ui.column().classes("overlay gap-1") as ov:
                overlay = {"box": ov, "preview": ui.image().classes("w-full").style("max-height:300px; object-fit:contain"),
                           "title": ui.label().classes("work"), "bar": ui.html("", sanitize=False).classes("w-full"),
                           "detail": ui.label().classes("hint tnum")}
            ov.set_visibility(False)
            overlay["preview"].set_visibility(False)
        with ui.row().classes("w-full items-center gap-2") as result_bar:
            ui.label("Keep this result?").classes("t-small")
            button("Keep", lambda: keep(), icon="check")
            button("Try again", lambda: try_again(), icon="refresh")
            button("Undo", lambda: undo_result(), icon="undo")
            ui.switch("Before / after").bind_value(S, "compare_on")
        with ui.row().classes("w-full items-center gap-2"):
            earlier = ui.label("").classes("hint")
            ui.space()

            async def on_upload(e):
                up.reset()  # max_files=1: clear the finished file so the next upload is accepted
                try:
                    p = await run.io_bound(save_upload, await e.file.read(), e.file.name)  # decoding a big photo must not block the page
                except ValueError as ex:
                    ui.notify(str(ex), type="warning")
                    return
                except Exception as ex:
                    ui.notify(err(ex), type="negative")
                    return
                S.update(image=p, src=p if S["tool"] else None, mask_id=None)
                draw_canvas()
                ui.notify("Image added. Pick what to do with it above the image", type="positive")

            up = ui.upload(auto_upload=True, on_upload=on_upload, max_files=1, max_file_size=MAX_UPLOAD,
                           on_rejected=lambda: ui.notify("That file is over 50 MB. Export it smaller first.", type="warning")) \
                .props("accept=image/*,.heic,.heif hide-upload-btn").style("display:none")
            up_btn = button("Upload", lambda: up.run_method("pickFiles"), icon="upload").tooltip("Upload a photo to edit")
        with ui.row().classes("w-full items-center gap-2") as actions:
            reuse_b = button("Reuse", lambda: reuse_shown(), icon="settings_backup_restore").tooltip("Fill the form with the settings of this image")
            area_b = button("Change a part", lambda: set_tool("area"), icon="brush")
            whole_b = button("Change all", lambda: set_tool("whole"), icon="auto_fix_high").tooltip("Change the whole image with your prompt")
            extend_b = button("Extend", lambda: set_tool("extend"), icon="aspect_ratio")
            up_b = button("Upscale", lambda: set_tool("upscale"), icon="open_in_full")
            ui.space()
            dl_b = button("", lambda: ui.download.file(S["image"]), icon="download").tooltip("Download the PNG")
            clear_b = button("", lambda: (S.update(image=None, tool=None, mask_id=None), draw_canvas()), icon="close") \
                .tooltip("Empty the image area. The image stays in History")
        recent = ui.row().classes("w-full gap-2")
        up_btn.move(toolbar)
        actions.move(toolbar)
        actions.classes(remove="w-full", add="grow")

    def reviewing() -> bool:
        return bool(pending["result"]) and pending["result"] == S["image"]

    def shown_img():
        return S["image"] if (reviewing() or not S["tool"]) else S["src"]

    def compare_key():
        return (pending["result"], S["compare_on"]) if reviewing() else None

    def mask_preload(src) -> str | None:
        """'Edit again': the mask saved next to an area result, if this image has one."""
        m = files.meta(src)
        p = Path(str(src)[:-4] + "_mask.png")
        return url(p) if m and m.get("tool") == "area" and p.exists() else None

    def on_painter_changed(args):
        painter["empty"] = bool(args.get("empty", True))

    def on_device(args):
        if args.get("coarse") and not painter["device"]:  # touch screens start in Move, so a finger can scroll the page
            painter["device"] = True
            if S["ptool"] == "brush":
                S["ptool"] = "move"
                painter_set(tool="move")

    def draw_canvas():
        img = shown_img()
        if S["tool"] == "area" and shown["tool"] != "area":
            area_default()
        shown.update(img=img, tool=S["tool"], cmp=compare_key(), need=None)
        src = S["src"] if S["tool"] else None
        if src != painter["src"]:  # a new source image: a fresh painter and an empty mask
            painter_box.clear()
            painter.update(el=None, src=None, empty=True)
            S["mask_id"] = None
            if src and Path(src).exists():
                u = files.display_url(src)  # never the original: it may be 24 MP
                if u is None:
                    shown["need"] = src
                else:
                    with painter_box:
                        painter["el"] = MaskPainter(u, S["ptool"], int(S["brush"]),
                                                    None if painter.get("fresh") == src else mask_preload(src),
                                                    on_changed=on_painter_changed, on_device=on_device)
                    painter["src"] = src
        painter_box.set_visibility(S["tool"] == "area" and not reviewing() and painter["el"] is not None)
        view_box.clear()
        with view_box:
            if not img or not Path(img).exists():
                with ui.column().classes("items-center gap-1"):
                    ui.label("Your image shows up here").classes("muted")
                    ui.label("Write a prompt and press Generate, or upload an image to change or upscale it.") \
                        .classes("hint")
            elif S["tool"] == "area" and not reviewing():
                if painter["el"] is None:
                    ui.label("Preparing the image…").classes("muted")
            elif reviewing() and S["compare_on"] and pending["same_aspect"] and S["src"] and files.display_url(S["src"]):
                BeforeAfter(files.display_url(S["src"]), files.view_url(img) or url(img))
            else:
                u = files.view_url(img)
                if u is None:
                    shown["need"] = shown["need"] or img
                ui.image(u or url(img)).classes("w-full").props("fit=contain no-transition")

    def keep():
        painter["fresh"] = S["image"]  # Keep starts from an empty mask; only picking an old image reloads its mask
        S.update(src=S["image"], mask_id=None)
        pending["result"] = None
        draw_canvas()

    def undo_result():
        if S["src"]:
            S["image"] = S["src"]
        pending["result"] = None
        draw_canvas()

    async def try_again():
        S["seed"] = -1  # a new seed
        await go()

    async def go():
        """Generate; for a painted edit the mask goes to the app first."""
        if busy_task():
            return
        if S["tool"] == "area":
            if painter["el"] is None or painter["empty"]:
                ui.notify("Paint the part to change first", type="warning")
                return
            try:
                S["mask_id"] = await painter["el"].export(code)
            except ValueError as ex:
                ui.notify(str(ex), type="warning")
                return
        await do_generate(S, on_done)

    def on_done(paths, tool):
        files.poke()
        if not tool or tool == "upscale":
            pending["result"] = None
            return
        try:
            from PIL import Image
            with Image.open(paths[0]) as a, Image.open(S["src"]) as b:  # sizes only: a header read
                same = abs(a.width / a.height - b.width / b.height) < 0.01
        except Exception:
            same = False
        pending.update(result=paths[0], same_aspect=same)

    def draw_recent():
        items = files.items(12)
        sig = [(str(p), files.thumb_ready(p, 160)) for p, _ in items] + [shown_img()]  # redraw when a thumb appears
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

    def on_mode():
        """A mode tab while a result is shown: the next edit works on that result (what the user sees)."""
        if reviewing():
            S["src"] = S["image"]
            pending["result"] = None
        draw_canvas()

    auto = {"strength": None}  # the strength last set automatically; anything else is the user's choice

    def area_default():
        """Painted-part strength: 0.85 Z-Image / 0.75 SDXL (the plan's defaults), unless the user moved the slider."""
        if auto["strength"] is None or S["strength"] == auto["strength"]:
            S["strength"] = auto["strength"] = 0.85 if S["kind"] == "Z-Image" else 0.75

    def on_kind():
        if S["tool"] == "area":
            area_default()

    def set_tool(tool):
        if tool and not S["image"]:
            ui.notify("Generate, pick or upload an image first", type="warning")
            return
        S.update(tool=tool, src=S["image"] if tool else None)
        pending["result"] = None
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
        if S["tool"] and not S["src"]:
            S["src"] = S["image"]
        sig = (S["kind"], tuple(lora_names()))  # the picker follows the model kind and new LoRAs
        if shown.get("loras") != sig:
            shown["loras"] = sig
            draw_loras()
        if (shown["img"] != shown_img() or shown["tool"] != S["tool"] or shown["cmp"] != compare_key()
                or (shown["need"] and files.display_ready(shown["need"]))):
            draw_canvas()
        result_bar.set_visibility(reviewing())
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
                src = f"/work/previews/{t.preview}?v={t.preview_n}" if t.preview else ""  # a built string: no disk access here
                if src and overlay["preview"].source != src:
                    overlay["preview"].set_source(src)
                overlay["preview"].set_visibility(bool(src))
        else:
            if gen_b.text != idle_label:
                gen_b.text = idle_label
            gen_b.enable()
            stop_b.set_visibility(False)
            if "box" in overlay:
                overlay["box"].set_visibility(False)
        th = {"area": "Paint over the part to change, then press the button. The prompt describes what should be there.",
              "whole": "Uses your prompt and the strength above the image.",
              "extend": "Fills the new space around the image. The prompt describes the whole scene.",
              "upscale": "Press Done above the image to go back to making new images."}.get(S["tool"], "")
        if tool_hint.text != th:
            tool_hint.text = th
        # image actions and hints
        banner.set_visibility(bool(S["tool"]))
        if S["tool"] and tool_l.text != TOOL_TEXT[S["tool"]]:
            tool_l.text = TOOL_TEXT[S["tool"]]
        has = bool(S["image"])
        actions.set_visibility(has)
        for b in (area_b, whole_b, extend_b, up_b):
            b.set_visibility(not S["tool"])
        meta_p = files.meta(S["image"]) if has else None
        uploaded = bool(meta_p and meta_p.get("uploaded"))
        reuse_b.set_visibility(has and not S["tool"] and not uploaded)
        latest = files.items(1)
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
            await go()

    prompt.on("keydown", on_key, args=["key", "ctrlKey", "metaKey"])
    draw_canvas()
    tick()
    ui.timer(0.5, tick)
