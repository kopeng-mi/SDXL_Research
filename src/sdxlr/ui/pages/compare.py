"""Compare: LoRAs (rows) x strengths (columns) on the same seed, as one grid image."""

from __future__ import annotations

from nicegui import ui

from ..core import WORK, err, lora_names, model_names, submit_compare, url, z_names
from ..theme import JobLine, labeled, main_button, seg, title


def page(S):
    def kind_defaults(v):
        z = v == "Z-Image"
        if (S["c_steps"], float(S["c_cfg"])) in ((20, 5.0), (9, 0.0)):
            S.update(c_steps=9 if z else 20, c_cfg=0.0 if z else 5.0)

    with ui.row().classes("w-full gap-8 no-wrap items-start two-col"):
        with ui.column().classes("gap-4").style("width:440px; min-width:360px"):
            title("Compare")
            ui.label("Each LoRA or mix becomes a row, each strength a column, all with the same seed.").classes("hint")
            with ui.row().classes("w-full items-end no-wrap gap-2"):
                with ui.column().classes("gap-0"):
                    ui.label("Model").classes("lbl")
                    seg(["SDXL", "Z-Image"], S, "c_kind", on_change=lambda e: kind_defaults(e.value))
                labeled("Checkpoint", ui.select(model_names()).bind_value(S, "c_base")) \
                    .bind_visibility_from(S, "c_kind", value="SDXL")
                labeled("Z-Image model", ui.select(z_names(), new_value_mode="add-unique").bind_value(S, "c_z")) \
                    .bind_visibility_from(S, "c_kind", value="Z-Image")
            labeled("LoRAs", ui.select(lora_names(), multiple=True, with_input=True).bind_value(S, "c_loras")
                    .props("use-chips"), "Each one at full strength, scaled by the columns")
            labeled("Mixes (optional, replace the LoRAs above)",
                    ui.input(placeholder="face_a.safetensors:0.44+face_b.safetensors:0.44").bind_value(S, "c_combos"),
                    "One mix per row: name:weight joined with +, rows separated by commas")
            labeled("Strengths", ui.input().bind_value(S, "c_str"), "Numbers separated by spaces, like 0.6 0.8 1.0")
            labeled("Prompt", ui.textarea().bind_value(S, "c_prompt").props("autogrow"))
            labeled("Grid name", ui.input().bind_value(S, "c_name"), "The same name shows its last grid on the right")
            with ui.expansion("Steps, CFG, seed and negative").classes("w-full"):
                with ui.row().classes("w-full no-wrap gap-2"):
                    labeled("Steps", ui.number(step=1, format="%d").bind_value(S, "c_steps"))
                    labeled("CFG", ui.number(step=0.5).bind_value(S, "c_cfg"))
                    labeled("Seed", ui.number(step=1, format="%d").bind_value(S, "c_seed"))
                labeled("Negative prompt (SDXL)", ui.input().bind_value(S, "c_neg")).bind_visibility_from(S, "c_kind", value="SDXL")

            def go():
                try:
                    line.track(submit_compare(S))
                except ValueError as e:
                    ui.notify(err(e), type="warning")

            main_button("Make the grid", go)
            line = JobLine(("compare",))
        box = ui.element("div").classes("canvas").style("flex:1 1 0; min-width:0")

    seen = {"m": "x"}

    def tick():
        p = WORK / "compare" / (S["c_name"].strip() or "grid") / "grid.png"
        m = p.stat().st_mtime if p.exists() else None
        if m != seen["m"]:
            seen["m"] = m
            box.clear()
            with box:
                if m:
                    ui.image(url(p)).classes("w-full").props("fit=contain no-transition no-spinner")
                else:
                    with ui.column().classes("items-center gap-1"):
                        ui.label(f"No grid named \"{S['c_name']}\" yet").classes("muted")
                        ui.label("Pick LoRAs and press Make the grid.").classes("hint")

    tick()
    ui.timer(2.0, tick)
