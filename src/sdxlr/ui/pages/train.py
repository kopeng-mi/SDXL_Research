"""Train: the form on the left; time estimate, job status and tips on the right."""

from __future__ import annotations

from nicegui import ui

from ..core import datasets, err, images, model_names, submit_train, train_estimate
from ..theme import JobLine, labeled, main_button, seg, title


def page(S):
    if not S["t_ds"] and datasets():
        S["t_ds"] = S["ds_pick"] if S["ds_pick"] in datasets() else datasets()[-1]
    kind_seen = {"v": S["t_kind"]}

    def kind_defaults(v):
        if v != kind_seen["v"]:
            kind_seen["v"] = v
            z = v == "Z-Image"
            S.update(t_epochs=10 if z else 15, t_repeats=4 if z else 6, t_dim=16 if z else 24, t_lr=1e-4 if z else 2e-4)

    with ui.row().classes("w-full gap-8 no-wrap items-start two-col"):
        with ui.column().classes("gap-4").style("width:500px; min-width:380px"):
            title("Train a LoRA")
            with ui.column().classes("gap-0"):
                ui.label("Model to train for").classes("lbl")
                seg(["SDXL", "Z-Image"], S, "t_kind", on_change=lambda e: kind_defaults(e.value))
            with ui.row().classes("w-full no-wrap gap-2"):
                ds = labeled("Dataset", ui.select(datasets()).bind_value(S, "t_ds"))
                name = labeled("LoRA name", ui.input(placeholder="my_face_v1").bind_value(S, "t_name"),
                               "Files are saved as name-000001.safetensors, one per epoch")
            labeled("Base checkpoint", ui.select(model_names(), clearable=True).bind_value(S, "t_base")
                    .props("label='SDXL 1.0 base'"), "Train on the checkpoint you will generate with") \
                .bind_visibility_from(S, "t_kind", value="SDXL")
            with ui.row().classes("w-full no-wrap gap-2"):
                labeled("Epochs", ui.number(min=1, step=1, format="%d").bind_value(S, "t_epochs"), "Passes over the photos")
                labeled("Repeats", ui.number(min=1, step=1, format="%d").bind_value(S, "t_repeats"), "Each photo per pass")
                labeled("Batch", ui.number(min=1, max=4, step=1, format="%d").bind_value(S, "t_batch"),
                        "Photos per step").bind_visibility_from(S, "t_kind", value="SDXL")
                labeled("Resolution", ui.select({512: "512 px", 768: "768 px (untested)"}).bind_value(S, "t_res")) \
                    .bind_visibility_from(S, "t_kind", value="Z-Image")
            with ui.expansion("More training options").classes("w-full"):
                with ui.row().classes("w-full no-wrap gap-2"):
                    labeled("Rank", ui.number(min=1, step=1, format="%d").bind_value(S, "t_dim"), "Bigger = more detail, bigger file")
                    labeled("Learning rate", ui.number(format="%.1e").bind_value(S, "t_lr"))
                with ui.row().classes("w-full no-wrap gap-2 items-end").bind_visibility_from(S, "t_kind", value="SDXL"):
                    labeled("Text encoder learning rate", ui.number(format="%.1e").bind_value(S, "t_te_lr"), "0 = off")
                    with ui.column().classes("gap-0"):
                        ui.label("Network").classes("lbl")
                        seg({"LoRA": "LoRA", "LoCon": "LoCon (more layers)"}, S, "t_type")
                labeled("Private HF repo for checkpoints", ui.input(placeholder="yourname/my-loras").bind_value(S, "t_hf"))
                ui.checkbox("Auto resume: keep the dataset and the latest checkpoint in that repo, "
                            "so the same LoRA name continues in a later session").bind_value(S, "t_resume")

            def start():
                try:
                    line.track(submit_train(S))
                    ui.notify("Training queued. The image model is unloaded while it runs")
                except ValueError as e:
                    ui.notify(err(e), type="warning")
                    if "name" in str(e):
                        name.run_method("focus")

            main_button("Start training", start)

        with ui.column().classes("gap-4").style("flex:1 1 0; min-width:0; padding-top:44px"):
            with ui.column().classes("panel w-full gap-1"):
                ui.label("How long").classes("lbl")
                eta_h = ui.label().classes("t-head tnum")
                eta_d = ui.label().classes("hint")
                ds_info = ui.label().classes("hint")
            with ui.column().classes("panel w-full gap-2"):
                ui.label("Progress").classes("lbl")
                line = JobLine(("train",))
                ui.label("Nothing training yet.").classes("hint").bind_visibility_from(line.box, "visible", backward=lambda v: not v)
            with ui.column().classes("panel w-full gap-1"):
                ui.label("Tips").classes("lbl")
                for tip in ("50 to 150 varied photos work well; write captions first in Dataset.",
                            "SDXL trains at 1024 px, about 5 s a step on a free T4.",
                            "Z-Image trains at 512 px on the base model (about 6 s a step) and is used on Turbo.",
                            "A LoRA is saved after every epoch. Compare them to pick the best one."):
                    ui.label(tip).classes("t-small")

    last = {"k": None}

    def tick():
        k = (S["t_kind"], S["t_ds"], S["t_repeats"], S["t_epochs"], S["t_batch"], S["t_res"])
        if k != last["k"]:
            last["k"] = k
            h, d = train_estimate(S)
            eta_h.text, eta_d.text = h, d
            imgs = images(S["t_ds"])
            caps = sum(p.with_suffix(".txt").exists() for p in imgs)
            ds_info.text = (f"{S['t_ds']}: {len(imgs)} photos, {caps} with captions"
                            + (". Write captions in Dataset first for better results" if imgs and caps < len(imgs) else "")) \
                if S["t_ds"] else ""
        if list(ds.options) != datasets():
            ds.set_options(datasets(), value=S["t_ds"] if S["t_ds"] in datasets() else None)

    tick()
    ui.timer(1.0, tick)
