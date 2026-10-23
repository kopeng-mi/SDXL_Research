"""Small Gradio UI over the engine. Runs inside the main venv.

  python -m sdxlr.app [--share] [--port 7860]

One GPU job at a time (queue concurrency 1). "Unload" frees all VRAM, e.g.
before training. Presets (prompt + LoRA mix + settings) are YAML files.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
import yaml
from PIL import Image, PngImagePlugin

from .engine import SAMPLERS, Engine, Job, ram, vram
from .fetch import LORAS, MODELS, WORK, fetch, fetch_loras, list_loras, list_models

N_LORAS = 6
PRESETS = Path(os.environ.get("SDXLR_PRESETS", WORK / "presets"))
OUTPUTS = WORK / "outputs"
NEG_DEFAULT = ("(worst quality:1.3), (low quality:1.2), lowres, bad anatomy, bad hands, watermark, "
               "signature, blur, cartoon, anime, drawing")

eng = Engine()


def _models() -> list[str]:
    return [str(p) for p in list_models()]


def _loras() -> list[str]:
    return ["None"] + [str(p) for p in list_loras()]


def _presets() -> list[str]:
    return sorted(p.stem for p in PRESETS.glob("*.yaml"))


def _status(msg: str = "") -> str:
    model = Path(eng.model_path).name if eng.model_path else "no model"
    return f"{msg}  \n**{model}** | {vram()} | {ram()} | {len(eng.adapters)} LoRAs cached".strip()


def load_model(path, vae):
    if not path:
        return _status("pick a model first")
    return _status(eng.load(path, vae))


def unload():
    eng.unload()
    return _status("unloaded, VRAM free")


def do_fetch(src, kind):
    src = (src or "").strip()
    if not src:
        return gr.update(), gr.update(), _status("paste org/repo/file, a URL or a path")
    try:
        if kind == "model":
            msg = f"model: {fetch(src, MODELS).name}"
        else:
            msg = f"LoRAs: {', '.join(p.name for p in fetch_loras(src))}"
    except Exception as e:
        msg = f"fetch failed: {type(e).__name__}: {e}"
    return gr.update(choices=_models()), gr.update(), _status(msg)


def refresh():
    loras = _loras()
    return [gr.update(choices=_models()), gr.update(choices=_presets())] + [gr.update(choices=loras)] * N_LORAS


def _mask_from_editor(ed) -> tuple[Image.Image | None, Image.Image | None]:
    if not ed or ed.get("background") is None:
        return None, None
    bg = ed["background"]
    m = np.zeros((bg.height, bg.width), dtype=np.uint8)
    for layer in ed.get("layers") or []:
        a = np.asarray(layer.convert("RGBA"))[..., 3]
        m = np.maximum(m, (a > 0).astype(np.uint8) * 255)
    return bg.convert("RGB"), Image.fromarray(m)


def _save(images, meta: dict) -> list[str]:
    day = OUTPUTS / time.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    info = PngImagePlugin.PngInfo()
    info.add_text("sdxlr", json.dumps(meta))
    paths = []
    for i, im in enumerate(images):
        p = day / f"{time.strftime('%H%M%S')}_{meta['seed'] + i}.png"
        im.save(p, pnginfo=info)
        paths.append(str(p))
    return paths


def generate(mode, prompt, negative, steps, cfg, sampler, width, height, seed, count, clip_skip,
             editor, strength, *lora_args):
    if eng.pipe is None:
        raise gr.Error("Load a model first")
    loras = [(p, float(w)) for p, w in zip(lora_args[::2], lora_args[1::2]) if p and p != "None" and w]
    image = mask = None
    if mode == "Inpaint":
        image, mask = _mask_from_editor(editor)
        if image is None or not mask.getbbox():
            raise gr.Error("Inpaint: upload an image and paint the area to change")
    job = Job(prompt=prompt, negative=negative, steps=int(steps), cfg=float(cfg), sampler=sampler,
              width=int(width), height=int(height), seed=int(seed), count=int(count),
              clip_skip=int(clip_skip), loras=loras, image=image, mask=mask, strength=float(strength))
    try:
        images, used_seed, info = eng.generate(job)
    except Exception as e:  # keep the app alive; show the reason
        raise gr.Error(f"{type(e).__name__}: {e}")
    meta = {"prompt": prompt, "negative": negative, "steps": job.steps, "cfg": job.cfg, "sampler": sampler,
            "size": [job.width, job.height], "seed": used_seed, "clip_skip": job.clip_skip,
            "loras": [[Path(p).name, w] for p, w in loras], "model": Path(eng.model_path).name, "mode": mode}
    return _save(images, meta), _status(info)


PRESET_KEYS = ["prompt", "negative", "steps", "cfg", "sampler", "width", "height", "seed", "count", "clip_skip"]


def save_preset(name, *vals):
    name = (name or "").strip()
    if not name:
        return gr.update(), _status("preset needs a name")
    base, lora_args = vals[:len(PRESET_KEYS)], vals[len(PRESET_KEYS):]
    data = dict(zip(PRESET_KEYS, base))
    data["loras"] = [[Path(p).name, float(w)] for p, w in zip(lora_args[::2], lora_args[1::2]) if p and p != "None"]
    PRESETS.mkdir(parents=True, exist_ok=True)
    (PRESETS / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return gr.update(choices=_presets(), value=name), _status(f"saved preset {name}")


def load_preset(name):
    f = PRESETS / f"{name}.yaml"
    if not name or not f.exists():
        return [gr.update()] * (len(PRESET_KEYS) + 2 * N_LORAS) + [_status("no such preset")]
    data = yaml.safe_load(f.read_text())
    by_name = {Path(p).name: p for p in _loras()}
    loras = (data.get("loras") or [])[:N_LORAS]
    out = [data.get(k, gr.update()) for k in PRESET_KEYS]
    missing = [n for n, _ in loras if n not in by_name]
    for i in range(N_LORAS):
        if i < len(loras):
            out += [by_name.get(loras[i][0], "None"), loras[i][1]]
        else:
            out += ["None", 0.0]
    msg = f"loaded preset {name}" + (f"; missing LoRAs: {', '.join(missing)}" if missing else "")
    return out + [_status(msg)]


def build() -> gr.Blocks:
    with gr.Blocks(title="SDXL_Research") as ui:
        with gr.Row():
            model = gr.Dropdown(_models(), label="Base model", scale=4)
            vae = gr.Radio(["fp16fix", "model"], value="fp16fix", label="VAE", scale=1)
            load_b = gr.Button("Load", variant="primary", scale=1)
            unload_b = gr.Button("Unload (free VRAM)", scale=1)
        with gr.Accordion("Download models / LoRAs", open=False):
            with gr.Row():
                src = gr.Textbox(label="org/repo/file, URL or path (.safetensors or .zip of LoRAs)", scale=4)
                kind = gr.Radio(["model", "loras"], value="loras", label="Type", scale=1)
                fetch_b = gr.Button("Fetch", scale=1)
        status = gr.Markdown(_status())

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="Prompt", lines=3, value="(ohwx_person:1.2), woman, ")
                negative = gr.Textbox(label="Negative", lines=2, value=NEG_DEFAULT)
                with gr.Tabs():
                    with gr.Tab("Generate") as tab_gen:
                        with gr.Row():
                            width = gr.Slider(512, 1536, 896, step=64, label="Width")
                            height = gr.Slider(512, 1536, 1152, step=64, label="Height")
                    with gr.Tab("Inpaint") as tab_inp:
                        editor = gr.ImageEditor(type="pil", label="Paint the area to change", layers=False,
                                                brush=gr.Brush(colors=["#ffffff"], color_mode="fixed"))
                        strength = gr.Slider(0.1, 1.0, 0.75, step=0.05, label="Denoise strength")
                mode = gr.State("Generate")
                tab_gen.select(lambda: "Generate", None, mode)
                tab_inp.select(lambda: "Inpaint", None, mode)
                with gr.Row():
                    steps = gr.Slider(4, 60, 25, step=1, label="Steps")
                    cfg = gr.Slider(1, 12, 5, step=0.5, label="CFG")
                    clip_skip = gr.Slider(1, 3, 2, step=1, label="Clip skip")
                with gr.Row():
                    sampler = gr.Dropdown(list(SAMPLERS), value="dpmpp_2m_karras", label="Sampler")
                    seed = gr.Number(-1, label="Seed (-1 random)", precision=0)
                    count = gr.Slider(1, 4, 1, step=1, label="Images")
                go = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Output", columns=2, height=640)
                with gr.Accordion("LoRAs", open=True):
                    lora_inputs = []
                    for i in range(N_LORAS):
                        with gr.Row():
                            lora_inputs.append(gr.Dropdown(_loras(), value="None", label=f"LoRA {i + 1}", scale=3))
                            lora_inputs.append(gr.Slider(-2, 2, 0.0, step=0.01, label="Weight", scale=2))
                with gr.Row():
                    preset_pick = gr.Dropdown(_presets(), label="Preset", scale=2)
                    preset_load = gr.Button("Load", scale=1)
                with gr.Row():
                    preset_name = gr.Textbox(label="Save as", scale=2)
                    preset_save = gr.Button("Save", scale=1)
                refresh_b = gr.Button("Refresh lists")

        base_inputs = [prompt, negative, steps, cfg, sampler, width, height, seed, count, clip_skip]
        load_b.click(load_model, [model, vae], status)
        unload_b.click(unload, None, status)
        fetch_b.click(do_fetch, [src, kind], [model, preset_pick, status])
        refresh_b.click(refresh, None, [model, preset_pick, *lora_inputs[::2]])
        go.click(generate, [mode, prompt, negative, steps, cfg, sampler, width, height, seed, count, clip_skip,
                            editor, strength, *lora_inputs], [gallery, status])
        preset_save.click(save_preset, [preset_name, *base_inputs, *lora_inputs], [preset_pick, status])
        preset_load.click(load_preset, preset_pick, [*base_inputs, *lora_inputs, status])
    return ui


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--model", help="load this model at start")
    a = ap.parse_args()
    for d in (MODELS, LORAS, PRESETS, OUTPUTS):
        d.mkdir(parents=True, exist_ok=True)
    if a.model:
        print(eng.load(str(fetch(a.model, MODELS))), flush=True)
    ui = build()
    ui.queue(default_concurrency_limit=1, max_size=8)  # one GPU job at a time
    ui.launch(server_name="0.0.0.0", server_port=a.port, share=a.share,
              allowed_paths=[str(OUTPUTS)], show_error=True)


if __name__ == "__main__":
    main()
