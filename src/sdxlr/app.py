"""SDXL_Research UI (Gradio 6). Runs inside the main venv.

  python -m sdxlr.app [--port 7860] [--share] [--sync]

Tabs: Generate, Dataset, Train, Compare, Library. Generation runs in this
process; caption/train/compare/downloads run as queued child processes
(jobs.py). A GPU lock keeps generation and GPU jobs from overlapping, and the
image model is unloaded before a GPU job starts.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, PngImagePlugin

from . import presets, settings
from .engine import SAMPLERS, Z_ENCODERS, Z_MODELS, Engine, Job, ram, vram
from .fetch import LORAS, MODELS, WORK, ZIMAGE, add_file, fetch_loras, list_loras, list_models, list_zimage
from .jobs import JobManager, fmt_secs
from .modes import ASPECTS, MODES, PONY_NEG, PONY_TAGS, STYLES, train_eta

DATASETS = WORK / "datasets"
OUTPUTS = WORK / "outputs"
IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
N_LORAS = 6
NEG_SDXL = "(worst quality:1.3), (low quality:1.2), lowres, bad anatomy, bad hands, watermark, blur, cartoon, anime, drawing"

eng = Engine()
gpu_lock = threading.Lock()


def _before_gpu_job(job):
    gpu_lock.acquire()  # waits for a running generation to finish
    eng.unload()


def _after_gpu_job(job):
    gpu_lock.release()


jm = JobManager(on_gpu_job_start=_before_gpu_job, on_gpu_job_end=_after_gpu_job)


# ---------- helpers ----------
def _lora_choices():
    return ["None"] + [p.name for p in list_loras()]


def _lora_path(name):
    return next((str(p) for p in list_loras() if p.name == name), None)


def _model_names():
    return [p.name for p in list_models()]


def _model_path(name):
    return next((str(p) for p in list_models() if p.name == name), name)


def _z_names():
    return [*Z_MODELS, *(p.name for p in list_zimage())]


def _z_path(name):
    """Built-in preset, user checkpoint in the Library, or any link/path typed in."""
    return Z_MODELS.get(name) or next((str(p) for p in list_zimage() if p.name == name), name)


def _datasets():
    return sorted(p.name for p in DATASETS.iterdir() if p.is_dir()) if DATASETS.exists() else []


def _images(ds):
    d = DATASETS / ds if ds else None
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG) if d and d.exists() else []


def _header():
    model = "no model loaded"
    if eng.kind == "zimage":
        model = "Z-Image · " + Path(eng.model_path.split("|")[1]).name
    elif eng.kind == "sdxl":
        model = "SDXL · " + Path(eng.model_path).name
    running = [j for j in jm.recent(20) if j.status == "running"]
    queued = sum(j.status == "queued" for j in jm.jobs.values())
    job = running[0].summary() if running else "idle"
    return (f"**{model}** &nbsp;·&nbsp; {vram()} &nbsp;·&nbsp; {ram()} &nbsp;·&nbsp; Jobs: {job}"
            + (f" (+{queued} queued)" if queued else ""))


def _jobs_table():
    rows = [[j.id, j.kind, j.status, f"{j.done}/{j.total}" if j.total else "", fmt_secs(j.eta) if j.eta else "", j.title]
            for j in jm.recent(10)]
    return rows or [["", "", "", "", "", "no jobs yet"]]


# ---------- Generate ----------
def load_model(kind, sdxl_model, vae, z_model, z_te, z_quant):
    if jm.gpu_busy():
        raise gr.Error("A GPU job is running (see Jobs). Load the model after it finishes.")
    with gpu_lock:
        try:
            if kind == "Z-Image":
                msg = eng.load_zimage(_z_path(z_model), Z_ENCODERS.get(z_te, z_te), z_quant)
            else:
                if not sdxl_model:
                    raise gr.Error("Pick an SDXL checkpoint (download one in Library).")
                msg = eng.load(_model_path(sdxl_model), vae)
        except gr.Error:
            raise
        except Exception as e:
            raise gr.Error(f"Load failed: {type(e).__name__}: {e}")
    gr.Info(msg)
    return _header()


def unload_model():
    with gpu_lock:
        eng.unload()
    return _header()


def _mask(ed):
    if not ed or ed.get("background") is None:
        return None, None
    bg = ed["background"]
    m = np.zeros((bg.height, bg.width), dtype=np.uint8)
    for layer in ed.get("layers") or []:
        m = np.maximum(m, (np.asarray(layer.convert("RGBA"))[..., 3] > 0).astype(np.uint8) * 255)
    return bg.convert("RGB"), Image.fromarray(m)


def generate(mode, prompt, negative, style, aspect, count, seed, steps, cfg, sampler, clip_skip,
             use_adv, inpaint_on, editor, strength, *lora_args):
    if eng.pipe is None:
        raise gr.Error("Load a model first (top of this tab).")
    if not gpu_lock.acquire(blocking=False):
        raise gr.Error("The GPU is busy with a job (see Jobs). Try again when it finishes.")
    try:
        kind = eng.kind
        m = MODES[kind][mode]
        loras = [(_lora_path(n), float(w)) for n, w in zip(lora_args[::2], lora_args[1::2]) if n and n != "None" and w]
        loras = [(p, w) for p, w in loras if p]
        if not use_adv:
            steps, cfg, sampler = m["steps"], m["cfg"], m.get("sampler", sampler)
        if kind == "sdxl" and m.get("lora"):
            from .fetch import fetch
            loras.append((str(fetch(m["lora"][0], LORAS)), m["lora"][1]))
        full = ", ".join(x for x in (prompt.strip(), STYLES.get(style, "")) if x)
        neg = negative
        if kind == "sdxl" and "pony" in (eng.model_path or "").lower():
            full, neg = f"{PONY_TAGS}, {full}", f"{PONY_NEG}, {negative}"
        w, h = ASPECTS[aspect]
        image = msk = None
        if inpaint_on:
            image, msk = _mask(editor)
            if image is None or not msk.getbbox():
                raise gr.Error("Inpaint: add an image and paint over the area to change.")
        job = Job(prompt=full, negative=neg, steps=int(steps), cfg=float(cfg), sampler=sampler, width=w, height=h,
                  seed=int(seed), count=int(count), clip_skip=int(clip_skip), loras=loras,
                  image=image, mask=msk, strength=float(strength))
        try:
            images, used_seed, info = eng.generate(job)
        except Exception as e:
            raise gr.Error(f"{type(e).__name__}: {e}")
    finally:
        gpu_lock.release()
    meta = {"prompt": full, "negative": neg, "mode": mode, "steps": job.steps, "cfg": job.cfg, "sampler": sampler,
            "size": [w, h], "seed": used_seed, "loras": [[Path(p).name, x] for p, x in loras], "model": eng.model_path}
    day = OUTPUTS / time.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    png = PngImagePlugin.PngInfo()
    png.add_text("sdxlr", json.dumps(meta))
    paths = []
    for i, im in enumerate(images):
        p = day / f"{time.strftime('%H%M%S')}_{used_seed + i}.png"
        im.save(p, pnginfo=png)
        paths.append(str(p))
    return paths, info, _header()


PRESET_FIELDS = ["mode", "prompt", "negative", "style", "aspect", "count", "seed", "steps", "cfg", "sampler", "clip_skip"]


def save_preset(name, kind, *vals):
    name = (name or "").strip()
    if not name:
        raise gr.Error("Give the preset a name.")
    base, lora_args = vals[:len(PRESET_FIELDS)], vals[len(PRESET_FIELDS):]
    data = dict(zip(PRESET_FIELDS, base))
    data["kind"] = kind
    data["loras"] = [[n, float(w)] for n, w in zip(lora_args[::2], lora_args[1::2]) if n and n != "None"]
    presets.save(name, data)
    gr.Info(f"Saved preset {name}")
    return gr.update(choices=presets.list_presets(), value=name)


def load_preset(name):
    if not name:
        raise gr.Error("Pick a preset.")
    d = presets.load(name)
    out = [d.get(k, gr.update()) for k in PRESET_FIELDS]
    loras = (d.get("loras") or [])[:N_LORAS]
    have = set(_lora_choices())
    missing = [n for n, _ in loras if n not in have]
    for i in range(N_LORAS):
        n, w = loras[i] if i < len(loras) else ("None", 0.8)
        out += [n if n in have else "None", w]
    if missing:
        gr.Warning(f"Missing LoRAs (download or pull them in Library): {', '.join(missing)}")
    return [d.get("kind", "Z-Image"), *out]


def delete_preset(name):
    if name:
        presets.delete(name)
    return gr.update(choices=presets.list_presets(), value=None)


# ---------- Dataset ----------
def import_dataset(name, files, link):
    name = (name or "").strip() or "dataset"
    srcs = [f if isinstance(f, str) else f.name for f in (files or [])]
    srcs += [s.strip() for s in (link or "").replace("\n", ",").split(",") if s.strip()]
    if not srcs:
        raise gr.Error("Drop a zip / images, or paste a link (URL, org/repo/file.zip, Drive path, or 'test').")
    jm.submit("import", f"import → {name}", ["-m", "sdxlr.data", "--dest", DATASETS / name, *srcs], gpu=False)
    gr.Info("Import started (see Jobs). Press Refresh when it is done.")
    return gr.update(choices=sorted({*_datasets(), name}), value=name)


def show_dataset(ds):
    imgs = _images(ds)
    caps = sum(p.with_suffix(".txt").exists() for p in imgs)
    info = f"**{len(imgs)}** images · **{caps}** captioned · **{len(imgs) - caps}** without caption" if ds else ""
    return [(str(p), p.stem) for p in imgs], info, None, "", ""


def pick_image(ds, evt: gr.SelectData):
    p = _images(ds)[evt.index]
    t = p.with_suffix(".txt")
    return str(p), t.read_text(encoding="utf-8") if t.exists() else "", str(p)


def save_caption(path, text):
    if not path:
        raise gr.Error("Click an image first.")
    Path(path).with_suffix(".txt").write_text(text.strip(), encoding="utf-8")
    gr.Info("Caption saved")


def run_caption(ds, model, trigger, cls, prompt, overwrite, preview):
    if not ds:
        raise gr.Error("Pick a dataset.")
    args = ["-m", "sdxlr.caption", "--images", DATASETS / ds, "--model", model, "--trigger", trigger,
            "--class-word", cls, "--limit", int(preview or 0)]
    if (prompt or "").strip():
        args += ["--prompt", prompt]
    if overwrite:
        args += ["--overwrite"]
    jm.submit("caption", f"caption {ds} ({model})", args, gpu=True)
    gr.Info("Captioning queued (see Jobs)")


# ---------- Train ----------
def train_estimate(kind, ds, repeats, epochs, batch, res):
    n = len(_images(ds))
    if not n:
        return "Pick a dataset to see the time estimate."
    k = "zimage" if kind == "Z-Image" else "sdxl"
    b = 1 if k == "zimage" else int(batch or 1)
    r = int(res) if k == "zimage" else 1024
    steps, secs, est = train_eta(k, n, int(repeats or 1), int(epochs or 1), b, r)
    warn = " · ⚠ longer than a free T4 session (~5 h): lower epochs or repeats" if secs > 4.5 * 3600 else ""
    return f"**{steps} steps** · about **{fmt_secs(secs)}**{' (estimate)' if est else ''} for {n} images{warn}"


def run_train(kind, ds, name, sdxl_base, repeats, epochs, batch, res, dim, lr, te_lr, lora_type, hf_repo, resume):
    if not ds or not _images(ds):
        raise gr.Error("Pick a dataset with images.")
    name = (name or "").strip() or ds
    if not any(p.with_suffix(".txt").exists() for p in _images(ds)):
        gr.Warning("No captions found: caption the dataset first for better results.")
    if kind == "Z-Image":
        args = ["-m", "sdxlr.train_zimage", "--name", name, "--images", DATASETS / ds, "--set",
                f"resolution={int(res)}", f"epochs={int(epochs)}", f"repeats={int(repeats)}",
                f"network_dim={int(dim)}", f"network_alpha={int(dim)}", f"lr={lr}"]
    else:
        base = _model_path(sdxl_base) if sdxl_base else "stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"
        args = ["-m", "sdxlr.train", "--name", name, "--images", DATASETS / ds, "--model", base,
                "--set", f"epochs={int(epochs)}", f"repeats={int(repeats)}", f"batch_size={int(batch)}",
                f"network_dim={int(dim)}", f"network_alpha={max(1, int(dim) * 2 // 3)}", f"unet_lr={lr}",
                f"text_encoder_lr={te_lr}", f"lora_type={lora_type}"]
    if (hf_repo or "").strip():
        args += ["--hf-repo", hf_repo.strip()]
    if resume:
        if not (hf_repo or "").strip():
            raise gr.Error("Resume needs the HF repo the earlier run uploaded to.")
        args += ["--resume"]
    jm.submit("train", f"train {kind} {name}", args, gpu=True)
    gr.Info("Training queued (see Jobs). The image model is unloaded while it runs.")


# ---------- Compare ----------
def run_compare(kind, sdxl_base, z_model, loras_sel, combos, strengths, prompt, negative, steps, cfg, seed, name):
    name = (name or "").strip() or "grid"
    args = ["-m", "sdxlr.compare", "--name", name, "--strengths", *strengths.split(),
            "--prompt", prompt, "--negative", negative, "--steps", int(steps), "--cfg", cfg, "--seed", int(seed)]
    if kind == "Z-Image":
        args += ["--zimage", "--model", _z_path(z_model)]
    else:
        if not sdxl_base:
            raise gr.Error("Pick an SDXL checkpoint.")
        args += ["--model", _model_path(sdxl_base)]
    mixes = [c.strip() for c in (combos or "").split(",") if c.strip()]
    if mixes:
        args += ["--combos", *mixes]
    elif loras_sel:
        args += ["--combos", *[f"{n}:1.0" for n in loras_sel]]
    else:
        raise gr.Error("Pick LoRAs or write mixes.")
    jm.submit("compare", f"compare {name}", args, gpu=True)
    gr.Info("Compare queued (see Jobs). Press Show grid when it is done.")


def show_grid(name):
    p = WORK / "compare" / ((name or "").strip() or "grid") / "grid.png"
    return str(p) if p.exists() else None


# ---------- Library ----------
LIB_KINDS = {"LoRA / zip of LoRAs": "loras", "SDXL checkpoint": "model", "Z-Image model": "zimage"}


def lib_download(link, kind):
    links = [s.strip() for s in (link or "").replace("\n", ",").split(",") if s.strip()]
    if not links:
        raise gr.Error("Paste a link: HF org/repo/file or huggingface.co link, Civitai, any https URL, or a Drive path.")
    jm.submit("download", f"download: {', '.join(Path(x).name for x in links)[:60]}",
              ["-m", "sdxlr.fetch", LIB_KINDS[kind], *links], gpu=False)
    gr.Info("Download started (see Jobs)")


def lib_upload(files, kind):
    if not files:
        raise gr.Error("Drop .safetensors / .gguf files (or a zip of LoRAs).")
    k, added = LIB_KINDS[kind], []
    for f in files:
        f = f if isinstance(f, str) else f.name
        if k == "loras":
            added += [p.name for p in fetch_loras(f)] if f.endswith(".zip") else [add_file(f, LORAS).name]
        else:
            added.append(add_file(f, MODELS if k == "model" else ZIMAGE).name)
    gr.Info(f"Added {', '.join(added)}")
    return lib_table(), None


def lib_table():
    rows = [["model", p.name, f"{p.stat().st_size / 1e9:.2f} GB"] for p in list_models()]
    rows += [["z-image", p.name, f"{p.stat().st_size / 1e9:.2f} GB"] for p in list_zimage()]
    rows += [["lora", p.name, f"{p.stat().st_size / 1e6:.0f} MB"] for p in list_loras()]
    return rows or [["", "nothing downloaded yet", ""]]


def lib_delete(name):
    for p in [*list_models(), *list_zimage(), *list_loras()]:
        if p.name == (name or "").strip():
            p.unlink()
            gr.Info(f"Deleted {p.name}")
            return lib_table()
    raise gr.Error(f"Not found: {name}")


def save_settings(sync_repo, datasets_repo):
    settings.set(sync_repo=sync_repo, datasets_repo=datasets_repo)
    gr.Info("Settings saved")


def sync(direction):
    repo = settings.get()["sync_repo"]
    if not repo:
        raise gr.Error("Set and save the sync repo first (e.g. yourname/sdxlr-sync).")
    try:
        msg = presets.push(repo) if direction == "push" else presets.pull(repo)
    except Exception as e:
        raise gr.Error(f"Sync failed: {type(e).__name__}: {e}")
    gr.Info(msg)
    return gr.update(choices=presets.list_presets())


# ---------- Jobs ----------
def job_log(job_id):
    j = jm.jobs.get((job_id or "").strip())
    return "\n".join(j.log) if j else ""


def cancel_job(job_id):
    jm.cancel((job_id or "").strip())
    return _jobs_table()


# ---------- layout ----------
THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate", radius_size="lg",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]).set(
    # plain text labels instead of filled badges
    block_title_background_fill="transparent", block_title_background_fill_dark="transparent",
    block_title_text_color="*neutral_600", block_title_text_color_dark="*neutral_300",
    block_label_background_fill="transparent", block_label_background_fill_dark="transparent",
    block_title_text_weight="500",
)
CSS = """
.gradio-container {max-width: 1400px !important; margin: auto;}
#header {padding: 8px 14px; border-radius: 12px; background: var(--block-background-fill);}
#gen-btn {min-height: 56px; font-size: 1.05rem;}
footer {display: none !important;}
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="SDXL_Research") as ui:
        header = gr.Markdown(_header(), elem_id="header")

        with gr.Tabs():
            # ===== Generate =====
            with gr.Tab("Generate"):
                with gr.Row(equal_height=True):
                    kind = gr.Radio(["Z-Image", "SDXL"], value="Z-Image", label="Model type", scale=1)
                    sdxl_model = gr.Dropdown(_model_names(), label="SDXL checkpoint", visible=False, scale=3)
                    vae = gr.Radio(["fp16fix", "model"], value="fp16fix", label="VAE", visible=False, scale=1)
                    z_model = gr.Dropdown(_z_names(), value="turbo Q8", allow_custom_value=True, label="Z-Image model", scale=2)
                    z_te = gr.Dropdown(list(Z_ENCODERS), value="official", allow_custom_value=True, label="Text encoder", scale=2)
                    z_quant = gr.Radio(["nf4", "int8"], value="nf4", label="Encoder size", scale=1)
                    with gr.Column(scale=1, min_width=110):
                        load_b = gr.Button("Load", variant="primary", size="sm")
                        unload_b = gr.Button("Unload", size="sm")
                kind.change(lambda k: [gr.update(visible=k == "SDXL")] * 2 + [gr.update(visible=k == "Z-Image")] * 3,
                            kind, [sdxl_model, vae, z_model, z_te, z_quant])

                with gr.Row():
                    with gr.Column(scale=5):
                        mode = gr.Radio(["Quality", "Speed", "Fast"], value="Speed", label="Mode",
                                        info="Quality: more steps · Speed: default · Fast: fewest steps")
                        prompt = gr.Textbox(label="Prompt", lines=3,
                                            placeholder="ohwx_person, a woman in a green silk saree, standing in a garden")
                        with gr.Row():
                            style = gr.Dropdown(list(STYLES), value="Photo", label="Style", scale=1)
                            aspect = gr.Dropdown(list(ASPECTS), value="Portrait 896x1152", label="Size", scale=2)
                            count = gr.Slider(1, 4, 1, step=1, label="Images", scale=1)
                            seed = gr.Number(-1, label="Seed (-1 = random)", precision=0, scale=1)
                        go = gr.Button("Generate", variant="primary", elem_id="gen-btn")
                        gen_info = gr.Markdown()
                        with gr.Accordion("Negative prompt (SDXL)", open=False):
                            negative = gr.Textbox(value=NEG_SDXL, lines=2, show_label=False)
                        with gr.Accordion("Inpaint", open=False):
                            inpaint_on = gr.Checkbox(False, label="Inpaint instead of a new image")
                            editor = gr.ImageEditor(type="pil", label="Image (paint the area to change)", layers=False,
                                                    brush=gr.Brush(colors=["#ffffff"], color_mode="fixed"))
                            strength = gr.Slider(0.1, 1.0, 0.75, step=0.05, label="Change strength")
                        with gr.Accordion("Advanced", open=False):
                            use_adv = gr.Checkbox(False, label="Use these instead of the mode")
                            with gr.Row():
                                steps = gr.Slider(4, 60, 20, step=1, label="Steps")
                                cfg = gr.Slider(0, 12, 5, step=0.5, label="CFG")
                                sampler = gr.Dropdown(list(SAMPLERS), value="dpmpp_2m_karras", label="Sampler (SDXL)")
                                clip_skip = gr.Slider(1, 3, 2, step=1, label="Clip skip (SDXL)")
                    with gr.Column(scale=4):
                        gallery = gr.Gallery(columns=2, height=560, show_label=False)
                        send_inp = gr.Button("Send selected to Inpaint", size="sm")
                        gr.Markdown("#### LoRAs")
                        lora_inputs = []
                        for i in range(N_LORAS):
                            with gr.Row():
                                lora_inputs.append(gr.Dropdown(_lora_choices(), value="None", show_label=False, scale=3))
                                lora_inputs.append(gr.Slider(-1, 2, 0.8, step=0.01, show_label=False, scale=2))
                        with gr.Row():
                            preset_pick = gr.Dropdown(presets.list_presets(), label="Preset", scale=3)
                            p_load = gr.Button("Load", size="sm", scale=1)
                            p_del = gr.Button("Delete", size="sm", scale=1)
                        with gr.Row():
                            preset_name = gr.Textbox(label="Save as", scale=3)
                            p_save = gr.Button("Save", size="sm", scale=1)

                sel = gr.State(None)

                def _sel(evt: gr.SelectData):
                    v = evt.value
                    return v["image"]["path"] if isinstance(v, dict) and v.get("image") else None

                gallery.select(_sel, None, sel)
                send_inp.click(lambda p: (gr.update(value={"background": p, "layers": [], "composite": p}) if p else gr.update(),
                                          gr.update(value=True)), sel, [editor, inpaint_on])
                load_b.click(load_model, [kind, sdxl_model, vae, z_model, z_te, z_quant], header)
                unload_b.click(unload_model, None, header)
                gen_inputs = [mode, prompt, negative, style, aspect, count, seed, steps, cfg, sampler, clip_skip,
                              use_adv, inpaint_on, editor, strength, *lora_inputs]
                go.click(generate, gen_inputs, [gallery, gen_info, header])
                preset_vals = [mode, prompt, negative, style, aspect, count, seed, steps, cfg, sampler, clip_skip]
                p_save.click(save_preset, [preset_name, kind, *preset_vals, *lora_inputs], preset_pick)
                p_load.click(load_preset, preset_pick, [kind, *preset_vals, *lora_inputs])
                p_del.click(delete_preset, preset_pick, preset_pick)

            # ===== Dataset =====
            with gr.Tab("Dataset"):
                with gr.Row():
                    with gr.Column(scale=2):
                        ds_name = gr.Textbox(label="Dataset name", value="my_photos")
                        ds_files = gr.File(label="Drop a zip or images (captions optional: same-name .txt)",
                                           file_count="multiple", file_types=[".zip", "image", ".txt"])
                        ds_link = gr.Textbox(label="…or paste links",
                                             placeholder="https://… · org/repo/file.zip · /content/drive/MyDrive/… · test")
                        ds_import = gr.Button("Import", variant="primary")
                        gr.Markdown("#### Captions")
                        cap_model = gr.Dropdown(["qwen", "qwen-2b", "joy"], value="qwen", label="Captioner",
                                                info="qwen = Qwen3-VL 4B (accurate) · qwen-2b (faster) · joy = JoyCaption (uncensored)")
                        with gr.Row():
                            cap_trigger = gr.Textbox("ohwx_person", label="Trigger word")
                            cap_class = gr.Textbox("woman", label="Class word")
                        cap_prompt = gr.Textbox(label="Custom instruction (blank = identity preset; {cls} = class word)", lines=2)
                        with gr.Row():
                            cap_preview = gr.Number(0, label="Only first N (0 = all)", precision=0)
                            cap_over = gr.Checkbox(False, label="Overwrite existing")
                        cap_go = gr.Button("Caption")
                    with gr.Column(scale=3):
                        with gr.Row():
                            ds_pick = gr.Dropdown(_datasets(), label="Dataset", scale=4)
                            ds_refresh = gr.Button("Refresh", size="sm", scale=1)
                        ds_info = gr.Markdown()
                        ds_gallery = gr.Gallery(columns=6, height=360, show_label=False, allow_preview=False)
                        with gr.Row():
                            cap_img = gr.Image(type="filepath", height=220, show_label=False, interactive=False, scale=1)
                            with gr.Column(scale=2):
                                cap_text = gr.Textbox(label="Caption (click an image)", lines=6)
                                cap_save = gr.Button("Save caption", size="sm")
                        cap_path = gr.State("")
                ds_out = [ds_gallery, ds_info, cap_img, cap_text, cap_path]
                ds_import.click(import_dataset, [ds_name, ds_files, ds_link], ds_pick)
                ds_pick.change(show_dataset, ds_pick, ds_out)
                ds_refresh.click(lambda d: gr.update(choices=_datasets(), value=d), ds_pick, ds_pick).then(
                    show_dataset, ds_pick, ds_out)
                ds_gallery.select(pick_image, ds_pick, [cap_img, cap_text, cap_path])
                cap_save.click(save_caption, [cap_path, cap_text])
                cap_go.click(run_caption, [ds_pick, cap_model, cap_trigger, cap_class, cap_prompt, cap_over, cap_preview])

            # ===== Train =====
            with gr.Tab("Train"):
                with gr.Row():
                    with gr.Column(scale=3):
                        t_kind = gr.Radio(["SDXL", "Z-Image"], value="SDXL", label="Train for")
                        with gr.Row():
                            t_ds = gr.Dropdown(_datasets(), label="Dataset", scale=3)
                            t_ds_refresh = gr.Button("↻", size="sm", scale=0, min_width=40)
                            t_name = gr.Textbox(label="LoRA name", scale=3)
                        t_base = gr.Dropdown(_model_names(), label="SDXL base checkpoint (blank = SDXL 1.0 base)")
                        with gr.Row():
                            t_epochs = gr.Number(15, label="Epochs", precision=0)
                            t_repeats = gr.Number(6, label="Repeats", precision=0)
                            t_batch = gr.Number(3, label="Batch (SDXL)", precision=0)
                            t_res = gr.Dropdown([512, 768], value=512, label="Resolution (Z-Image)")
                        t_eta = gr.Markdown("Pick a dataset to see the time estimate.")
                        with gr.Accordion("Advanced", open=False):
                            with gr.Row():
                                t_dim = gr.Number(24, label="Rank (dim)", precision=0)
                                t_lr = gr.Number(2e-4, label="Learning rate")
                                t_te_lr = gr.Number(2e-5, label="Text encoder LR (SDXL, 0 = off)")
                                t_type = gr.Radio(["LoRA", "LoCon"], value="LoRA", label="Type (SDXL)")
                            with gr.Row():
                                t_hf = gr.Textbox(label="Private HF repo for LoRAs + resume state", placeholder="yourname/my-loras")
                                t_resume = gr.Checkbox(False, label="Resume from that repo")
                        t_go = gr.Button("Start training", variant="primary")
                    with gr.Column(scale=2):
                        gr.Markdown("**Tips**\n- 50-150 varied photos, captioned in the Dataset tab\n"
                                    "- SDXL: 1024 px, ~5 s/step on T4\n"
                                    "- Z-Image: 512 px, ~6 s/step on T4; trained on base, used on Turbo\n"
                                    "- A LoRA is saved every epoch: compare them in Compare\n"
                                    "- Progress and logs: Jobs (bottom)")
                t_kind.change(lambda k: (gr.update(value=10 if k == "Z-Image" else 15), gr.update(value=4 if k == "Z-Image" else 6),
                                         gr.update(value=16 if k == "Z-Image" else 24), gr.update(value=1e-4 if k == "Z-Image" else 2e-4)),
                              t_kind, [t_epochs, t_repeats, t_dim, t_lr])
                eta_in = [t_kind, t_ds, t_repeats, t_epochs, t_batch, t_res]
                for c in eta_in:
                    c.change(train_estimate, eta_in, t_eta)
                t_ds_refresh.click(lambda: gr.update(choices=_datasets()), None, t_ds)
                t_go.click(run_train, [t_kind, t_ds, t_name, t_base, t_repeats, t_epochs, t_batch, t_res, t_dim, t_lr,
                                       t_te_lr, t_type, t_hf, t_resume])

            # ===== Compare =====
            with gr.Tab("Compare"):
                with gr.Row():
                    with gr.Column(scale=2):
                        c_kind = gr.Radio(["SDXL", "Z-Image"], value="SDXL", label="Model type")
                        c_base = gr.Dropdown(_model_names(), label="SDXL checkpoint")
                        c_z = gr.Dropdown(_z_names(), value="turbo Q8", allow_custom_value=True, label="Z-Image model")
                        c_loras = gr.CheckboxGroup([p.name for p in list_loras()], label="LoRAs (one row each)")
                        c_combos = gr.Textbox(label="…or mixes, comma-separated",
                                              placeholder="face_a.safetensors:0.44+face_b.safetensors:0.44")
                        c_str = gr.Textbox("0.6 0.8 1.0", label="Strengths (columns)")
                        c_prompt = gr.Textbox("ohwx_person, woman, portrait photo, natural light", label="Prompt", lines=2)
                        c_neg = gr.Textbox(NEG_SDXL, label="Negative (SDXL)", lines=1)
                        with gr.Row():
                            c_steps = gr.Number(20, label="Steps", precision=0)
                            c_cfg = gr.Number(5, label="CFG")
                            c_seed = gr.Number(42, label="Seed", precision=0)
                        c_name = gr.Textbox("grid1", label="Grid name")
                        with gr.Row():
                            c_refresh = gr.Button("↻ LoRAs", size="sm")
                            c_go = gr.Button("Run compare", variant="primary")
                    with gr.Column(scale=3):
                        c_show = gr.Button("Show grid", size="sm")
                        c_img = gr.Image(type="filepath", show_label=False, height=720)
                c_kind.change(lambda k: (gr.update(value=9 if k == "Z-Image" else 20), gr.update(value=0 if k == "Z-Image" else 5)),
                              c_kind, [c_steps, c_cfg])
                c_refresh.click(lambda: gr.update(choices=[p.name for p in list_loras()]), None, c_loras)
                c_go.click(run_compare, [c_kind, c_base, c_z, c_loras, c_combos, c_str, c_prompt, c_neg,
                                         c_steps, c_cfg, c_seed, c_name])
                c_show.click(show_grid, c_name, c_img)

            # ===== Library =====
            with gr.Tab("Library"):
                with gr.Row():
                    with gr.Column(scale=2):
                        l_link = gr.Textbox(label="Paste links (comma or one per line)", lines=3,
                                            placeholder="cyberdelia/CyberRealisticPony/CyberRealisticPony_V17.0_FP16.safetensors\n"
                                                        "https://civitai.com/api/download/models/12345")
                        l_kind = gr.Radio(list(LIB_KINDS), value="LoRA / zip of LoRAs", label="Type")
                        l_go = gr.Button("Download", variant="primary")
                        l_up = gr.File(label="…or upload files (.safetensors, .gguf, zip of LoRAs)", file_count="multiple",
                                       file_types=[".safetensors", ".gguf", ".ckpt", ".zip"])
                        gr.Markdown("#### Sync (private HF repo)")
                        st = settings.get()
                        s_sync = gr.Textbox(st["sync_repo"], label="Sync repo for presets + LoRAs", placeholder="yourname/sdxlr-sync")
                        s_ds = gr.Textbox(st["datasets_repo"], label="Datasets repo (optional)")
                        s_save = gr.Button("Save settings", size="sm")
                        with gr.Row():
                            s_push = gr.Button("Push ↑")
                            s_pull = gr.Button("Pull ↓")
                        gr.Markdown("<small>HF token: Colab Secrets → HF_TOKEN. Civitai: CIVITAI_TOKEN.</small>")
                    with gr.Column(scale=3):
                        l_refresh = gr.Button("Refresh", size="sm")
                        l_table = gr.Dataframe(lib_table(), headers=["type", "file", "size"], interactive=False)
                        with gr.Row():
                            l_del_name = gr.Textbox(label="File name to delete", scale=3)
                            l_del = gr.Button("Delete", variant="stop", size="sm", scale=1)
                l_go.click(lib_download, [l_link, l_kind])
                l_refresh.click(lib_table, None, l_table)
                l_del.click(lib_delete, l_del_name, l_table)
                s_save.click(save_settings, [s_sync, s_ds])
                s_push.click(lambda: sync("push"), None, preset_pick)
                s_pull.click(lambda: sync("pull"), None, preset_pick)

        # ===== Jobs (always visible) =====
        with gr.Accordion("Jobs", open=False):
            jobs_df = gr.Dataframe(_jobs_table(), headers=["id", "kind", "status", "progress", "ETA", "title"],
                                   interactive=False)
            with gr.Row():
                job_id = gr.Textbox(label="Job id (click a row)", scale=3)
                job_cancel = gr.Button("Cancel job", variant="stop", size="sm", scale=1)
            job_out = gr.Textbox(label="Log", lines=12, max_lines=12, autoscroll=True)

        def _pick_job(evt: gr.SelectData):
            return evt.row_value[0] if evt.row_value else ""

        jobs_df.select(_pick_job, None, job_id)
        job_cancel.click(cancel_job, job_id, jobs_df)
        list_outs = [*lora_inputs[::2], sdxl_model, t_base, c_base, z_model, c_z, c_loras, ds_pick, t_ds]

        def refresh_lists():
            loras, models, zs = _lora_choices(), _model_names(), _z_names()
            return ([gr.update(choices=loras)] * N_LORAS + [gr.update(choices=models)] * 3 + [gr.update(choices=zs)] * 2
                    + [gr.update(choices=[p.name for p in list_loras()])] + [gr.update(choices=_datasets())] * 2)

        l_up.upload(lib_upload, [l_up, l_kind], [l_table, l_up]).then(refresh_lists, None, list_outs)
        seen_done: set[str] = set()

        def tick(jid, ds):
            # refresh whatever a just-finished job changed
            new = [j for j in jm.jobs.values() if j.status in ("done", "failed") and j.id not in seen_done]
            seen_done.update(j.id for j in new)
            kinds = {j.kind for j in new}
            keep = gr.update()
            ds_view = show_dataset(ds)[:2] if kinds & {"import", "caption"} else (keep, keep)
            lists = refresh_lists() if kinds & {"download", "train", "import", "upload"} else [keep] * len(list_outs)
            return (_header(), _jobs_table(), job_log(jid), *ds_view, *lists)

        timer = gr.Timer(2.0)
        timer.tick(tick, [job_id, ds_pick],
                   [header, jobs_df, job_out, ds_gallery, ds_info, *list_outs],
                   concurrency_limit=None, show_progress="hidden")  # never waits behind a generation

        ui.load(refresh_lists, None, list_outs)
        # dropdowns start on their first choice without firing .change: fill dependents once
        ui.load(show_dataset, ds_pick, ds_out)
        ui.load(train_estimate, eta_in, t_eta)
    return ui


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--sync", action="store_true", help="pull presets + LoRAs from the sync repo at start")
    a = ap.parse_args()
    for d in (MODELS, LORAS, DATASETS, OUTPUTS, presets.PRESETS):
        d.mkdir(parents=True, exist_ok=True)
    if a.sync and settings.get()["sync_repo"]:
        try:
            print(presets.pull(settings.get()["sync_repo"]), flush=True)
        except Exception as e:
            print(f"sync pull failed: {e}", flush=True)
    ui = build()
    ui.queue(default_concurrency_limit=1, max_size=16)
    ui.launch(server_name="0.0.0.0", server_port=a.port, share=a.share, theme=THEME, css=CSS,
              allowed_paths=[str(WORK)], show_error=True)


if __name__ == "__main__":
    main()
