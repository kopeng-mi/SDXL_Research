"""SDXL_Research UI (NiceGUI). Runs inside the main venv.

  python -m sdxlr.webui [--port 7860] [--access tailscale|none] [--sync]

Access: Tailscale gives a fixed private address, http://sdxlr:7860, on your
tailnet (needs TS_AUTHKEY). The app signs out of Tailscale when it stops, so an
ephemeral key frees the name for the next Colab VM.

The event loop must stay free (it answers the browser's keep-alive pings):
memory/disk numbers are sampled in a background thread, and timers only push
changes to the page.

Layout: a sidebar (pages + live status: model, memory, current task, session
code) and one page at a time. Model loading and generation run in this process
(threads); caption/train/compare/downloads run as child-process jobs (jobs.py).
A GPU lock keeps generation and GPU jobs apart; the image model is unloaded
before a GPU job starts.

State: the form values of each session code live in STATES (shared by every
tab that uses the code) and are saved to /content/work/sessions every few
seconds, so a dropped connection or a new link picks up where it left off.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import secrets
import shutil
import threading
import time
import zipfile
from pathlib import Path

from nicegui import app, run, ui
from PIL import Image, ImageDraw, PngImagePlugin

from . import history, presets, resources, session, settings
from .activity import LOGS, act, log, setup_logging
from .engine import FP16_VAE, SAMPLERS, Z_ENCODERS, Z_MODELS, Z_REPO, Engine, Job
from .fetch import LORAS, MODELS, WORK, ZIMAGE, add_file, fetch, fetch_loras, list_loras, list_models, list_zimage
from .jobs import JobManager, fmt_secs
from .modes import ASPECTS, MODES, PONY_NEG, PONY_TAGS, STYLES, train_eta

DATASETS = WORK / "datasets"
OUTPUTS = WORK / "outputs"
UPLOADS = WORK / "uploads"
THUMBS = WORK / "thumbs"
IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
NEG_SDXL = "(worst quality:1.3), (low quality:1.2), lowres, bad anatomy, bad hands, watermark, blur, cartoon, anime, drawing"
LIB_KINDS = {"LoRA": "loras", "SDXL checkpoint": "model", "Z-Image model": "zimage"}
CAPTIONERS = {"qwen": "Qwen3-VL 4B, accurate sentences", "qwen-2b": "Qwen3-VL 2B, faster",
              "joy": "JoyCaption, uncensored", "wd": "WD tagger, tags for Pony/anime"}

eng = Engine()
gpu_lock = threading.Lock()


def _before_gpu_job(job):
    gpu_lock.acquire()  # waits for a running generation to finish
    eng.unload()


jm = JobManager(on_gpu_job_start=_before_gpu_job, on_gpu_job_end=lambda job: gpu_lock.release())

DEFAULTS = {
    # generate
    "kind": "Z-Image", "sdxl_model": None, "vae": "fp16fix", "z_model": "turbo Q8", "z_te": "official",
    "z_quant": "nf4", "mode": "Speed", "prompt": "", "negative": NEG_SDXL, "style": "Photo",
    "aspect": "Portrait 896x1152", "count": 1, "seed": -1, "use_adv": False, "steps": 20, "cfg": 5.0,
    "sampler": "dpmpp_2m_karras", "clip_skip": 2, "loras": [], "image": None, "info": "",
    "edit": False, "edit_src": None, "boxes": [], "strength": 0.75,
    # dataset
    "ds_name": "my_photos", "ds_link": "", "ds_pick": None, "cap_model": "qwen", "cap_trigger": "ohwx_person",
    "cap_class": "woman", "cap_prompt": "", "cap_limit": 0, "cap_over": False,
    # train
    "t_kind": "SDXL", "t_ds": None, "t_name": "", "t_base": None, "t_epochs": 15, "t_repeats": 6, "t_batch": 3,
    "t_res": 512, "t_dim": 24, "t_lr": 2e-4, "t_te_lr": 2e-5, "t_type": "LoRA", "t_hf": "", "t_resume": False,
    # compare
    "c_kind": "SDXL", "c_base": None, "c_z": "turbo Q8", "c_loras": [], "c_combos": "", "c_str": "0.6 0.8 1.0",
    "c_prompt": "ohwx_person, woman, portrait photo, natural light", "c_neg": NEG_SDXL, "c_steps": 20, "c_cfg": 5.0,
    "c_seed": 42, "c_name": "grid1",
    # library
    "l_link": "", "l_kind": "LoRA",
}
STATES: dict[str, dict] = {}
_saved: dict[str, str] = {}


def state_for(code: str) -> dict:
    if code not in STATES:
        STATES[code] = {**copy.deepcopy(DEFAULTS), **(session.load(code) or {})}
    return STATES[code]


def _autosave() -> None:
    while True:
        time.sleep(5)
        for code, s in list(STATES.items()):
            blob = json.dumps(s, default=str, sort_keys=True)
            if _saved.get(code) != blob:
                session.save(code, s)
                _saved[code] = blob


# ---------- small helpers ----------
def _lora_names():
    return [p.name for p in list_loras()]


def _lora_path(name):
    return next((str(p) for p in list_loras() if p.name == name), None)


def _model_names():
    return [p.name for p in list_models()]


def _model_path(name):
    return next((str(p) for p in list_models() if p.name == name), name)


def _z_names():
    return [*Z_MODELS, *(p.name for p in list_zimage())]


def _z_path(name):
    return Z_MODELS.get(name) or next((str(p) for p in list_zimage() if p.name == name), name)


def _datasets():
    return sorted(p.name for p in DATASETS.iterdir() if p.is_dir()) if DATASETS.exists() else []


def _images(ds):
    d = DATASETS / ds if ds else None
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG) if d and d.exists() else []


def url(p) -> str:
    """Browser URL of a file under /content/work."""
    return f"/work/{Path(p).relative_to(WORK).as_posix()}?v={int(Path(p).stat().st_mtime)}"


def thumb(p: Path, size: int = 320) -> str:
    t = THUMBS / Path(p).relative_to(WORK).with_suffix(".jpg")
    if not t.exists() or t.stat().st_mtime < p.stat().st_mtime:
        t.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(p).convert("RGB")
        im.thumbnail((size, size))
        im.save(t, quality=85)
    return url(t)


def err(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


# ---------- model ----------
def wanted(S) -> str:
    if S["kind"] == "Z-Image":
        return f"zimage|{_z_path(S['z_model'])}|{Z_ENCODERS.get(S['z_te'], S['z_te'])}|{S['z_quant']}"
    return _model_path(S["sdxl_model"]) if S["sdxl_model"] else ""


def loaded_label() -> str:
    if eng.kind == "zimage":
        return "Z-Image " + Path(eng.model_path.split("|")[1]).stem
    if eng.kind == "sdxl":
        return "SDXL " + Path(eng.model_path).stem
    return "No model loaded"


def _load(S, task) -> None:
    """Load the model picked in S (caller holds gpu_lock)."""
    if eng.pipe is not None and eng.model_path == wanted(S) and (S["kind"] == "Z-Image" or eng.vae_mode == S["vae"]):
        return
    if S["kind"] == "Z-Image":
        eng.load_zimage(_z_path(S["z_model"]), Z_ENCODERS.get(S["z_te"], S["z_te"]), S["z_quant"],
                        status=task.set_stage)
    else:
        if not S["sdxl_model"]:
            raise ValueError("Pick an SDXL checkpoint (add one in Library)")
        eng.load(_model_path(S["sdxl_model"]), S["vae"], status=task.set_stage)


def load_blocking(S, task) -> str:
    if jm.gpu_busy() or not gpu_lock.acquire(blocking=False):
        raise RuntimeError("The GPU is busy with a job. Load after it finishes (see Activity).")
    try:
        _load(S, task)
    finally:
        gpu_lock.release()
    return loaded_label()


def _mask(size, boxes):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for x0, y0, x1, y1 in boxes:
        d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=255)
    return m


def generate_blocking(S, task) -> tuple[list[str], str]:
    if jm.gpu_busy() or not gpu_lock.acquire(blocking=False):
        raise RuntimeError("The GPU is busy with a job. Generate after it finishes (see Activity).")
    try:
        _load(S, task)
        kind = eng.kind
        m = MODES[kind][S["mode"]]
        steps, cfg, sampler = (S["steps"], S["cfg"], S["sampler"]) if S["use_adv"] else \
            (m["steps"], m["cfg"], m.get("sampler", S["sampler"]))
        loras = [(_lora_path(n), float(w)) for n, w in S["loras"] if n and w]
        loras = [(p, w) for p, w in loras if p]
        if kind == "sdxl" and m.get("lora"):
            task.set_stage("Getting the Fast mode LoRA")
            loras.append((str(fetch(m["lora"][0], LORAS)), m["lora"][1]))
        prompt = S["prompt"].strip()
        full = ", ".join(x for x in (prompt, STYLES.get(S["style"], "")) if x)
        neg = S["negative"] if kind == "sdxl" else ""
        if kind == "sdxl" and "pony" in (eng.model_path or "").lower():
            full, neg = f"{PONY_TAGS}, {full}", f"{PONY_NEG}, {neg}"
        w, h = ASPECTS[S["aspect"]]
        image = mask = None
        if S["edit"] and S["edit_src"] and S["boxes"]:
            image = Image.open(S["edit_src"]).convert("RGB")
            mask = _mask(image.size, S["boxes"])
        job = Job(prompt=full, negative=neg, steps=int(steps), cfg=float(cfg), sampler=sampler, width=w, height=h,
                  seed=int(S["seed"]), count=int(S["count"]), clip_skip=int(S["clip_skip"]), loras=loras,
                  image=image, mask=mask, strength=float(S["strength"]), on_step=task.step)
        task.set_stage("Changing the area" if image else "Generating")
        t0 = time.time()
        images, seed, info = eng.generate(job)
    finally:
        gpu_lock.release()
    meta = {"prompt_raw": prompt, "prompt": full, "negative_raw": S["negative"], "negative": neg, "style": S["style"],
            "mode": S["mode"], "steps": job.steps, "cfg": job.cfg, "sampler": sampler, "clip_skip": job.clip_skip,
            "aspect": S["aspect"], "size": [w, h], "seed": seed, "count": job.count,
            "loras": [[n, float(x)] for n, x in S["loras"] if n and x],
            "kind": S["kind"], "sdxl_model": S["sdxl_model"], "vae": S["vae"], "z_model": S["z_model"],
            "z_te": S["z_te"], "z_quant": S["z_quant"], "inpaint": bool(image), "strength": job.strength,
            "seconds": round(time.time() - t0, 1), "made": time.strftime("%Y-%m-%d %H:%M:%S")}
    day = OUTPUTS / time.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    png = PngImagePlugin.PngInfo()
    png.add_text("sdxlr", json.dumps(meta))
    paths = []
    for i, im in enumerate(images):
        p = day / f"{time.strftime('%H%M%S')}_{seed + i}.png"
        im.save(p, pnginfo=png)
        history.save(p, {**meta, "seed": seed + i})
        paths.append(str(p))
    return paths, info


def reuse(S, meta: dict) -> None:
    """Fill the form from a history JSON so the same image can be made again."""
    for k in ("negative", "style", "mode", "steps", "cfg", "sampler", "clip_skip", "aspect", "seed", "kind",
              "sdxl_model", "vae", "z_model", "z_te", "z_quant", "strength"):
        if k in meta and meta[k] is not None:
            S[k] = meta[k]
    S["prompt"] = meta.get("prompt_raw", meta.get("prompt", ""))
    if "negative_raw" in meta:
        S["negative"] = meta["negative_raw"]
    S["loras"] = [list(x) for x in meta.get("loras", [])]
    S["use_adv"] = True  # same steps/cfg/sampler as recorded
    S["count"] = 1
    missing = [n for n, _ in S["loras"] if n not in _lora_names()]
    if missing:
        ui.notify(f"Missing LoRAs (add them in Library): {', '.join(missing)}", type="warning")


# ---------- jobs (child processes) ----------
def submit_caption(S):
    if not S["ds_pick"]:
        raise ValueError("Pick a dataset")
    args = ["-m", "sdxlr.caption", "--images", DATASETS / S["ds_pick"], "--model", S["cap_model"],
            "--trigger", S["cap_trigger"], "--class-word", S["cap_class"], "--limit", int(S["cap_limit"] or 0)]
    if S["cap_prompt"].strip():
        args += ["--prompt", S["cap_prompt"]]
    if S["cap_over"]:
        args += ["--overwrite"]
    jm.submit("caption", f"Caption {S['ds_pick']} ({S['cap_model']})", args, gpu=True)


def train_estimate(S) -> str:
    n = len(_images(S["t_ds"]))
    if not n:
        return "Pick a dataset to see how long it takes."
    k = "zimage" if S["t_kind"] == "Z-Image" else "sdxl"
    b = 1 if k == "zimage" else int(S["t_batch"] or 1)
    steps, secs, est = train_eta(k, n, int(S["t_repeats"] or 1), int(S["t_epochs"] or 1), b,
                                 int(S["t_res"]) if k == "zimage" else 1024)
    warn = ". Longer than a free T4 session (about 5 h): lower epochs or repeats" if secs > 4.5 * 3600 else ""
    return f"{steps} steps, about {fmt_secs(secs)}{' (estimate)' if est else ''} for {n} images{warn}"


def submit_train(S):
    ds = S["t_ds"]
    if not ds or not (_images(ds) or S["t_resume"]):
        raise ValueError("Pick a dataset with images")
    if S["t_resume"] and not S["t_hf"].strip():
        raise ValueError("Auto resume needs a private HF repo")
    name = S["t_name"].strip() or ds
    if S["t_kind"] == "Z-Image":
        args = ["-m", "sdxlr.train_zimage", "--name", name, "--images", DATASETS / ds, "--set",
                f"resolution={int(S['t_res'])}", f"epochs={int(S['t_epochs'])}", f"repeats={int(S['t_repeats'])}",
                f"network_dim={int(S['t_dim'])}", f"network_alpha={int(S['t_dim'])}", f"lr={S['t_lr']}"]
    else:
        base = _model_path(S["t_base"]) if S["t_base"] else "stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"
        args = ["-m", "sdxlr.train", "--name", name, "--images", DATASETS / ds, "--model", base,
                "--set", f"epochs={int(S['t_epochs'])}", f"repeats={int(S['t_repeats'])}",
                f"batch_size={int(S['t_batch'])}", f"network_dim={int(S['t_dim'])}",
                f"network_alpha={max(1, int(S['t_dim']) * 2 // 3)}", f"unet_lr={S['t_lr']}",
                f"text_encoder_lr={S['t_te_lr']}", f"lora_type={S['t_type']}"]
    if S["t_hf"].strip():
        args += ["--hf-repo", S["t_hf"].strip()]
    if S["t_resume"]:
        args += ["--resume"]
    jm.submit("train", f"Train {S['t_kind']} LoRA {name}", args, gpu=True)


def submit_compare(S):
    name = S["c_name"].strip() or "grid"
    args = ["-m", "sdxlr.compare", "--name", name, "--strengths", *S["c_str"].split(), "--prompt", S["c_prompt"],
            "--negative", S["c_neg"], "--steps", int(S["c_steps"]), "--cfg", S["c_cfg"], "--seed", int(S["c_seed"])]
    if S["c_kind"] == "Z-Image":
        args += ["--zimage", "--model", _z_path(S["c_z"])]
    else:
        if not S["c_base"]:
            raise ValueError("Pick an SDXL checkpoint")
        args += ["--model", _model_path(S["c_base"])]
    mixes = [c.strip() for c in S["c_combos"].split(",") if c.strip()]
    if mixes:
        args += ["--combos", *mixes]
    elif S["c_loras"]:
        args += ["--combos", *[f"{n}:1.0" for n in S["c_loras"]]]
    else:
        raise ValueError("Pick LoRAs or write mixes")
    jm.submit("compare", f"Compare {name}", args, gpu=True)


def logs_zip() -> Path:
    out = WORK / "sdxlr_logs.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in LOGS.rglob("*.log"):
            z.write(f, f.relative_to(LOGS))
    return out


# ---------- look ----------
HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {--bg:#1C1F24; --panel:#23272E; --line:#353B45; --text:#E6E8EB; --muted:#9AA3AE; --work:#E8A33D; --ok:#5FB98A;}
*, *::before, *::after {transition:none !important; animation:none !important;}
body, .q-layout, .q-page-container {background:var(--bg) !important; color:var(--text);
  font-family:'IBM Plex Sans', system-ui, sans-serif; font-size:15px; line-height:1.5;}
.q-field, .q-btn, .q-item, .q-toggle, .q-tab, .q-menu {font-family:inherit;}
.q-btn {text-transform:none; letter-spacing:0; font-weight:500; white-space:nowrap;}
.q-btn-toggle {flex-shrink:0;}
.q-drawer {background:var(--panel) !important;}
.q-drawer--left {border-right:1px solid var(--line);}
.t-title {font-size:24px; font-weight:500; line-height:1.25;}
.t-head {font-size:18px; font-weight:500;}
.t-small {font-size:13px;}
.muted {color:var(--muted);}
.work {color:var(--work);}
.tnum {font-variant-numeric:tabular-nums;}
.nav {width:100%; justify-content:flex-start; color:var(--muted); padding:6px 12px; border-radius:8px;}
.nav.active {color:var(--text); background:#2D323B;}
.section {border-top:1px solid var(--line); padding-top:12px;}
.bar {display:flex; height:6px; border-radius:3px; overflow:hidden; background:var(--line);}
.bar > span {height:100%; display:block;}
.bar .ghost {background:repeating-linear-gradient(45deg,#6B7480 0 4px,#4A515C 4px 8px);}
.bar .over {background:repeating-linear-gradient(45deg,#C0604F 0 4px,#7A3B31 4px 8px);}
.dot {display:inline-block; width:8px; height:8px; border-radius:4px; margin-right:6px; background:var(--line);}
.dot.ok {background:var(--ok);} .dot.busy {background:var(--work);}
.progress {height:3px; background:var(--line); border-radius:2px; overflow:hidden;}
.progress > span {display:block; height:100%; background:var(--work);}
.canvas {background:var(--panel); border:1px solid var(--line); border-radius:12px; min-height:520px;
  display:flex; align-items:center; justify-content:center; overflow:hidden;}
.thumb {width:96px; height:96px; object-fit:cover; border-radius:8px; cursor:pointer; border:1px solid var(--line);}
.thumb:hover {border-color:var(--muted);}
.grid-thumb {width:148px; height:148px; object-fit:cover; border-radius:8px; cursor:pointer;}
.gen-btn {height:48px; font-size:16px;}
.q-btn.gen-btn.busy {background:var(--panel) !important; color:var(--work) !important; outline:1px solid var(--work);}
.q-field--outlined .q-field__control:before {border-color:var(--line);}
</style>
"""
SEG = ["#7C8DA6", "#8FA89A", "#A8987C", "#9A8FA8", "#6F7B8A", "#A87C86", "#7CA3A8"]


def bar_html(total: float, blocks, preview=None) -> str:
    used = sum(g for _, g in blocks)
    segs = "".join(f'<span style="width:{100 * g / total:.2f}%;background:{SEG[i % len(SEG)]}" '
                   f'title="{n}: {g:.1f} GB"></span>' for i, (n, g) in enumerate(blocks) if g >= 0.05)
    if preview:
        cls = "over" if preview[1] > total - used else "ghost"
        segs += (f'<span class="{cls}" style="width:{min(100 * preview[1] / total, 100 - 100 * used / total):.2f}%" '
                 f'title="{preview[0]}: about {preview[1]:.1f} GB"></span>')
    return f'<div class="bar">{segs}</div>'


USAGE: dict = {}


def _sample_usage() -> None:
    """nvidia-smi, psutil and folder sizes are slow: measure here, never in the event loop."""
    while True:
        try:
            USAGE.update(resources.usage(eng, jm))
        except Exception as e:
            log.info(f"usage sample failed: {e}")
        time.sleep(2)


def set_if(el, attr: str, value) -> None:
    """Only send a change to the browser when the value really changed."""
    if getattr(el, attr) != value:
        setattr(el, attr, value)


def load_preview(S):
    if eng.pipe is not None and eng.model_path == wanted(S):
        return None
    if S["kind"] == "Z-Image":
        zp = _z_path(S["z_model"])
        local = MODELS / "zimage" / Path(zp).name if zp in Z_MODELS.values() else Path(zp)
        return resources.estimate("Z-Image", None, str(local), S["z_quant"])
    return resources.estimate("SDXL", _model_path(S["sdxl_model"]), None, S["z_quant"]) if S["sdxl_model"] else None


# ---------- pages ----------
PAGES = [("Generate", "/"), ("Dataset", "/dataset"), ("Train", "/train"), ("Compare", "/compare"),
         ("Library", "/library"), ("History", "/history"), ("Activity", "/activity")]


def field(el, S, key):
    return el.bind_value(S, key).props("outlined dense").classes("w-full")


def page_generate(S, sheet):
    with ui.row().classes("w-full gap-8 no-wrap items-start"):
        with ui.column().classes("gap-4").style("width:480px; min-width:380px"):
            ui.label("Generate").classes("t-title")
            with ui.row().classes("w-full items-end gap-2 no-wrap"):
                ui.toggle(["Z-Image", "SDXL"]).bind_value(S, "kind").props("no-caps unelevated toggle-color=grey-8")
                field(ui.select(_z_names(), label="Model", with_input=True, new_value_mode="add-unique"), S, "z_model") \
                    .classes("grow").bind_visibility_from(S, "kind", value="Z-Image")
                field(ui.select(_model_names(), label="Checkpoint", with_input=True), S, "sdxl_model") \
                    .classes("grow").bind_visibility_from(S, "kind", value="SDXL")
            with ui.row().classes("gap-2"):
                load_b = ui.button("Load model", on_click=lambda: do_load(S)).props("flat dense no-caps")
                ui.button("Unload", on_click=do_unload).props("flat dense no-caps")
                ui.button("More settings", on_click=sheet.toggle).props("flat dense no-caps")

            with ui.row().classes("w-full items-start no-wrap gap-1"):
                field(ui.textarea("Prompt", placeholder="ohwx_person, a woman in a green silk saree, standing in a garden"),
                      S, "prompt").props("autogrow").classes("grow")
                with ui.button(icon="history").props("flat dense round").tooltip("Recent prompts"):
                    with ui.menu() as menu:
                        def fill_menu():
                            menu.clear()
                            with menu:
                                for p in history.prompts() or ["No prompts yet"]:
                                    ui.menu_item(p[:90], on_click=lambda p=p: S.__setitem__("prompt", p))
                        menu.on("before-show", fill_menu)
            with ui.row().classes("w-full items-end gap-3"):
                ui.toggle(["Quality", "Speed", "Fast"]).bind_value(S, "mode").props("no-caps unelevated toggle-color=grey-8")
                field(ui.select(list(ASPECTS), label="Size"), S, "aspect").classes("grow")
            with ui.row().classes("w-full gap-3 no-wrap"):
                field(ui.number("Images", min=1, max=4, step=1, format="%d"), S, "count").classes("w-24")
                field(ui.number("Seed, -1 = random", step=1, format="%d"), S, "seed").classes("grow")

            ui.label("LoRAs").classes("t-small muted")
            lora_box = ui.column().classes("w-full gap-1")

            def draw_loras():
                lora_box.clear()
                names = _lora_names()
                with lora_box:
                    for i, row in enumerate(S["loras"]):
                        with ui.row().classes("w-full items-center no-wrap gap-2"):
                            ui.select(names, value=row[0] if row[0] in names else None, with_input=True,
                                      on_change=lambda e, i=i: S["loras"][i].__setitem__(0, e.value)) \
                                .props("outlined dense").classes("grow")
                            sl = ui.slider(min=-1, max=2, step=0.01, value=row[1],
                                           on_change=lambda e, i=i: S["loras"][i].__setitem__(1, e.value)).classes("w-28")
                            ui.label().bind_text_from(sl, "value", lambda v: f"{v:.2f}").classes("tnum t-small w-10")
                            ui.button(icon="close", on_click=lambda i=i: (S["loras"].pop(i), draw_loras())) \
                                .props("flat dense round size=sm")
                    ui.button("Add LoRA", on_click=lambda: (S["loras"].append([None, 0.8]), draw_loras())) \
                        .props("flat dense no-caps").classes("muted")

            draw_loras()
            gen_b = ui.button("Generate", on_click=lambda: do_generate(S)).classes("w-full gen-btn").props("unelevated no-caps color=grey-3 text-color=grey-10")
            info = ui.label().bind_text_from(S, "info").classes("t-small muted tnum")

        with ui.column().classes("gap-3").style("flex:1 1 0; min-width:0"):
            canvas = ui.element("div").classes("canvas w-full")
            status = ui.label().classes("t-small work tnum")
            prog = ui.html('<div class="progress"><span style="width:0%"></span></div>', sanitize=False).classes("w-full")
            with ui.row().classes("gap-2"):
                ui.button("Reuse settings", on_click=lambda: reuse_current(S)).props("flat dense no-caps")
                edit_b = ui.button("Change an area", on_click=lambda: toggle_edit(S)).props("flat dense no-caps")
                ui.button("Clear area", on_click=lambda: (S["boxes"].clear(), draw_canvas())) \
                    .props("flat dense no-caps").bind_visibility_from(S, "edit")
            recent = ui.row().classes("gap-2")

    shown = {"img": None, "edit": None, "boxes": -1}

    def svg_boxes():
        return "".join(f'<rect x="{min(a, c)}" y="{min(b, d)}" width="{abs(c - a)}" height="{abs(d - b)}" '
                       f'fill="rgba(232,163,61,.25)" stroke="#E8A33D" stroke-width="3"/>' for a, b, c, d in S["boxes"])

    start = {}

    def on_mouse(e):
        if e.type == "mousedown":
            start["p"] = (e.image_x, e.image_y)
        elif e.type == "mouseup" and "p" in start:
            x0, y0 = start.pop("p")
            if abs(e.image_x - x0) > 8 and abs(e.image_y - y0) > 8:
                S["boxes"].append([x0, y0, e.image_x, e.image_y])
                draw_canvas()

    def draw_canvas():
        img = S["edit_src"] if S["edit"] else S["image"]
        shown.update(img=img, edit=S["edit"], boxes=len(S["boxes"]))
        canvas.clear()
        with canvas:
            if img and Path(img).exists():
                if S["edit"]:
                    ii = ui.interactive_image(url(img), on_mouse=on_mouse, events=["mousedown", "mouseup"], cross=True) \
                        .classes("w-full")
                    ii.content = svg_boxes()
                else:
                    ui.image(url(img)).classes("w-full").props("fit=contain no-transition no-spinner")
            else:
                ui.label("Your image shows up here").classes("muted")
        recent.clear()
        with recent:
            for p, m in history.items(10):
                ui.image(thumb(p, 192)).classes("thumb").props("no-transition no-spinner").on(
                    "click", lambda p=p: (S.__setitem__("image", str(p)), S.__setitem__("edit", False), draw_canvas()))

    def tick():
        if shown["img"] != (S["edit_src"] if S["edit"] else S["image"]) or shown["edit"] != S["edit"] \
                or shown["boxes"] != len(S["boxes"]):
            draw_canvas()
        t = next((t for t in act.running() if t.kind in ("load", "generate")), None)
        if t:
            pct = 100 * t.done / t.total if t.total else 0
            step = f" {t.done}/{t.total}" if t.total else ""
            set_if(status, "text", f"{t.stage}{step}" + (f". {t.detail}" if t.detail else ""))
            set_if(prog, "content", f'<div class="progress"><span style="width:{pct:.0f}%"></span></div>')
            set_if(gen_b, "text", f"{t.stage}{step}")
            if "busy" not in gen_b.classes:
                gen_b.classes(add="busy")
        else:
            set_if(status, "text", "")
            set_if(prog, "content", '<div class="progress"><span style="width:0%"></span></div>')
            set_if(gen_b, "text", "Change the area" if S["edit"] else "Generate")
            if "busy" in gen_b.classes:
                gen_b.classes(remove="busy")
        set_if(edit_b, "text", "Back to the image" if S["edit"] else "Change an area")
        set_if(load_b, "text", "Loaded" if eng.pipe is not None and eng.model_path == wanted(S) else "Load model")

    draw_canvas()
    ui.timer(0.5, tick)

    # the side sheet: settings people rarely change
    sheet.clear()
    with sheet:
        with ui.column().classes("w-full gap-3 p-4"):
            ui.label("More settings").classes("t-head")
            field(ui.select(list(STYLES), label="Style"), S, "style")
            field(ui.textarea("Negative prompt (SDXL)"), S, "negative").props("autogrow")
            with ui.column().classes("w-full gap-2 section"):
                ui.label("Change an area").classes("t-small muted")
                ui.label().bind_text_from(S, "strength", lambda v: f"Strength {v:.2f}").classes("t-small tnum")
                ui.slider(min=0.1, max=1.0, step=0.05).bind_value(S, "strength")
            with ui.column().classes("w-full gap-2 section"):
                ui.checkbox("Use these instead of the mode").bind_value(S, "use_adv")
                with ui.row().classes("w-full no-wrap gap-2"):
                    field(ui.number("Steps", min=1, max=80, step=1, format="%d"), S, "steps").classes("grow")
                    field(ui.number("CFG", min=0, max=15, step=0.5), S, "cfg").classes("grow")
                field(ui.select(list(SAMPLERS), label="Sampler (SDXL)"), S, "sampler")
                field(ui.number("Clip skip (SDXL)", min=1, max=3, step=1, format="%d"), S, "clip_skip")
            with ui.column().classes("w-full gap-2 section"):
                ui.label("Model options").classes("t-small muted")
                field(ui.select(list(Z_ENCODERS), label="Z-Image text encoder", with_input=True,
                                new_value_mode="add-unique"), S, "z_te")
                ui.toggle({"nf4": "nf4, smaller", "int8": "int8, +2 GB"}).bind_value(S, "z_quant").props("no-caps unelevated toggle-color=grey-8")
                ui.toggle({"fp16fix": "fp16-fix VAE", "model": "Model's VAE"}).bind_value(S, "vae").props("no-caps unelevated toggle-color=grey-8")
            with ui.column().classes("w-full gap-2 section"):
                ui.label("Presets").classes("t-small muted")
                pname = ui.input("Name").props("outlined dense")
                psel = ui.select(presets.list_presets(), label="Saved presets").props("outlined dense")

                def p_save():
                    if not pname.value.strip():
                        ui.notify("Give the preset a name", type="warning")
                        return
                    presets.save(pname.value.strip(), {k: S[k] for k in (
                        "kind", "sdxl_model", "z_model", "mode", "prompt", "negative", "style", "aspect", "count",
                        "seed", "use_adv", "steps", "cfg", "sampler", "clip_skip", "loras")})
                    psel.options = presets.list_presets()
                    psel.update()
                    ui.notify(f"Saved preset {pname.value.strip()}")

                def p_load():
                    if psel.value:
                        S.update(presets.load(psel.value))
                        draw_loras()
                        ui.notify(f"Loaded preset {psel.value}")

                with ui.row().classes("gap-2"):
                    ui.button("Save", on_click=p_save).props("flat dense no-caps")
                    ui.button("Load", on_click=p_load).props("flat dense no-caps")
                    ui.button("Delete", on_click=lambda: (presets.delete(psel.value) if psel.value else None,
                                                          psel.set_options(presets.list_presets()))).props("flat dense no-caps")

    def toggle_edit(S):
        if not S["edit"] and not S["image"]:
            ui.notify("Generate or pick an image first", type="warning")
            return
        S["edit"] = not S["edit"]
        if S["edit"]:
            S["edit_src"], S["boxes"] = S["image"], []
            ui.notify("Drag on the image to mark the area to change, then press Change the area")
        draw_canvas()

    def reuse_current(S):
        if not S["image"]:
            return
        j = Path(S["image"]).with_suffix(".json")
        if j.exists():
            reuse(S, json.loads(j.read_text(encoding="utf-8")))
            draw_loras()
            ui.notify("Settings filled in from this image")


async def do_load(S):
    if any(t.kind in ("load", "generate") for t in act.running()):
        ui.notify("Already working on a model", type="warning")
        return
    task = act.start("load", S["z_model"] if S["kind"] == "Z-Image" else (S["sdxl_model"] or "SDXL"))
    try:
        label = await run.io_bound(load_blocking, copy.deepcopy(S), task)
        task.finish()
        ui.notify(f"{label} is ready")
    except Exception as e:
        task.fail(err(e))
        ui.notify(err(e), type="negative", multi_line=True)


def do_unload():
    with gpu_lock:
        eng.unload()
    log.info("[load] model unloaded")
    ui.notify("Model unloaded")


async def do_generate(S):
    if any(t.kind in ("load", "generate") for t in act.running()):
        ui.notify("Still working on the last one", type="warning")
        return
    if not S["prompt"].strip():
        ui.notify("Write a prompt first", type="warning")
        return
    task = act.start("generate", S["prompt"][:50])
    try:
        paths, info = await run.io_bound(generate_blocking, copy.deepcopy(S), task)
        task.finish(info)
        S["image"], S["info"], S["edit"] = paths[0], info, False
    except Exception as e:
        task.fail(err(e))
        ui.notify(err(e), type="negative", multi_line=True)


def page_dataset(S, sheet):
    staged: list[str] = []
    with ui.row().classes("w-full gap-8 no-wrap items-start"):
        with ui.column().classes("gap-3").style("width:420px; min-width:340px"):
            ui.label("Dataset").classes("t-title")
            ui.label("Add photos").classes("t-head")
            field(ui.input("Dataset name"), S, "ds_name")

            async def on_up(e):
                d = UPLOADS / "dataset"
                d.mkdir(parents=True, exist_ok=True)
                p = d / Path(e.file.name).name
                await e.file.save(p)
                staged.append(str(p))
                staged_l.text = f"{len(staged)} files ready to import"

            ui.upload(label="Drop a zip or photos (captions optional: same-name .txt)", multiple=True,
                      auto_upload=True, on_upload=on_up).props("flat bordered color=grey-9 accept=.zip,.txt,image/*").classes("w-full")
            staged_l = ui.label().classes("t-small muted")
            field(ui.textarea("Or paste links", placeholder="https://..., org/repo/file.zip, /content/drive/MyDrive/..., test"),
                  S, "ds_link").props("autogrow")

            def do_import():
                name = S["ds_name"].strip() or "dataset"
                srcs = staged + [s.strip() for s in S["ds_link"].replace("\n", ",").split(",") if s.strip()]
                if not srcs:
                    ui.notify("Drop files or paste a link first", type="warning")
                    return
                jm.submit("import", f"Import into {name}", ["-m", "sdxlr.data", "--dest", DATASETS / name, *srcs], gpu=False)
                staged.clear()
                staged_l.text = ""
                pick.set_options(sorted({*_datasets(), name}), value=name)
                S["ds_pick"] = name
                ui.notify("Importing. Progress is in the sidebar")

            ui.button("Import", on_click=do_import).props("unelevated no-caps color=grey-3 text-color=grey-10").classes("gen-btn w-full")

            with ui.column().classes("w-full gap-3 section"):
                ui.label("Write captions").classes("t-head")
                field(ui.select(CAPTIONERS, label="Captioner"), S, "cap_model")
                with ui.row().classes("w-full no-wrap gap-2"):
                    field(ui.input("Trigger word"), S, "cap_trigger").classes("grow")
                    field(ui.input("Class word"), S, "cap_class").classes("grow")
                with ui.expansion("Advanced").classes("w-full"):
                    field(ui.textarea("Custom instruction (empty = identity preset; {cls} = class word)"), S, "cap_prompt")
                    field(ui.number("Only the first N (0 = all)", min=0, step=1, format="%d"), S, "cap_limit")
                    ui.checkbox("Overwrite existing captions").bind_value(S, "cap_over")

                def do_caption():
                    try:
                        submit_caption(S)
                        ui.notify("Captioning queued. Progress is in the sidebar")
                    except Exception as e:
                        ui.notify(str(e), type="warning")

                ui.button("Write captions", on_click=do_caption).props("unelevated no-caps color=grey-3 text-color=grey-10").classes("gen-btn w-full")

        with ui.column().classes("gap-3").style("flex:1 1 0; min-width:0"):
            with ui.row().classes("w-full items-end gap-3"):
                pick = field(ui.select(_datasets(), label="Dataset"), S, "ds_pick").classes("w-64")
                counts = ui.label().classes("muted tnum")
            grid = ui.row().classes("gap-2")

    def edit_caption(p: Path):
        t = p.with_suffix(".txt")
        with ui.dialog() as d, ui.card().classes("w-[720px] max-w-full").style("background:var(--panel)"):
            ui.image(thumb(p, 720)).classes("w-full").props("fit=contain no-transition no-spinner").style("max-height:420px")
            ta = ui.textarea("Caption", value=t.read_text(encoding="utf-8") if t.exists() else "").props("outlined autogrow").classes("w-full")
            with ui.row():
                ui.button("Save caption", on_click=lambda: (t.write_text(ta.value.strip(), encoding="utf-8"),
                                                            ui.notify("Caption saved"), d.close(), draw())).props("unelevated no-caps color=grey-3 text-color=grey-10")
                ui.button("Close", on_click=d.close).props("flat no-caps")
        d.open()

    seen = {"sig": None}

    def draw():
        imgs = _images(S["ds_pick"])
        caps = sum(p.with_suffix(".txt").exists() for p in imgs)
        seen["sig"] = (S["ds_pick"], len(imgs), caps)
        counts.text = f"{len(imgs)} photos, {caps} with captions" if S["ds_pick"] else ""
        grid.clear()
        with grid:
            if not imgs:
                ui.label("Add photos on the left, then pick the dataset here.").classes("muted")
            for p in imgs:
                with ui.column().classes("gap-1").style("width:148px"):
                    ui.image(thumb(p)).classes("grid-thumb").props("no-transition no-spinner").on("click", lambda p=p: edit_caption(p))
                    t = p.with_suffix(".txt")
                    ui.label(t.read_text(encoding="utf-8")[:70] if t.exists() else "No caption").classes("t-small muted") \
                        .style("line-height:1.3; height:34px; overflow:hidden")

    def tick():
        if not set(_datasets()) <= set(pick.options):
            pick.set_options(sorted({*_datasets(), *pick.options}), value=S["ds_pick"])
        imgs = _images(S["ds_pick"])
        if seen["sig"] != (S["ds_pick"], len(imgs), sum(p.with_suffix(".txt").exists() for p in imgs)):
            draw()

    draw()
    ui.timer(2.0, tick)


def page_train(S, sheet):
    with ui.column().classes("gap-3").style("width:560px; max-width:100%"):
        ui.label("Train a LoRA").classes("t-title")
        ui.toggle(["SDXL", "Z-Image"]).bind_value(S, "t_kind").props("no-caps unelevated toggle-color=grey-8")
        with ui.row().classes("w-full no-wrap gap-2"):
            ds = field(ui.select(_datasets(), label="Dataset"), S, "t_ds").classes("grow")
            field(ui.input("LoRA name"), S, "t_name").classes("grow")
        field(ui.select(_model_names(), label="Base checkpoint (empty = SDXL 1.0 base)", clearable=True), S, "t_base") \
            .bind_visibility_from(S, "t_kind", value="SDXL")
        with ui.row().classes("w-full no-wrap gap-2"):
            field(ui.number("Epochs", min=1, step=1, format="%d"), S, "t_epochs").classes("grow")
            field(ui.number("Repeats", min=1, step=1, format="%d"), S, "t_repeats").classes("grow")
            field(ui.number("Batch", min=1, max=4, step=1, format="%d"), S, "t_batch").classes("grow") \
                .bind_visibility_from(S, "t_kind", value="SDXL")
            field(ui.select({512: "512 px", 768: "768 px (untested)"}, label="Resolution"), S, "t_res").classes("grow") \
                .bind_visibility_from(S, "t_kind", value="Z-Image")
        eta = ui.label().classes("tnum")
        with ui.expansion("Advanced").classes("w-full"):
            with ui.row().classes("w-full no-wrap gap-2"):
                field(ui.number("Rank", min=1, step=1, format="%d"), S, "t_dim").classes("grow")
                field(ui.number("Learning rate", format="%.1e"), S, "t_lr").classes("grow")
            with ui.row().classes("w-full no-wrap gap-2").bind_visibility_from(S, "t_kind", value="SDXL"):
                field(ui.number("Text encoder LR (0 = off)", format="%.1e"), S, "t_te_lr").classes("grow")
                ui.toggle(["LoRA", "LoCon"]).bind_value(S, "t_type").props("no-caps unelevated toggle-color=grey-8")
            field(ui.input("Private HF repo for checkpoints", placeholder="yourname/my-loras"), S, "t_hf")
            ui.checkbox("Auto resume: keep the dataset and the latest checkpoint in that repo; "
                        "the same LoRA name later continues").bind_value(S, "t_resume")

        def start():
            try:
                submit_train(S)
                ui.notify("Training queued. The image model is unloaded while it runs")
            except Exception as e:
                ui.notify(str(e), type="warning")

        ui.button("Start training", on_click=start).props("unelevated no-caps color=grey-3 text-color=grey-10").classes("gen-btn w-full")
        ui.label("A LoRA is saved every epoch; compare them in Compare. SDXL trains at 1024 px (about 5 s a step "
                 "on a T4), Z-Image at 512 px (about 6 s a step) on the base model and is used on Turbo.") \
            .classes("t-small muted")

    last = {"k": None}

    def tick():
        k = (S["t_kind"], S["t_ds"], S["t_repeats"], S["t_epochs"], S["t_batch"], S["t_res"])
        if k != last["k"]:
            last["k"] = k
            eta.text = train_estimate(S)
        if ds.options != _datasets():
            ds.set_options(_datasets(), value=S["t_ds"] if S["t_ds"] in _datasets() else None)

    kind_seen = {"v": S["t_kind"]}

    def kind_defaults():  # switching trainer resets the numbers to that trainer's defaults
        if S["t_kind"] != kind_seen["v"]:
            kind_seen["v"] = S["t_kind"]
            z = S["t_kind"] == "Z-Image"
            S.update(t_epochs=10 if z else 15, t_repeats=4 if z else 6, t_dim=16 if z else 24, t_lr=1e-4 if z else 2e-4)

    ui.timer(1.0, tick)
    ui.timer(0.5, kind_defaults)


def page_compare(S, sheet):
    with ui.row().classes("w-full gap-8 no-wrap items-start"):
        with ui.column().classes("gap-3").style("width:420px; min-width:340px"):
            ui.label("Compare").classes("t-title")
            ui.toggle(["SDXL", "Z-Image"]).bind_value(S, "c_kind").props("no-caps unelevated toggle-color=grey-8")
            field(ui.select(_model_names(), label="Checkpoint"), S, "c_base").bind_visibility_from(S, "c_kind", value="SDXL")
            field(ui.select(_z_names(), label="Z-Image model", new_value_mode="add-unique"), S, "c_z") \
                .bind_visibility_from(S, "c_kind", value="Z-Image")
            field(ui.select(_lora_names(), label="LoRAs, one row each", multiple=True, with_input=True), S, "c_loras").props("use-chips")
            field(ui.input("Or mixes, comma separated", placeholder="face_a.safetensors:0.44+face_b.safetensors:0.44"), S, "c_combos")
            field(ui.input("Strengths (columns)"), S, "c_str")
            field(ui.textarea("Prompt"), S, "c_prompt").props("autogrow")
            field(ui.input("Grid name"), S, "c_name")
            with ui.expansion("Advanced").classes("w-full"):
                field(ui.input("Negative (SDXL)"), S, "c_neg")
                with ui.row().classes("w-full no-wrap gap-2"):
                    field(ui.number("Steps", step=1, format="%d"), S, "c_steps").classes("grow")
                    field(ui.number("CFG", step=0.5), S, "c_cfg").classes("grow")
                    field(ui.number("Seed", step=1, format="%d"), S, "c_seed").classes("grow")

            def go():
                try:
                    submit_compare(S)
                    ui.notify("Compare queued. The grid shows up here when it is done")
                except Exception as e:
                    ui.notify(str(e), type="warning")

            ui.button("Make the grid", on_click=go).props("unelevated no-caps color=grey-3 text-color=grey-10").classes("gen-btn w-full")
        box = ui.element("div").classes("canvas").style("flex:1 1 0; min-width:0")

    seen = {"m": None}

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
                    ui.label("The grid for this name shows up here").classes("muted")

    def kind_defaults():
        z = S["c_kind"] == "Z-Image"
        if (S["c_steps"], S["c_cfg"]) in ((20, 5.0), (9, 0.0)):
            S.update(c_steps=9 if z else 20, c_cfg=0.0 if z else 5.0)

    tick()
    ui.timer(2.0, tick)
    ui.timer(1.0, kind_defaults)


def page_library(S, sheet):
    with ui.row().classes("w-full gap-8 no-wrap items-start"):
        with ui.column().classes("gap-3").style("width:440px; min-width:340px"):
            ui.label("Library").classes("t-title")
            ui.toggle(list(LIB_KINDS)).bind_value(S, "l_kind").props("no-caps unelevated toggle-color=grey-8")
            field(ui.textarea("Links, one per line", placeholder="cyberdelia/CyberRealisticPony/CyberRealisticPony_V17.0_FP16.safetensors\n"
                                                                 "https://huggingface.co/org/repo/blob/main/file.safetensors\n"
                                                                 "https://civitai.com/api/download/models/12345"), S, "l_link").props("autogrow")

            def dl():
                links = [s.strip() for s in S["l_link"].replace("\n", ",").split(",") if s.strip()]
                if not links:
                    ui.notify("Paste a link first", type="warning")
                    return
                jm.submit("download", f"Download {', '.join(Path(x).name for x in links)[:60]}",
                          ["-m", "sdxlr.fetch", LIB_KINDS[S["l_kind"]], *links], gpu=False)
                S["l_link"] = ""
                ui.notify("Downloading. Progress is in the sidebar")

            ui.button("Download", on_click=dl).props("unelevated no-caps color=grey-3 text-color=grey-10").classes("gen-btn w-full")

            async def on_up(e):
                d = UPLOADS / "library"
                d.mkdir(parents=True, exist_ok=True)
                p = d / Path(e.file.name).name
                await e.file.save(p)
                k = LIB_KINDS[S["l_kind"]]
                if k == "loras":
                    names = [x.name for x in fetch_loras(str(p))] if p.suffix == ".zip" else [add_file(str(p), LORAS).name]
                else:
                    names = [add_file(str(p), MODELS if k == "model" else ZIMAGE).name]
                ui.notify(f"Added {', '.join(names)}")
                draw()

            ui.upload(label="Or upload files (up to about 100 MB each through the link; use links for big models)",
                      multiple=True, auto_upload=True, on_upload=on_up).props("flat bordered color=grey-9 accept=.safetensors,.gguf,.ckpt,.zip").classes("w-full")
            ui.label("Private HF files and Civitai need HF_TOKEN and CIVITAI_TOKEN in Colab Secrets.").classes("t-small muted")
            with ui.expansion("Sync presets and LoRAs with a private HF repo").classes("w-full"):
                st = settings.get()
                sync = ui.input("Sync repo", value=st["sync_repo"], placeholder="yourname/sdxlr-sync").props("outlined dense").classes("w-full")

                def do_sync(direction):
                    settings.set(sync_repo=sync.value.strip())
                    if not sync.value.strip():
                        ui.notify("Set the sync repo first", type="warning")
                        return
                    try:
                        msg = presets.push(sync.value.strip()) if direction == "push" else presets.pull(sync.value.strip())
                        ui.notify(msg)
                        draw()
                    except Exception as e:
                        ui.notify(err(e), type="negative")

                with ui.row():
                    ui.button("Upload to repo", on_click=lambda: do_sync("push")).props("flat dense no-caps")
                    ui.button("Download from repo", on_click=lambda: do_sync("pull")).props("flat dense no-caps")

        with ui.column().classes("gap-2").style("flex:1 1 0; min-width:0"):
            with ui.row().classes("w-full items-center"):
                ui.label("On this VM").classes("t-head")
                ui.space()
                ui.button("Clear download cache", on_click=lambda: clear_cache()).props("flat dense no-caps") \
                    .tooltip("Deletes re-downloadable cache files that are not in use")
            table = ui.column().classes("w-full gap-0")

    def clear_cache():
        freed = resources.clear_download_cache([eng.model_path or "", Z_REPO, FP16_VAE])
        ui.notify(f"Freed {freed:.1f} GB")

    seen = {"sig": None}

    def files():
        return ([("SDXL checkpoint", p) for p in list_models()] + [("Z-Image model", p) for p in list_zimage()]
                + [("LoRA", p) for p in list_loras()])

    def delete(p: Path):
        p.unlink(missing_ok=True)
        ui.notify(f"Deleted {p.name}")
        draw()

    def draw():
        fs = files()
        seen["sig"] = [str(p) for _, p in fs]
        table.clear()
        with table:
            if not fs:
                ui.label("Nothing here yet. Paste a link or upload a file.").classes("muted")
            for kind, p in fs:
                with ui.row().classes("w-full items-center no-wrap gap-3 py-1").style("border-bottom:1px solid var(--line)"):
                    ui.label(p.name).classes("grow").style("overflow-wrap:anywhere")
                    ui.label(kind).classes("t-small muted w-32")
                    ui.label(f"{p.stat().st_size / 2**30:.2f} GB").classes("t-small muted tnum w-20 text-right")
                    ui.button(icon="delete_outline", on_click=lambda p=p: delete(p)).props("flat dense round size=sm") \
                        .tooltip(f"Delete {p.name}")

    def tick():
        if seen["sig"] != [str(p) for _, p in files()]:
            draw()

    draw()
    ui.timer(3.0, tick)


def page_history(S, sheet):
    with ui.column().classes("w-full gap-3"):
        with ui.row().classes("w-full items-center"):
            ui.label("History").classes("t-title")
            ui.space()

            async def up(e):
                try:
                    meta = history.read(await e.file.read(), e.file.name)
                    reuse(S, meta)
                    ui.notify("Settings loaded. Open Generate to make it again")
                    ui.navigate.to("/")
                except Exception as ex:
                    ui.notify(err(ex), type="negative")

            ui.upload(label="Load settings from a .json or a PNG made here", auto_upload=True, on_upload=up) \
                .props("flat bordered color=grey-9 accept=.json,.png").classes("w-96")
        grid = ui.row().classes("gap-3")

    def show(p: Path, m: dict):
        with ui.dialog() as d, ui.card().classes("w-[900px] max-w-full").style("background:var(--panel)"):
            with ui.row().classes("w-full no-wrap gap-4"):
                ui.image(url(p)).classes("w-1/2").props("fit=contain no-transition no-spinner")
                with ui.column().classes("w-1/2 gap-2"):
                    ui.label(m.get("prompt_raw") or m.get("prompt", "")).classes("t-head").style("line-height:1.35")
                    model = m.get("z_model") if m.get("kind") == "Z-Image" else m.get("sdxl_model")
                    rows = [("Model", f"{m.get('kind')} {model}"), ("Mode", m.get("mode")),
                            ("Steps and CFG", f"{m.get('steps')}, {m.get('cfg')}"), ("Seed", m.get("seed")),
                            ("Size", "x".join(map(str, m.get("size", [])))),
                            ("LoRAs", ", ".join(f"{n} {w:.2f}" for n, w in m.get("loras", [])) or "none"),
                            ("Made", f"{m.get('made', '')}, {m.get('seconds', '?')} s")]
                    for k, v in rows:
                        with ui.row().classes("no-wrap gap-2 t-small"):
                            ui.label(k).classes("muted w-28")
                            ui.label(str(v))
                    with ui.row().classes("gap-2 pt-2"):
                        ui.button("Make it again", on_click=lambda: (reuse(S, m), d.close(), ui.navigate.to("/"))) \
                            .props("unelevated no-caps color=grey-3 text-color=grey-10").classes("gen-btn")
                        ui.button("Download settings", on_click=lambda: ui.download.file(p.with_suffix(".json"))) \
                            .props("flat no-caps")
                        ui.button("Download image", on_click=lambda: ui.download.file(p)).props("flat no-caps")
        d.open()

    seen = {"n": None}

    def draw():
        items = history.items(120)
        seen["n"] = (len(items), items[0][0].name if items else None)
        grid.clear()
        with grid:
            if not items:
                ui.label("Images you generate show up here, with the settings to make them again.").classes("muted")
            for p, m in items:
                with ui.column().classes("gap-1").style("width:148px"):
                    ui.image(thumb(p)).classes("grid-thumb").props("no-transition no-spinner").on("click", lambda p=p, m=m: show(p, m))
                    ui.label((m.get("prompt_raw") or m.get("prompt", ""))[:60]).classes("t-small muted") \
                        .style("line-height:1.3; height:34px; overflow:hidden")

    def tick():
        items = history.items(1)
        if seen["n"] is None or (items and items[0][0].name != seen["n"][1]):
            draw()

    draw()
    ui.timer(3.0, tick)


def page_activity(S, sheet):
    sel = {"job": None}
    with ui.column().classes("w-full gap-3"):
        with ui.row().classes("w-full items-center"):
            ui.label("Activity").classes("t-title")
            ui.space()
            ui.button("Download all logs", on_click=lambda: ui.download.file(logs_zip())).props("flat dense no-caps")
        rows = ui.column().classes("w-full gap-0")
        with ui.row().classes("w-full items-center"):
            title = ui.label("App log").classes("t-head")
            ui.space()
            cancel = ui.button("Cancel this job", on_click=lambda: (jm.cancel(sel["job"]), ui.notify("Cancelled"))) \
                .props("flat dense no-caps")
        logv = ui.log(max_lines=400).classes("w-full").style("height:360px; background:var(--panel); font-size:13px")
    pos = {"file": None, "off": 0}

    def entries():
        out = []
        for t in act.tasks:
            prog = f"{t.done}/{t.total}" if t.total else ""
            out.append((t.started, None, t.title, t.kind, t.status, t.stage if t.status == "running" else (t.error or t.detail),
                        prog, fmt_secs(t.elapsed)))
        for j in jm.jobs.values():
            prog = f"{j.done}/{j.total}" if j.total else ""
            eta = f", {fmt_secs(j.eta)} left" if j.status == "running" and j.eta else ""
            took = fmt_secs((j.ended or time.time()) - j.started) if j.started else ""
            out.append((j.created, j.id, j.title, j.kind, j.status, eta.lstrip(", "), prog, took))
        return sorted(out, key=lambda r: r[0], reverse=True)[:30]

    def draw():
        rows.clear()
        with rows:
            es = entries()
            if not es:
                ui.label("Nothing has run yet.").classes("muted")
            for created, jid, ttl, kind, status, note, prog, took in es:
                cls = "work" if status in ("running", "queued") else ("muted" if status == "done" else "")
                r = ui.row().classes("w-full items-center no-wrap gap-3 py-1").style(
                    "border-bottom:1px solid var(--line);" + ("cursor:pointer" if jid else ""))
                with r:
                    ui.label(time.strftime("%H:%M", time.localtime(created))).classes("t-small muted tnum w-12")
                    ui.label(ttl).classes("grow")
                    ui.label(note or "").classes("t-small muted")
                    ui.label(prog).classes("t-small tnum w-20 text-right")
                    ui.label(took).classes("t-small muted tnum w-16 text-right")
                    ui.label(status).classes(f"t-small w-20 {cls}")
                if jid:
                    r.on("click", lambda jid=jid, ttl=ttl: pick(jid, ttl))

    def pick(jid, ttl):
        sel["job"] = jid
        title.text = ttl
        pos.update(file=None, off=0)
        logv.clear()

    def tail():
        f = LOGS / "jobs" / f"{sel['job']}.log" if sel["job"] else LOGS / "app.log"
        cancel.set_visibility(bool(sel["job"]) and jm.jobs.get(sel["job"]) is not None
                              and jm.jobs[sel["job"]].status in ("running", "queued"))
        if pos["file"] != f:
            pos.update(file=f, off=max(f.stat().st_size - 40000, 0) if f.exists() else 0)
        if f.exists() and f.stat().st_size > pos["off"]:
            with open(f, "rb") as fh:
                fh.seek(pos["off"])
                data = fh.read()
            pos["off"] += len(data)
            for line in data.decode("utf-8", "replace").splitlines():
                logv.push(line)

    draw()
    tail()
    ui.timer(2.0, draw)
    ui.timer(1.0, tail)


BUILDERS = {"/": page_generate, "/dataset": page_dataset, "/train": page_train, "/compare": page_compare,
            "/library": page_library, "/history": page_history, "/activity": page_activity}


# ---------- frame: sidebar + pages ----------
def root():
    ui.add_head_html(HEAD)
    ui.dark_mode(True)
    ui.colors(primary="#9AA3AE")
    code = app.storage.browser.get("code")
    if not session.valid(code) or session.load(code) is None:
        # a new browser (or a new VM) has no cookie: one user, so continue the latest session
        code = session.latest() or session.new_code()
        app.storage.browser["code"] = code
    S = state_for(code)

    drawer = ui.right_drawer(value=False).props("bordered overlay width=380")
    with drawer:
        sheet = ui.column().classes("w-full")
    sheet.toggle = drawer.toggle  # pages fill the sheet; Generate's "More settings" opens it

    navs = {}
    with ui.left_drawer(value=True, fixed=True).props("width=248").classes("p-4"):
        with ui.column().classes("w-full h-full gap-1 no-wrap"):
            ui.label("SDXL Research").classes("t-head pb-3")
            for name, path in PAGES:
                navs[path] = ui.button(name, on_click=lambda path=path: ui.navigate.to(path)).props("flat no-caps align=left").classes("nav")
            ui.space()
            with ui.column().classes("w-full gap-2 section"):
                model_l = ui.html("", sanitize=False).classes("t-small")
                mem = ui.html("", sanitize=False).classes("w-full")
            with ui.column().classes("w-full gap-1 section"):
                task_l = ui.label("Nothing running").classes("t-small muted")
                task_d = ui.label().classes("t-small muted tnum")
                task_p = ui.html("", sanitize=False).classes("w-full")
            with ui.column().classes("w-full gap-1 section"):
                ui.label(f"Session {code}").classes("t-small tnum").tooltip(
                    "Open the link again, or type this code in another browser, to continue where you left off")
                with ui.row().classes("w-full no-wrap gap-1 items-center"):
                    code_in = ui.input(placeholder="Other code").props("outlined dense maxlength=4").classes("grow")

                    def resume():
                        c = (code_in.value or "").strip()
                        if session.load(c) is None:
                            ui.notify(f"No session {c} on this VM", type="warning")
                            return
                        app.storage.browser["code"] = c
                        ui.navigate.reload()

                    ui.button("Open", on_click=resume).props("flat dense no-caps")

    def page(builder):
        def build():
            for p, b in navs.items():
                b.classes(remove="active")
            navs[next(p for p, f in BUILDERS.items() if f is builder)].classes(add="active")
            if builder is not page_generate:
                drawer.value = False
                sheet.clear()
            builder(S, sheet)
        return build

    with ui.column().classes("w-full p-6").style("max-width:1400px"):
        ui.sub_pages({p: page(b) for p, b in BUILDERS.items()}).classes("w-full")

    def side_tick():
        # model
        busy = next((t for t in act.running() if t.kind in ("load", "generate")), None)
        if busy and busy.kind == "load":
            dot, state = "busy", "Loading"
        elif eng.pipe is not None:
            dot, state = ("busy", "Generating") if busy else ("ok", "Ready")
        else:
            dot, state = "", "Pick a model and press Generate"
        set_if(model_l, "content", f'<div><span class="dot {dot}"></span>{loaded_label() if eng.pipe is not None else "No model loaded"}</div>'
                                   f'<div class="muted">{state}</div>')
        # memory
        u = dict(USAGE)
        prev = load_preview(S)
        html = []
        for name, (total, blocks) in u.items():
            if name == "VRAM" and prev and eng.pipe is not None:
                blocks = [b for b in blocks if not b[0].startswith(("SDXL", "Z-Image", "working"))]
            used = sum(g for _, g in blocks)
            html.append(f'<div class="t-small" style="display:flex;justify-content:space-between">'
                        f'<span class="muted">{name}</span><span class="tnum">{used:.1f} / {total:.0f} GB</span></div>'
                        + bar_html(total, blocks, prev if name == "VRAM" else None))
        set_if(mem, "content", '<div style="display:flex;flex-direction:column;gap:6px">' + "".join(html) + "</div>")
        # current work: in-process task first, then a running job
        job = next((j for j in jm.recent(20) if j.status == "running"), None)
        queued = sum(j.status == "queued" for j in jm.jobs.values())
        if busy:
            step = f" {busy.done}/{busy.total}" if busy.total else ""
            task_l.classes(replace="t-small work")
            task_l.text = f"{busy.stage}{step}"
            task_d.text = busy.detail or busy.title
            pct = 100 * busy.done / busy.total if busy.total else 0
        elif job:
            task_l.classes(replace="t-small work")
            task_l.text = job.title
            task_d.text = (f"{job.done}/{job.total}" + (f", {fmt_secs(job.eta)} left" if job.eta else "")) if job.total else "Working"
            pct = 100 * job.done / job.total if job.total else 0
        else:
            task_l.classes(replace="t-small muted")
            task_l.text, task_d.text, pct = "Nothing running", "", 0
        if queued:
            task_d.text += f" ({queued} queued)"
        set_if(task_p, "content", f'<div class="progress"><span style="width:{pct:.0f}%"></span></div>' if (busy or job) else "")

    side_tick()
    ui.timer(1.0, side_tick)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--access", choices=["tailscale", "none"], default="tailscale")
    ap.add_argument("--sync", action="store_true", help="pull presets + LoRAs from the sync repo at start")
    a = ap.parse_args()
    setup_logging()
    for d in (MODELS, LORAS, DATASETS, OUTPUTS, UPLOADS, THUMBS, presets.PRESETS):
        d.mkdir(parents=True, exist_ok=True)
    if a.sync and settings.get()["sync_repo"]:
        try:
            log.info(presets.pull(settings.get()["sync_repo"]))
        except Exception as e:
            log.info(f"sync pull failed: {e}")
    threading.Thread(target=_autosave, daemon=True).start()
    app.add_media_files("/work", WORK)
    secret_f = WORK / ".ui_secret"
    if not secret_f.exists():
        secret_f.write_text(secrets.token_hex(16))
    threading.Thread(target=_sample_usage, daemon=True).start()

    def announce(u):
        log.info("=" * 60)
        log.info(f"Open the app: {u}")
        log.info("=" * 60)

    if a.access == "tailscale":
        from . import tailnet

        def ts_up():
            try:
                announce(tailnet.up(a.port) + "  (on a device signed in to your Tailscale)")
            except Exception as e:
                log.info(f"Tailscale failed: {e}")
                log.info(f"Add TS_AUTHKEY in Colab Secrets, run Setup again, then Launch. Local address: http://127.0.0.1:{a.port}")

        app.on_startup(lambda: threading.Thread(target=ts_up, daemon=True).start())
        app.on_shutdown(tailnet.down)
    log.info(f"app starting on port {a.port}")
    ui.run(root, host="0.0.0.0", port=a.port, title="SDXL Research", show=False, reload=False, dark=True,
           storage_secret=secret_f.read_text(), reconnect_timeout=600, favicon="◐")


if __name__ == "__main__":
    main()
