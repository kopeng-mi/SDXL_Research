"""Shared state and the work behind the UI (no layout here).

Model loading and generation run in this process (threads, via run.io_bound);
caption/train/compare/downloads run as child-process jobs (jobs.py). A GPU lock
keeps generation and GPU jobs apart; the image model is unloaded before a GPU
job starts.

State: the form values of each session code live in STATES (shared by every
tab that uses the code) and are saved to WORK/sessions every few seconds, so a
dropped connection or a new link picks up where it left off.

Every submit_* / *_blocking function checks its inputs first and raises
ValueError with a sentence the user can act on.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, PngImagePlugin

from .. import history, resources, session
from ..activity import LOGS, act, log
from ..engine import SAMPLERS, Z_ENCODERS, Z_MODELS, Engine, Job  # noqa: F401  (SAMPLERS used by pages)
from ..fetch import LORAS, MODELS, WORK, fetch, list_loras, list_models, list_zimage
from ..jobs import JobManager, fmt_secs
from ..modes import ASPECTS, MODES, PONY_NEG, PONY_TAGS, STYLES, train_eta

DATASETS = WORK / "datasets"
OUTPUTS = WORK / "outputs"
UPLOADS = WORK / "uploads"
THUMBS = WORK / "thumbs"
IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
NEG_SDXL = "(worst quality:1.3), (low quality:1.2), lowres, bad anatomy, bad hands, watermark, blur, cartoon, anime, drawing"
SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"

eng = Engine()
gpu_lock = threading.Lock()


def _before_gpu_job(job):
    gpu_lock.acquire()  # waits for a running generation to finish
    if eng.pipe is not None:
        log.info(f"[job {job.id}] unloading the image model to free the GPU")
    eng.unload()


jm = JobManager(on_gpu_job_start=_before_gpu_job, on_gpu_job_end=lambda job: gpu_lock.release())

DEFAULTS = {
    # generate
    "kind": "Z-Image", "sdxl_model": None, "vae": "fp16fix", "z_model": "turbo Q8", "z_te": "official",
    "z_quant": "nf4", "mode": "Speed", "prompt": "", "negative": NEG_SDXL, "style": "Photo",
    "aspect": "Portrait 896x1152", "count": 1, "seed": -1, "last_seed": None, "use_adv": False, "steps": 20,
    "cfg": 5.0, "sampler": "dpmpp_2m_karras", "clip_skip": 2, "loras": [], "image": None, "info": "",
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


def autosave_loop() -> None:
    while True:
        time.sleep(5)
        for code, s in list(STATES.items()):
            blob = json.dumps(s, default=str, sort_keys=True)
            if _saved.get(code) != blob:
                session.save(code, s)
                _saved[code] = blob


USAGE: dict = {}
OPEN_JOB: dict[str, str] = {}  # browser tab id -> job whose log Activity should open


def sample_usage_loop() -> None:
    """NVML, psutil and folder sizes are slow: measure here, never in the event loop."""
    while True:
        try:
            USAGE.update(resources.usage(eng, jm))
        except Exception as e:
            log.info(f"usage sample failed: {e}")
        time.sleep(2)


# ---------- names and files ----------
def lora_names():
    return [p.name for p in list_loras()]


def lora_path(name):
    return next((str(p) for p in list_loras() if p.name == name), None)


def model_names():
    return [p.name for p in list_models()]


def model_path(name):
    return next((str(p) for p in list_models() if p.name == name), name)


def z_names():
    return [*Z_MODELS, *(p.name for p in list_zimage())]


def z_path(name):
    return Z_MODELS.get(name) or next((str(p) for p in list_zimage() if p.name == name), name)


def datasets():
    return sorted(p.name for p in DATASETS.iterdir() if p.is_dir()) if DATASETS.exists() else []


def images(ds):
    d = DATASETS / ds if ds else None
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG) if d and d.exists() else []


def url(p) -> str:
    """Browser URL of a file under WORK (served at /work)."""
    return f"/work/{Path(p).relative_to(WORK).as_posix()}?v={int(Path(p).stat().st_mtime)}"


def thumb(p: Path, size: int = 320) -> str:
    t = THUMBS / Path(p).relative_to(WORK).with_suffix(f".{size}.jpg")
    if not t.exists() or t.stat().st_mtime < Path(p).stat().st_mtime:
        t.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(p).convert("RGB")
        im.thumbnail((size, size))
        im.save(t, quality=85)
    return url(t)


def size_text(n: int) -> str:
    return f"{n / 2**30:.2f} GB" if n >= 2**30 else f"{n / 2**20:.0f} MB" if n >= 2**20 else f"{max(n, 1) / 2**10:.0f} KB"


def err(e: Exception) -> str:
    return str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"


def logs_zip() -> Path:
    out = WORK / "sdxlr_logs.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in LOGS.rglob("*.log"):
            z.write(f, f.relative_to(LOGS))
    return out


# ---------- model ----------
def wanted(S) -> str:
    if S["kind"] == "Z-Image":
        return f"zimage|{z_path(S['z_model'])}|{Z_ENCODERS.get(S['z_te'], S['z_te'])}|{S['z_quant']}"
    return model_path(S["sdxl_model"]) if S["sdxl_model"] else ""


def is_loaded(S) -> bool:
    return eng.pipe is not None and eng.model_path == wanted(S) and (S["kind"] == "Z-Image" or eng.vae_mode == S["vae"])


def picked_label(S) -> str:
    if S["kind"] == "Z-Image":
        return f"Z-Image {S['z_model']}"
    return f"SDXL {Path(S['sdxl_model']).stem}" if S["sdxl_model"] else "SDXL (no checkpoint picked)"


def loaded_label() -> str | None:
    if eng.kind == "zimage":
        return "Z-Image " + Path(eng.model_path.split("|")[1]).stem
    if eng.kind == "sdxl":
        return "SDXL " + Path(eng.model_path).stem
    return None


def check_model(S) -> None:
    if S["kind"] == "SDXL" and not S["sdxl_model"]:
        raise ValueError("Pick an SDXL checkpoint first (add one in Library)")


def busy_task():
    return next((t for t in act.running() if t.kind in ("load", "generate")), None)


def running_job():
    return next((j for j in jm.recent(30) if j.status == "running"), None)


def _prefetch(task, *args) -> None:
    """Download what a load needs in a child process, so Cancel can stop it."""
    task.set_stage("Checking files")
    task.proc = subprocess.Popen([sys.executable, "-u", "-m", "sdxlr.fetch", *args], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    tail = []
    for line in task.proc.stdout:
        tail = (tail + [line.rstrip()])[-5:]
        if line.startswith(("model file", "text encoder", "configs")):
            task.stage = {"m": "Got the model file", "t": "Got the text encoder", "c": "Got VAE and configs"}[line[0]]
    rc = task.proc.wait()
    task.proc = None
    task.check()
    if rc:
        raise RuntimeError("download failed: " + " | ".join(tail))


def _load(S, task) -> None:
    """Load the model picked in S (caller holds gpu_lock)."""
    if is_loaded(S):
        return
    check_model(S)
    if S["kind"] == "Z-Image":
        _prefetch(task, "prefetch-zimage", z_path(S["z_model"]), Z_ENCODERS.get(S["z_te"], S["z_te"]))
        eng.load_zimage(z_path(S["z_model"]), Z_ENCODERS.get(S["z_te"], S["z_te"]), S["z_quant"], status=task.set_stage)
    else:
        if S["vae"] == "fp16fix":
            _prefetch(task, "prefetch-sdxl")
        eng.load(model_path(S["sdxl_model"]), S["vae"], status=task.set_stage)


def _gpu_free() -> None:
    if jm.gpu_busy() or not gpu_lock.acquire(blocking=False):
        j = running_job()
        raise ValueError(f"The GPU is busy with \"{j.title if j else 'a job'}\". Try again when it finishes.")


def load_blocking(S, task) -> str:
    _gpu_free()
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


def check_generate(S) -> None:
    if not S["prompt"].strip():
        raise ValueError("Write a prompt first")
    check_model(S)
    if S["edit"] and not S["boxes"]:
        raise ValueError("Drag on the image to mark the area to change first")


def generate_blocking(S, task) -> tuple[list[str], str, int]:
    check_generate(S)
    _gpu_free()
    try:
        _load(S, task)
        kind = eng.kind
        m = MODES[kind][S["mode"]]
        steps, cfg, sampler = (S["steps"], S["cfg"], S["sampler"]) if S["use_adv"] else \
            (m["steps"], m["cfg"], m.get("sampler", S["sampler"]))
        loras = [(lora_path(n), float(w)) for n, w in S["loras"] if n and w]
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
        if S["edit"] and S["edit_src"]:
            image = Image.open(S["edit_src"]).convert("RGB")
            mask = _mask(image.size, S["boxes"])
        job = Job(prompt=full, negative=neg, steps=int(steps), cfg=float(cfg), sampler=sampler, width=w, height=h,
                  seed=int(S["seed"]), count=int(S["count"]), clip_skip=int(S["clip_skip"]), loras=loras,
                  image=image, mask=mask, strength=float(S["strength"]), on_step=task.step)
        task.set_stage("Changing the area" if image else "Generating")
        t0 = time.time()
        imgs, seed, info = eng.generate(job)
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
    for i, im in enumerate(imgs):
        p = day / f"{time.strftime('%H%M%S')}_{seed + i}.png"
        im.save(p, pnginfo=png)
        history.save(p, {**meta, "seed": seed + i})
        paths.append(str(p))
    return paths, info, seed


def reuse(S, meta: dict) -> list[str]:
    """Fill the form from a history JSON so the same image can be made again; returns missing LoRAs."""
    for k in ("negative", "style", "mode", "steps", "cfg", "sampler", "clip_skip", "aspect", "seed", "kind",
              "sdxl_model", "vae", "z_model", "z_te", "z_quant", "strength"):
        if k in meta and meta[k] is not None:
            S[k] = meta[k]
    S["prompt"] = meta.get("prompt_raw", meta.get("prompt", ""))
    if "negative_raw" in meta:
        S["negative"] = meta["negative_raw"]
    S["loras"] = [list(x) for x in meta.get("loras", [])]
    S["use_adv"] = True  # same steps/cfg/sampler as recorded
    S["count"], S["edit"] = 1, False
    return [n for n, _ in S["loras"] if n not in lora_names()]


# ---------- jobs (child processes) ----------
def _no_duplicate(kind: str, title: str) -> None:
    if any(j.kind == kind and j.title == title and j.status in ("queued", "running") for j in jm.jobs.values()):
        raise ValueError(f"\"{title}\" is already queued or running")


def submit_import(S, staged: list[str]):
    name = S["ds_name"].strip()
    if not name:
        raise ValueError("Give the dataset a name")
    srcs = staged + [s.strip() for s in S["ds_link"].replace("\n", ",").split(",") if s.strip()]
    if not srcs:
        raise ValueError("Drop photos or a zip, or paste a link first")
    title = f"Import into {name}"
    _no_duplicate("import", title)
    return jm.submit("import", title, ["-m", "sdxlr.data", "--dest", DATASETS / name, *srcs], gpu=False)


def submit_caption(S):
    if not S["ds_pick"]:
        raise ValueError("Pick a dataset first")
    if not images(S["ds_pick"]):
        raise ValueError("This dataset has no photos yet")
    title = f"Write captions for {S['ds_pick']}"
    _no_duplicate("caption", title)
    args = ["-m", "sdxlr.caption", "--images", DATASETS / S["ds_pick"], "--model", S["cap_model"],
            "--trigger", S["cap_trigger"], "--class-word", S["cap_class"], "--limit", int(S["cap_limit"] or 0)]
    if S["cap_prompt"].strip():
        args += ["--prompt", S["cap_prompt"]]
    if S["cap_over"]:
        args += ["--overwrite"]
    return jm.submit("caption", title, args, gpu=True)


def train_estimate(S) -> tuple[str, str]:
    """(headline, detail)"""
    n = len(images(S["t_ds"]))
    if not S["t_ds"]:
        return "Pick a dataset", "The time estimate shows here."
    if not n:
        return "No photos in this dataset", "Add photos in Dataset first."
    k = "zimage" if S["t_kind"] == "Z-Image" else "sdxl"
    b = 1 if k == "zimage" else int(S["t_batch"] or 1)
    steps, secs, est = train_eta(k, n, int(S["t_repeats"] or 1), int(S["t_epochs"] or 1), b,
                                 int(S["t_res"]) if k == "zimage" else 1024)
    detail = f"{steps} steps for {n} photos{' (estimate)' if est else ''}"
    if secs > 4.5 * 3600:
        detail += ". Longer than a free T4 session (about 5 h): lower the epochs or repeats, or turn on auto resume"
    return f"About {fmt_secs(secs)}", detail


def submit_train(S):
    ds = S["t_ds"]
    name = S["t_name"].strip()
    if not ds:
        raise ValueError("Pick a dataset first")
    if not (images(ds) or S["t_resume"]):
        raise ValueError("This dataset has no photos yet")
    if not name:
        raise ValueError("Give the LoRA a name")
    if S["t_resume"] and not S["t_hf"].strip():
        raise ValueError("Auto resume needs a private HF repo")
    title = f"Train {S['t_kind']} LoRA {name}"
    _no_duplicate("train", title)
    if S["t_kind"] == "Z-Image":
        args = ["-m", "sdxlr.train_zimage", "--name", name, "--images", DATASETS / ds, "--set",
                f"resolution={int(S['t_res'])}", f"epochs={int(S['t_epochs'])}", f"repeats={int(S['t_repeats'])}",
                f"network_dim={int(S['t_dim'])}", f"network_alpha={int(S['t_dim'])}", f"lr={S['t_lr']}"]
    else:
        args = ["-m", "sdxlr.train", "--name", name, "--images", DATASETS / ds,
                "--model", model_path(S["t_base"]) if S["t_base"] else SDXL_BASE,
                "--set", f"epochs={int(S['t_epochs'])}", f"repeats={int(S['t_repeats'])}",
                f"batch_size={int(S['t_batch'])}", f"network_dim={int(S['t_dim'])}",
                f"network_alpha={max(1, int(S['t_dim']) * 2 // 3)}", f"unet_lr={S['t_lr']}",
                f"text_encoder_lr={S['t_te_lr']}", f"lora_type={S['t_type']}"]
    if S["t_hf"].strip():
        args += ["--hf-repo", S["t_hf"].strip()]
    if S["t_resume"]:
        args += ["--resume"]
    return jm.submit("train", title, args, gpu=True)


def submit_compare(S):
    name = S["c_name"].strip()
    if not name:
        raise ValueError("Give the grid a name")
    try:
        strengths = [float(x) for x in S["c_str"].replace(",", " ").split()]
    except ValueError:
        raise ValueError("Strengths must be numbers, like 0.6 0.8 1.0") from None
    if not strengths:
        raise ValueError("Add at least one strength, like 0.8")
    mixes = [c.strip() for c in S["c_combos"].split(",") if c.strip()]
    if not mixes and not S["c_loras"]:
        raise ValueError("Pick LoRAs or write mixes")
    if S["c_kind"] == "SDXL" and not S["c_base"]:
        raise ValueError("Pick an SDXL checkpoint")
    if not S["c_prompt"].strip():
        raise ValueError("Write a prompt")
    title = f"Compare {name}"
    _no_duplicate("compare", title)
    args = ["-m", "sdxlr.compare", "--name", name, "--strengths", *map(str, strengths), "--prompt", S["c_prompt"],
            "--negative", S["c_neg"], "--steps", int(S["c_steps"]), "--cfg", S["c_cfg"], "--seed", int(S["c_seed"])]
    args += ["--zimage", "--model", z_path(S["c_z"])] if S["c_kind"] == "Z-Image" else ["--model", model_path(S["c_base"])]
    args += ["--combos", *(mixes or [f"{n}:1.0" for n in S["c_loras"]])]
    return jm.submit("compare", title, args, gpu=True)


def job_error(j) -> str:
    """The most useful line of a failed job's output, in plain words where we can."""
    lines = [l for l in j.log if l.strip()]
    for l in reversed(lines):
        if "Error" in l or "error" in l or "failed" in l:
            if "ModuleNotFoundError" in l:
                return "A package is missing. Run the Setup cell in the notebook again, then retry. (" + l.split(":")[-1].strip() + ")"
            if "OutOfMemoryError" in l or "out of memory" in l.lower():
                return "The GPU ran out of memory: lower the batch, resolution or image count"
            return l.strip()[-200:]
    return lines[-1][-200:] if lines else f"exit code {j.returncode}"


def job_last_line(j) -> str:
    """What a running job printed last (downloads and caching print no step counter)."""
    for l in reversed(j.log):
        if l.strip():
            return l.strip()[-120:]
    return "Starting"


def estimate_preview(S):
    """VRAM a picked-but-not-loaded model would take, for the hatched block in the sidebar bar."""
    if is_loaded(S):
        return None
    if S["kind"] == "Z-Image":
        zp = z_path(S["z_model"])
        local = MODELS / "zimage" / Path(zp).name if zp in Z_MODELS.values() else Path(zp)
        return resources.estimate("Z-Image", None, str(local), S["z_quant"])
    return resources.estimate("SDXL", model_path(S["sdxl_model"]), None, S["z_quant"]) if S["sdxl_model"] else None


__all__ = ["ASPECTS", "STYLES", "fmt_secs"]
