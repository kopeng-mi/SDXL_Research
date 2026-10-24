"""Shared state and helpers behind the UI (no layout here).

The work lives elsewhere: service.gen (load + generate through the engine
worker, engine.client), service.jobs (child-process job queue: caption,
train, compare, downloads) and service.files (names and listings). This
module wires them together, keeps the per-session form state and re-exports
what the pages use.

Every submit_* / *_blocking function checks its inputs first and raises
ValueError with a sentence the user can act on.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

from PIL import Image

from .. import history, resources
from ..activity import LOGS, log
from ..engine import SAMPLERS, Z_ENCODERS, Z_MODELS  # noqa: F401  (SAMPLERS used by pages)
from ..modes import ASPECTS, STYLES, train_eta  # noqa: F401  (ASPECTS/STYLES used by pages)
from ..paths import DATASETS, MODELS, OUTPUTS, SDXL_BASE, UPLOADS, WORK  # noqa: F401  (UPLOADS/DATASETS: pages)
from ..service import files
from ..service.files import (datasets, images, lora_names, lora_path, model_names, model_path, poke, thumb, url,  # noqa: F401
                             z_names, z_path)
from ..service.gen import (TOOL_LABEL, busy_task, check_generate, check_model, engc, generate_blocking,  # noqa: F401
                           is_loaded, load_blocking, loaded_label, picked_label, reuse, running_job, task_is_loading)
from ..service.jobs import fmt_secs, gpu_lock, jm  # noqa: F401
from ..service.state import STATES, state_for  # noqa: F401

USAGE: dict = {}
DISK = {"rate": 0.0}  # bytes/s the disk is filling: > 1 MB/s means something is downloading
OPEN_JOB: dict[str, str] = {}  # browser tab id -> job whose log Activity should open


def sample_usage_loop() -> None:
    """NVML, psutil and folder sizes are slow: measure here, never in the event loop."""
    import shutil
    root = "/content" if Path("/content").exists() else str(WORK.resolve().anchor)
    last, last_t = shutil.disk_usage(root).used, time.time()
    while True:
        try:
            USAGE.update(resources.usage(engc.beat, jm))
        except Exception as e:
            log.info(f"usage sample failed: {e}")
        time.sleep(2)
        used, now = shutil.disk_usage(root).used, time.time()
        DISK["rate"] = 0.6 * DISK["rate"] + 0.4 * max(used - last, 0) / (now - last_t)
        last, last_t = used, now


# ---------- files ----------
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


MAX_UPLOAD = 50 * 2**20  # Generate-page uploads; dataset photos go through their own importer


def save_upload(data: bytes, name: str) -> str:
    """An uploaded photo joins the outputs (and History) as a PNG, so the tools can work on it.
    The original file is never touched; generation works on a capped copy in the worker."""
    import io

    from PIL import UnidentifiedImageError

    from ..engine.edit import load_source
    if len(data) > MAX_UPLOAD:
        raise ValueError(f"{name} is {len(data) / 2**20:.0f} MB; the limit is 50 MB. Export it smaller first.")
    try:
        img = load_source(io.BytesIO(data))
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
        raise ValueError(f"{name} is not an image this app can read (JPEG, PNG, WebP). HEIC: export as JPEG first.") from None
    if min(img.size) < 256:
        raise ValueError(f"{name} is too small ({img.size[0]}x{img.size[1]} px); use one at least 256 px.")
    day = OUTPUTS / time.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(name).stem)[:40] or "image"
    p = day / f"{time.strftime('%H%M%S')}_upload_{stem}.png"
    k = 1
    while p.exists():  # same second and name
        k += 1
        p = day / f"{time.strftime('%H%M%S')}_upload_{stem}_{k}.png"
    img.save(p)
    history.save(p, {"uploaded": True, "name": name, "prompt_raw": "", "made": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "size": list(img.size)})
    poke()
    return str(p)


SHRED = {
    "images": ("Generated and uploaded images (History)", ["outputs", "thumbs", "compare"]),
    "loras": ("LoRAs, downloaded and trained", ["loras", "runs"]),
    "models": ("Checkpoints and Z-Image models", ["models"]),
    "datasets": ("Datasets (photos and captions)", ["datasets", "uploads"]),
    "cache": ("Download cache (downloads again when needed)", []),
}


def shred_sizes() -> dict[str, float]:
    from ..resources import HF_CACHE, _dir_gb
    return {k: sum(_dir_gb(WORK / d) for d in dirs) + (_dir_gb(HF_CACHE) if k == "cache" else 0)
            for k, (_, dirs) in SHRED.items()}


def shred(kinds: list[str]) -> float:
    """Delete everything of the picked kinds from the VM; returns GB freed. Stops the engine first."""
    import shutil
    from ..resources import HF_CACHE
    if jm.gpu_busy() or busy_task():
        raise ValueError("Wait until nothing is running, then try again")
    sizes = shred_sizes()
    with gpu_lock:
        engc.kill()
    for k in kinds:
        for d in SHRED[k][1]:
            shutil.rmtree(WORK / d, ignore_errors=True)
            (WORK / d).mkdir(parents=True, exist_ok=True)
        if k == "cache":
            shutil.rmtree(HF_CACHE / "hub", ignore_errors=True)
            shutil.rmtree(HF_CACHE / "xet", ignore_errors=True)
    files.forget_thumbs()
    poke()
    log.info(f"deleted from the VM: {', '.join(kinds)}")
    return sum(sizes[k] for k in kinds)


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
    """What a running job is doing: downloading (disk filling fast), or the last line it printed."""
    if DISK["rate"] > 2**20:
        return f"downloading model files, {DISK['rate'] / 2**20:.0f} MB/s (only the first time)"
    for l in reversed(j.log):
        if l.strip():
            return l.strip()[-120:]
    return "Starting"


def estimate_preview(S):
    """VRAM a picked-but-not-loaded model would take, for the hatched block in the sidebar bar."""
    if is_loaded(S):
        return None
    if S["kind"] == "Z-Image":
        return resources.estimate("Z-Image", None, None, S["z_quant"], z_gb=files.z_size_gb(S["z_model"]) or 7.2)
    return resources.estimate("SDXL", model_path(S["sdxl_model"]), None, S["z_quant"]) if S["sdxl_model"] else None


__all__ = ["ASPECTS", "STYLES", "fmt_secs"]
