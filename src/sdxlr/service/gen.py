"""UI state -> engine requests, and the checks in front of them.

Everything here runs in io_bound threads (never the event loop), with the
task (activity.Task) as the progress channel. Input problems raise ValueError
with a sentence the user can act on; engine deaths surface as a plain
WorkerDied message.

The engine itself lives behind engine.client (a worker process); this module
never touches torch.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

from PIL import Image

from .. import modes
from ..activity import act, log
from ..engine import client as engine_client
from ..fetch import LORAS, fetch
from ..paths import Z_ENCODERS, z_key
from . import masks
from .files import lora_path, model_path, z_path
from .jobs import gpu_lock, jm

engc = engine_client.make()  # the engine for the web process (fake when SDXLR_FAKE_ENGINE=1)

TOOL_LABEL = {None: "Generate", "area": "Change the painted part", "whole": "Change the whole image",
              "extend": "Extend the image", "upscale": "Upscale"}
EDIT_TOOLS = ("area", "whole", "extend")  # these make 1 to 4 variations


# ---------- model ----------
def wanted(S) -> str:
    if S["kind"] == "Z-Image":
        return z_key(z_path(S["z_model"]), Z_ENCODERS.get(S["z_te"], S["z_te"]), S["z_quant"])
    return model_path(S["sdxl_model"]) if S["sdxl_model"] else ""


def is_loaded(S) -> bool:
    return engc.loaded and engc.model_path == wanted(S) and (S["kind"] == "Z-Image" or engc.vae_mode == S["vae"])


def picked_label(S) -> str:
    if S["kind"] == "Z-Image":
        return f"Z-Image {S['z_model']}"
    return f"SDXL {Path(S['sdxl_model']).stem}" if S["sdxl_model"] else "SDXL (no checkpoint picked)"


def loaded_label() -> str | None:
    if engc.kind == "zimage":
        return "Z-Image " + Path(engc.model_path.split("|")[1]).stem
    if engc.kind == "sdxl":
        return "SDXL " + Path(engc.model_path).stem
    return None


def check_model(S) -> None:
    if S["kind"] == "SDXL" and not S["sdxl_model"]:
        raise ValueError("Pick an SDXL checkpoint first (add one in Library)")


def busy_task():
    return next((t for t in act.running() if t.kind in ("load", "generate")), None)


def task_is_loading(t) -> bool:
    """True while a load (or the load step of a generation) is going on."""
    return bool(t) and (t.kind == "load" or not t.stage.startswith(("Generating", "Changing", "Extending", "Upscaling", "Done")))


def running_job():
    return next((j for j in jm.recent(30) if j.status == "running"), None)


def before_gpu_job(job) -> None:
    """A caption/train/compare job needs the whole GPU: stop the engine worker
    for it (guaranteed free VRAM; the next Generate starts the worker again)."""
    gpu_lock.acquire()  # waits for a running generation to finish; the job's end hook releases it
    if engc.loaded:
        log.info(f"[job {job.id}] stopping the image engine to free the GPU")
    try:
        engc.kill()
    except Exception as e:  # the job still runs and releases the lock at its end
        log.info(f"[job {job.id}] stopping the image engine failed: {e}")


def _prefetch(task, *args) -> None:
    """Download what a load needs in a child process, so Cancel can stop it."""
    import os
    from .jobs import REPO
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPO / "src")}
    task.set_stage("Getting the model files")
    task.proc = subprocess.Popen([sys.executable, "-u", "-m", "sdxlr.fetch", *args], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, env=env)
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


def _check_ram(kind, sdxl_path, z_transformer, te_quant) -> None:
    """Refuse a load that cannot fit. If another model is in the worker, stop it and check again: its RAM
    counts against the check and the load would drop it anyway (a model switch otherwise keeps torch warm)."""
    from . import guard
    try:
        guard.check_ram(kind, sdxl_path, z_transformer, te_quant, engc.beat, jm)
    except ValueError:
        if not engc.loaded:
            raise
        engc.kill()
        guard.check_ram(kind, sdxl_path, z_transformer, te_quant, engc.beat, jm)


def _load(S, task) -> None:
    """Load the model picked in S (caller holds gpu_lock)."""
    if is_loaded(S):
        return
    check_model(S)
    if S["kind"] == "Z-Image":
        tr, te = z_path(S["z_model"]), Z_ENCODERS.get(S["z_te"], S["z_te"])
        if engc.needs_prefetch:  # the stub engine does not load real weights
            _check_ram("Z-Image", None, tr, S["z_quant"])
            _prefetch(task, "prefetch-zimage", tr, te)
        engc.call("load", {"kind": "zimage", "transformer": tr, "text_encoder": te, "te_quant": S["z_quant"]}, task)
    else:
        mp = model_path(S["sdxl_model"])
        if engc.needs_prefetch:
            _check_ram("SDXL", mp, None, S["z_quant"])
        if S["vae"] == "fp16fix" and engc.needs_prefetch:
            _prefetch(task, "prefetch-sdxl")
        engc.call("load", {"kind": "sdxl", "path": mp, "vae": S["vae"]}, task)


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


# ---------- generate ----------
def check_generate(S) -> None:
    if not S["prompt"].strip() and S["tool"] != "upscale":
        raise ValueError("Write a prompt first")
    check_model(S)
    if S["tool"] and not (S["src"] and Path(S["src"]).exists()):
        raise ValueError("Pick or upload an image first")
    if S["tool"] == "area":
        p = masks.path_for(S["mask_id"])
        if p is None or not p.exists():
            raise ValueError("Paint the part to change first")
    if S["tool"] == "extend" and not any(float(x or 0) > 0 for x in S["extend"]):
        raise ValueError("Pick how much to add on at least one side")


def generate_blocking(S, task) -> tuple[list[str], str, int]:
    check_generate(S)
    _gpu_free()
    try:
        _load(S, task)
        kind = "zimage" if S["kind"] == "Z-Image" else "sdxl"
        m = modes.MODES[kind][S["mode"]]
        steps, cfg, sampler = (S["steps"], S["cfg"], S["sampler"]) if S["use_adv"] else \
            (m["steps"], m["cfg"], m.get("sampler", S["sampler"]))
        loras = [(lora_path(n), max(-1.0, min(2.0, float(w)))) for n, w in S["loras"] if n and w]  # -1..2
        loras = [(p, w) for p, w in loras if p]
        if kind == "sdxl" and m.get("lora") and engc.needs_prefetch:
            task.set_stage("Getting the Fast mode LoRA")
            loras.append((str(fetch(m["lora"][0], LORAS)), m["lora"][1]))
        prompt = S["prompt"].strip()
        full = ", ".join(x for x in (prompt, modes.STYLES.get(S["style"], "")) if x)
        neg = S["negative"] if kind == "sdxl" else ""
        if kind == "sdxl" and "pony" in wanted(S).lower():
            full, neg = f"{modes.PONY_TAGS}, {full}", f"{modes.PONY_NEG}, {neg}"
        w, h = modes.ASPECTS[S["aspect"]]
        tool = S["tool"]
        strength = {"area": S["strength"], "whole": S["whole_strength"], "extend": 1.0,
                    "upscale": S["up_strength"]}.get(tool, 0.75)
        count = max(1, min(4, int(S["edit_count"]))) if tool in EDIT_TOOLS else 1 if tool else int(S["count"])
        if tool:
            image = Image.open(S["src"])  # size only; the worker opens it again for the work copy
            w, h = image.size
            image.close()
            # image-to-image runs only steps x strength of the schedule: keep at least 4 real steps
            steps = max(int(steps), math.ceil(4 / max(strength, 0.05)))
        job = {"prompt": full, "negative": neg, "steps": int(steps), "cfg": float(cfg), "sampler": sampler,
               "width": w, "height": h, "seed": int(S["seed"]), "count": count, "tool": tool, "blend": "plain",
               "feather": int(S["feather"]), "keep_size": bool(S["keep_size"]),
               "extend": [max(0.0, min(100.0, float(x or 0))) for x in S["extend"]] if tool == "extend" else None,
               "mask": str(masks.path_for(S["mask_id"])) if tool == "area" else None,
               "clip_skip": int(S["clip_skip"]), "loras": loras, "strength": float(strength),
               "image": S["src"] if tool else None, "boxes": []}
        meta = {"prompt_raw": prompt, "prompt": full, "negative_raw": S["negative"], "negative": neg,
                "style": S["style"], "mode": S["mode"], "steps": int(steps), "cfg": float(cfg), "sampler": sampler,
                "clip_skip": int(S["clip_skip"]), "aspect": S["aspect"], "count": count,
                "loras": [[n, float(x)] for n, x in S["loras"] if n and x],
                "kind": S["kind"], "sdxl_model": S["sdxl_model"], "vae": S["vae"], "z_model": S["z_model"],
                "z_te": S["z_te"], "z_quant": S["z_quant"], "tool": tool, "source": S["src"] if tool else None,
                "strength": float(strength), "scale": S["up_scale"] if tool == "upscale" else None,
                "feather": int(S["feather"]) if tool == "area" else None,
                "extend": job["extend"], "keep_size": bool(S["keep_size"]) if tool == "whole" else None}
        if tool == "upscale":
            task.set_stage(f"Upscaling {S['up_scale']:g}x")
            r = engc.call("upscale", {"job": job, "scale": float(S["up_scale"]), "meta": meta}, task)
        else:
            task.set_stage({"area": "Changing the painted part", "whole": "Changing the image",
                            "extend": "Extending the image"}.get(tool, "Generating"))
            r = engc.call("generate", {"job": job, "meta": meta}, task)
    finally:
        gpu_lock.release()
    return r["paths"], r["info"], r["seed"]


def reuse(S, meta: dict) -> list[str]:
    """Fill the form from a history JSON so the same image can be made again; returns missing LoRAs."""
    from .files import lora_names
    for k in ("negative", "style", "mode", "steps", "cfg", "sampler", "clip_skip", "aspect", "seed", "kind",
              "sdxl_model", "vae", "z_model", "z_te", "z_quant", "strength"):
        if k in meta and meta[k] is not None:
            S[k] = meta[k]
    S["prompt"] = meta.get("prompt_raw", meta.get("prompt", ""))
    if "negative_raw" in meta:
        S["negative"] = meta["negative_raw"]
    S["loras"] = [list(x) for x in meta.get("loras", [])]
    S["use_adv"] = True  # same steps/cfg/sampler as recorded
    S["count"], S["tool"] = 1, None
    return [n for n, _ in S["loras"] if n not in lora_names()]
