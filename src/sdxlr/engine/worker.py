"""Engine worker process: one request at a time, events back, cancel checks.

Only this process imports torch (via engine.models, inside main()). The web
process starts it through engine.client and never imports this module — the
spawn entry in client.py imports it in the child.

Protocol (dicts over a multiprocessing Pipe; images travel as file paths):
  request  {"id", "op": "load"|"unload"|"status"|"generate"|"upscale"|"shutdown", "args"}
  event    {"id", "type": "stage"|"step"|"beat"|"result"|"cancelled"|"error", ...}

Cancel: the web side sets the cancel event; ops raise Cancelled at stage
boundaries and in the denoise step callback. Model loads are not
interruptible — the client kills the worker for those and restarts it later.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

from PIL import Image

from ..paths import OUTPUTS, PREVIEWS, RUN

GB = 2**30
BEAT_SECS = float(os.environ.get("SDXLR_BEAT_SECS", "2"))  # tests shorten it
_preview = [None]  # on_preview(step, latents) for the current request; set in main()
_models = None  # the engine module: models (torch), or fake (stub) when SDXLR_FAKE_ENGINE is set; picked in main()


class Cancelled(Exception):
    """Raised inside the worker when the web side asked to stop."""


class UserError(Exception):
    """A problem with the request itself: its message is already a plain sentence for the user."""


def oom_score(val: int) -> None:
    """Raise this process's OOM score so Linux kills it before the web app."""
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(str(val))
    except OSError:
        pass  # not Linux, or no permission; the RAM guard still works without it


def model_parts(pipe, adapters) -> list[tuple[str, float]]:
    """[(part, GB)] of the loaded pipeline; LoRA weights counted separately."""
    def size(module) -> int:
        seen, n = set(), 0
        for t in [*module.parameters(), *module.buffers()]:
            if t.data_ptr() not in seen and t.device.type == "cuda":
                seen.add(t.data_ptr())
                n += t.numel() * t.element_size()
        return n

    names = {"unet": "UNet", "transformer": "DiT", "text_encoder": "text encoder",
             "text_encoder_2": "text encoder 2", "vae": "VAE"}
    out, lora = [], 0
    for attr, label in names.items():
        m = getattr(pipe, attr, None)
        if m is None:
            continue
        lb = sum(p.numel() * p.element_size() for n, p in m.named_parameters()
                 if "lora_" in n and p.device.type == "cuda")
        lora += lb
        out.append((label, (size(m) - lb) / GB))
    if lora:
        out.append((f"LoRAs ({len(adapters)})", lora / GB))
    return out


def work_copy(img, multiple: int, max_mp: float = 1.2, max_side: int = 2048):
    """A capped RGB work copy of a source image: (image, fx, fy).

    Big uploads would otherwise spike RAM and VRAM (the first img2img test
    froze the VM). The original file is never touched.
    """
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    f = min(1.0, (max_mp * 1e6 / (w * h)) ** 0.5 if w * h > max_mp * 1e6 else 1.0,
            max_side / max(w, h) if max(w, h) > max_side else 1.0)
    if f >= 1.0:
        return img, 1.0, 1.0
    w2, h2 = max(multiple, int(w * f) // multiple * multiple), max(multiple, int(h * f) // multiple * multiple)
    return img.resize((w2, h2), Image.LANCZOS), w2 / w, h2 / h


def save_images(images, meta: dict, seed: int) -> list[str]:
    """PNG with the sdxlr text chunk + a JSON sidecar (the History format)."""
    from PIL.PngImagePlugin import PngInfo
    day = OUTPUTS / time.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, im in enumerate(images):
        m = {**meta, "seed": seed + i, "size": list(im.size), "made": time.strftime("%Y-%m-%d %H:%M:%S")}
        p = day / f"{time.strftime('%H%M%S')}_{seed + i}.png"
        k = 1
        while p.exists():  # same second and seed happen in batches
            k += 1
            p = day / f"{time.strftime('%H%M%S')}_{seed + i}_{k}.png"
        png = PngInfo()
        png.add_text("sdxlr", json.dumps(m))
        im.save(p, pnginfo=png)
        try:  # the page shows this smaller copy; Download gives the PNG
            im.convert("RGB").save(p.with_suffix(".webp"), "WEBP", quality=90, method=4)
        except Exception:
            pass
        p.with_suffix(".json").write_text(json.dumps(m, indent=1), encoding="utf-8")
        paths.append(str(p))
    return paths


def open_source(path: str, boxes: list, multiple: int):
    """A source image as a capped work copy; box marks (original coordinates)
    become an L mask at work-copy size. Returns (image, mask | None)."""
    from PIL import Image, ImageDraw
    img, fx, fy = work_copy(Image.open(path), multiple)
    if not boxes:
        return img, None
    m = Image.new("L", img.size, 0)
    d = ImageDraw.Draw(m)
    for x0, y0, x1, y1 in boxes:
        d.rectangle([min(x0, x1) * fx, min(y0, y1) * fy, max(x0, x1) * fx, max(y0, y1) * fy], fill=255)
    return img, m


# ---- ops: (engine, args, callbacks) -> result dict ----

def _op_status(eng, args, status, step, check):
    return {"kind": eng.kind, "model_path": eng.model_path, "vae_mode": eng.vae_mode}


def _op_unload(eng, args, status, step, check):
    eng.unload()
    return {}


def _op_load(eng, args, status, step, check):
    if args.get("kind") == "zimage":
        label = eng.load_zimage(args["transformer"], args["text_encoder"], args["te_quant"], status=status)
    else:
        label = eng.load(args["path"], args.get("vae", "fp16fix"), status=status)
    return {"label": label}


def _job_from(eng, j: dict, step) -> "object":
    models = _models
    image = mask = None
    if j.get("image"):
        image, mask = open_source(j["image"], j.get("boxes") or [], 16 if eng.kind == "zimage" else 8)
    w, h = image.size if image is not None else (int(j["width"]), int(j["height"]))
    return models.Job(prompt=j["prompt"], negative=j.get("negative", ""), steps=int(j["steps"]),
                      cfg=float(j["cfg"]), width=w, height=h, seed=int(j.get("seed", -1)),
                      count=int(j.get("count", 1)), sampler=j.get("sampler", "dpmpp_2m_karras"),
                      clip_skip=int(j.get("clip_skip", 2)),
                      loras=[(p, float(v)) for p, v in j.get("loras") or []],
                      image=image, mask=mask, strength=float(j.get("strength", 0.75)), on_step=step,
                      on_preview=_preview[0])


def _plain_job(j: dict, seed: int, step, image=None, count: int = 1, strength: float | None = None) -> "object":
    """A Job built by hand: the pipelines get called by the caller (inpaint_crop / generate)."""
    models = _models
    w, h = image.size if image is not None else (int(j.get("width", 1024)), int(j.get("height", 1024)))  # edits take their size from the crop
    return models.Job(prompt=j["prompt"], negative=j.get("negative", ""), steps=int(j["steps"]),
                      cfg=float(j["cfg"]), width=w, height=h, seed=seed, count=count,
                      sampler=j.get("sampler", "dpmpp_2m_karras"), clip_skip=int(j.get("clip_skip", 2)),
                      loras=[(p, float(v)) for p, v in j.get("loras") or []], image=image,
                      strength=float(j.get("strength", 0.75) if strength is None else strength), on_step=step,
                      on_preview=_preview[0])


def save_edit_masks(paths: list[str], masks: list) -> None:
    """The mask next to each result, so 'edit again' can reuse it."""
    for p, m in zip(paths, masks):
        try:
            m.convert("L").save(p[:-4] + "_mask.png")
        except Exception:
            pass


def _edit_images(eng, args, step, check, status=None, label="", on_image=None):
    """Crop-and-stitch edits: area (painted mask or boxes) and extend (outpaint).
    on_image(img, mask, seed) is called as each variation finishes, so a cancel keeps the finished ones."""
    import random

    from . import edit
    j = args["job"]
    tool = j.get("tool")
    multiple = 16 if eng.kind == "zimage" else 8
    base, _fx, _fy = work_copy(edit.load_source(j["image"]), multiple)
    if tool == "extend":
        base, mask = edit.outpaint_canvas(base, tuple(j.get("extend") or (0, 0, 0, 0)), multiple)
    elif j.get("mask"):
        try:
            mask = edit.mask_from_painted(j["mask"], base.size)
        except ValueError as e:
            raise UserError(str(e)) from None
        except OSError:
            raise UserError("The painted mask is gone; paint it again.") from None
    else:
        mask = edit.mask_from_boxes(base.size, j.get("boxes") or [])
    if edit.mask_bbox(mask) is None:
        raise UserError("Nothing is painted. Mark the part to change first.")
    regions = [mask] if tool == "extend" else edit.split_regions(mask) or [mask]  # new space on two sides is one job
    count = max(1, min(4, int(j.get("count", 1))))
    seed = int(j.get("seed", -1)) if int(j.get("seed", -1)) >= 0 else random.randrange(2**31 - 1)
    feather = int(j.get("feather", 12))
    outs, masks = [], []
    for c in range(count):
        if status and count > 1:
            status(f"{label}, variation {c + 1} of {count}")
        img = base.copy()
        for rm in regions:
            plan = edit.prepare(img, rm, multiple, feather_px=feather, fill=tool != "extend")
            check()
            job = _plain_job(j, seed + c, step, strength=1.0 if tool == "extend" else None)  # a new area is repainted fully
            out = eng.inpaint_crop(job, plan.crop, plan.crop_mask)
            img = edit.paste_back(img, out, plan)
            check()
        outs.append(img)
        masks.append(mask)
        if on_image:
            on_image(img, mask, seed + c)
    return outs, masks, seed, f"{len(outs)} image(s) from {len(regions)} region(s)"


def _op_generate(eng, args, status, step, check):
    from . import edit
    t0 = time.time()
    j = args["job"]
    tool = j.get("tool")
    if j.get("blend", "plain") != "plain":  # the ControlNet blend comes with the T4 work
        raise UserError("That blend mode is not available yet. Use the plain one.")
    if tool in ("area", "extend"):
        paths = []

        def keep(img, mask, seed_c):  # saved at once: History keeps the finished variations if the next is cancelled
            meta_c = {**(args.get("meta") or {}), "seconds": round(time.time() - t0, 1)}
            p = save_images([img], meta_c, seed_c)
            save_edit_masks(p, [mask])
            paths.extend(p)

        label = "Extending the image" if tool == "extend" else "Changing the painted part"
        images, masks, seed, note = _edit_images(eng, args, step, check, status, label, keep)
        meta = {"seconds": round(time.time() - t0, 1)}
        return {"paths": paths, "seed": seed, "info": f"{meta['seconds']}s | {note} | seed {seed}",
                "size": list(images[0].size)}
    if tool == "whole":
        multiple = 16 if eng.kind == "zimage" else 8
        original = edit.load_source(j["image"])
        base, _, _ = work_copy(original, multiple)
        check()
        seed = int(j.get("seed", -1))
        if seed < 0:
            import random
            seed = random.randrange(2**31 - 1)
        images, paths, info = [], [], ""
        n = max(1, min(4, int(j.get("count", 1))))
        for c in range(n):  # variations run one after another, each saved as it finishes
            check()
            if n > 1:
                status(f"Changing the image, variation {c + 1} of {n}")
            ims, _, info = eng.generate(_plain_job(j, seed + c, step, image=base, count=1))
            if j.get("keep_size"):
                ims = [im.resize(original.size, Image.LANCZOS) for im in ims]
            meta = {**(args.get("meta") or {}), "seconds": round(time.time() - t0, 1)}
            paths += save_images(ims, meta, seed + c)
            images += ims
        return {"paths": paths, "seed": seed, "info": info, "size": list(images[0].size)}
    job = _job_from(eng, j, step)
    check()
    images, seed, info = eng.generate(job)
    check()
    t1 = time.time()
    meta = {**(args.get("meta") or {}), "seconds": round(t1 - t0, 1)}
    paths = save_images(images, meta, seed)
    return {"paths": paths, "seed": seed, "info": info + f" | saved {time.time() - t1:.1f}s",
            "size": list(images[0].size)}


def _op_upscale(eng, args, status, step, check):
    t0 = time.time()
    job = _job_from(eng, args["job"], step)
    check()
    big, seed, info = eng.upscale(job, float(args.get("scale", 2.0)), on_tile=step)
    t1 = time.time()
    meta = {**(args.get("meta") or {}), "seconds": round(t1 - t0, 1)}
    paths = save_images([big], meta, seed)
    return {"paths": paths, "seed": seed, "info": info + f" | saved {time.time() - t1:.1f}s",
            "size": list(big.size)}


OPS = {"status": _op_status, "unload": _op_unload, "load": _op_load,
       "generate": _op_generate, "upscale": _op_upscale}


def main(conn, cancel_event, parent_pid: int) -> None:
    global _models
    oom_score(1000)  # the web app survives RAM trouble; this process is rebuilt for free
    RUN.mkdir(parents=True, exist_ok=True)
    pid_f = RUN / "worker.pid"
    ctime = 0.0
    try:
        import psutil
        ctime = psutil.Process().create_time()
    except Exception:
        pass
    pid_f.write_text(f"{os.getpid()} {ctime:.2f}")  # the create time tells a stale worker from a reused pid
    if os.environ.get("SDXLR_FAKE_ENGINE"):
        from . import fake as models
    else:
        from . import models
    _models = models
    eng = models.Engine()
    send_lock = threading.Lock()
    cur = {"id": None}

    def state() -> dict:
        return {"kind": eng.kind, "model_path": eng.model_path, "vae_mode": eng.vae_mode,
                "adapters": dict(eng.adapters)}

    def send(msg: dict, stamp: str | None = None) -> None:
        with send_lock:  # the beat thread and the op handler share the pipe
            # the state is read here, under the lock: a beat can never carry a state older than a result sent before it
            if stamp == "beat":
                msg.update(state())
            elif stamp == "result":
                msg["result"] = {**msg["result"], "state": state()}
            try:
                conn.send(msg)
            except OSError:
                pass  # the web side went away; the orphan check ends us

    def check():
        if cancel_event.is_set():
            raise Cancelled

    def on_preview(i: int, latents) -> None:
        """Every 2nd step: the engine decodes a small preview; the event goes out once the JPEG is written."""
        if not hasattr(eng, "make_preview") or eng.pipe is None:
            return
        rid = cur["id"]
        path = PREVIEWS / f"{rid}.jpg"
        eng.make_preview(latents, path, lambda p, rid=rid, i=i: send({"type": "preview", "id": rid, "path": str(p), "step": i}))

    _preview[0] = on_preview
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    for old in PREVIEWS.glob("*"):  # leftovers of an earlier worker
        old.unlink(missing_ok=True)

    def status(text: str):
        check()
        send({"type": "stage", "id": cur["id"], "stage": text})

    def step(done: int, total: int):
        check()
        send({"type": "step", "id": cur["id"], "done": done, "total": total})

    def beat():
        rss = alloc = reserved = 0.0
        parts = []
        try:
            import psutil
            rss = psutil.Process().memory_info().rss / GB
        except Exception:
            pass
        if eng.pipe is not None:
            parts = model_parts(eng.pipe, eng.adapters)
        if "torch" in sys.modules:  # never import it here: a stub engine must stay torch-free
            try:
                torch = sys.modules["torch"]
                if torch.cuda.is_available():
                    alloc, reserved = torch.cuda.memory_allocated() / GB, torch.cuda.memory_reserved() / GB
            except Exception:
                pass
        msg = {"type": "beat", "parts": parts, "alloc": alloc, "reserved": reserved, "rss": rss}
        if hasattr(eng, "beat_extra"):
            msg.update(eng.beat_extra())
        send(msg, stamp="beat")

    def beat_loop():
        last = None
        while True:
            time.sleep(BEAT_SECS)
            try:
                beat()
            except Exception as e:  # keep beating; say each distinct problem once
                err = f"{type(e).__name__}: {e}"
                if err != last:
                    last = err
                    print(f"engine worker: heartbeat failed: {err}", file=sys.stderr, flush=True)

    def orphan_loop():
        while True:  # the web app is gone: holding the GPU helps nobody
            time.sleep(2)
            if os.getppid() != parent_pid:
                os._exit(1)

    threading.Thread(target=beat_loop, daemon=True).start()
    threading.Thread(target=orphan_loop, daemon=True).start()
    try:
        while True:
            try:
                req = conn.recv()
            except (EOFError, OSError):
                break
            if not isinstance(req, dict):
                continue
            cur["id"] = req.get("id")
            cancel_event.clear()
            try:
                check()
                if req.get("op") == "shutdown":
                    send({"type": "result", "id": cur["id"], "result": {}}, stamp="result")
                    break
                res = OPS[req.get("op")](eng, req.get("args") or {}, status, step, check)
                send({"type": "result", "id": cur["id"], "result": res}, stamp="result")
            except Cancelled:
                send({"type": "cancelled", "id": cur["id"]})
            except UserError as e:
                send({"type": "error", "id": cur["id"], "error": str(e), "plain": True})
            except Exception as e:
                import traceback
                send({"type": "error", "id": cur["id"], "error": f"{type(e).__name__}: {e}",
                      "tb": traceback.format_exc()[-4000:]})
            finally:
                (PREVIEWS / f"{cur['id']}.jpg").unlink(missing_ok=True)
    finally:
        pid_f.unlink(missing_ok=True)
