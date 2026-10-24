"""One T4 session that checks the step4 engine end to end, through the same service calls the UI makes.

Run in the Colab venv (Setup cell done, app NOT running: this script owns the GPU):

  SDXLR_DEBUG_OPS=1 /content/venv/bin/python tools/t4_check.py --images /content/t4_imgs
      [--sections loads,te,edits,big,sdxl,hog] [--sdxl base|/path/model.safetensors] [--loras a.safetensors,b.safetensors]

Sections (default: all but sdxl unless --sdxl is given; hog last, it ends with the guard killing the worker):
  loads  Z-Image Q8 / Q6 / Q4: load time, peak worker RAM and VRAM, Speed-mode image at 896x1152, first preview step
  te     text encoder official nf4 vs unsloth 4-bit: load time, peak RAM, same-seed image difference
  edits  per image (max 2): painted part (plain, better blend), extend 25 %, whole, upscale 2x (plain, tile model);
         pixels outside the grown + feathered mask must be identical to the work copy
  big    a synthetic 6000x4000 photo through a painted edit (RAM must stay flat)
  sdxl   the same Speed image + a painted edit with an SDXL checkpoint, with 0 / 1 / 3 LoRAs (timings)
  hog    the worker eats RAM until the guard kills it: the script (the "web process") must survive and the next
         Generate must work

Writes WORK/t4_check/report.md, report.json and sheet.jpg (a contact sheet of every result).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

from sdxlr.activity import Task  # noqa: E402
from sdxlr.engine import edit  # noqa: E402
from sdxlr.engine.worker import work_copy  # noqa: E402
from sdxlr.paths import LORAS, OUTPUTS, WORK  # noqa: E402
from sdxlr.service import files, guard, masks  # noqa: E402
from sdxlr.service.gen import engc, generate_blocking, load_blocking  # noqa: E402
from sdxlr.service.jobs import jm  # noqa: E402
from sdxlr.service.state import State  # noqa: E402

OUT = WORK / "t4_check"
CODE = "9999"  # session code for the masks
PROMPT = "portrait photo of a woman in a red coat on a city street, natural light"
REPORT: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "sections": {}}
RESULTS: list[tuple[str, str]] = []  # (label, image path) for the contact sheet


# ---------- helpers ----------
class Peaks:
    """Samples the worker heartbeat and MemAvailable while a step runs."""

    def __init__(self, task: Task | None = None) -> None:
        self.task, self.stop = task, threading.Event()
        self.rss = self.alloc = self.reserved = 0.0
        self.min_avail_gb = 1e9
        self.first_preview_step = None

    def __enter__(self):
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self) -> None:
        while not self.stop.is_set():
            b = engc.beat or {}
            self.rss = max(self.rss, b.get("rss") or 0)
            self.alloc = max(self.alloc, b.get("alloc") or 0)
            self.reserved = max(self.reserved, b.get("reserved") or 0)
            self.min_avail_gb = min(self.min_avail_gb, guard.mem_available_mb() / 1024)
            t = self.task
            if t is not None and self.first_preview_step is None and t.preview_n:
                self.first_preview_step = t.done
            time.sleep(0.25)

    def __exit__(self, *exc) -> None:
        self.stop.set()

    def as_dict(self) -> dict:
        return {"worker_rss_gb": round(self.rss, 2), "vram_alloc_gb": round(self.alloc, 2),
                "vram_reserved_gb": round(self.reserved, 2), "min_ram_free_gb": round(self.min_avail_gb, 2),
                "first_preview_step": self.first_preview_step}


def base_state(**kw) -> State:
    S = State()
    S.update(kind="Z-Image", z_model="turbo Q8", z_te="official", z_quant="nf4", mode="Speed",
             aspect="Portrait 896x1152", prompt=PROMPT, seed=42, count=1, use_adv=False, loras=[])
    S.update(**kw)
    return S


def run(label: str, fn, S: State) -> dict:
    """Run load_blocking / generate_blocking with a fresh task; never raises."""
    task = Task("generate", label)
    t0 = time.time()
    rec: dict = {"label": label}
    try:
        with Peaks(task) as pk:
            out = fn(S, task)
        rec.update(ok=True, seconds=round(time.time() - t0, 1), **pk.as_dict())
        if isinstance(out, tuple):
            paths, info, seed = out
            rec.update(paths=paths, info=info, seed=seed)
            for p in paths:
                RESULTS.append((label, p))
        else:
            rec["info"] = out
    except Exception as e:
        rec.update(ok=False, seconds=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}",
                   tb=traceback.format_exc()[-1500:])
    print(json.dumps({k: v for k, v in rec.items() if k != "tb"}), flush=True)
    return rec


def ellipse_mask(size: tuple[int, int], box=(0.3, 0.15, 0.7, 0.55)) -> bytes:
    """A painted-looking mask at display size (long side <= 2048), as the painter would POST it."""
    w, h = size
    f = min(1.0, 2048 / max(w, h))
    dw, dh = max(1, int(w * f)), max(1, int(h * f))
    m = Image.new("L", (dw, dh), 0)
    ImageDraw.Draw(m).ellipse([box[0] * dw, box[1] * dh, box[2] * dw, box[3] * dh], fill=255)
    import io
    b = io.BytesIO()
    m.save(b, "PNG")
    return b.getvalue()


def outside_identical(src: str, result: str, mask_png: bytes, multiple: int, feather: int) -> float | None:
    """Fraction of pixels outside the grown + feathered mask that equal the work copy (1.0 = byte-identical)."""
    import io

    import numpy as np
    base, _, _ = work_copy(edit.load_source(src), multiple)
    out = Image.open(result).convert("RGB")
    if out.size != base.size:
        return None
    m = Image.open(io.BytesIO(mask_png)).convert("L").resize(base.size)
    pad = max(8, int(0.01 * min(base.size))) + 3 * feather + 8  # grow + Gaussian blur tail (~3x the radius) + rounding
    k = pad * 2 + 1
    while k > 1:  # MaxFilter takes odd sizes up to a practical limit: dilate in steps
        step = min(k, 61) | 1
        m = m.filter(ImageFilter.MaxFilter(step))
        k -= step - 1
    outside = np.asarray(m) == 0
    same = (np.asarray(out) == np.asarray(base)).all(axis=2)
    return round(float(same[outside].mean()), 6) if outside.any() else None


def put_source(path: Path) -> str:
    """An input photo as the UI would hold it (PNG under outputs/, exif applied)."""
    day = OUTPUTS / "t4_inputs"
    day.mkdir(parents=True, exist_ok=True)
    p = day / (path.stem + ".png")
    edit.load_source(path).save(p)
    return str(p)


# ---------- sections ----------
def sec_loads(a) -> list[dict]:
    recs = []
    for z in ("turbo Q8", "turbo Q6", "turbo Q4"):
        engc.kill()  # clean peaks for every load
        S = base_state(z_model=z)
        recs.append(run(f"load {z}", load_blocking, S))
        recs.append(run(f"gen {z} Speed 896x1152", generate_blocking, S))
        recs.append(run(f"gen {z} again (warm)", generate_blocking, S))
    return recs


def sec_te(a) -> list[dict]:
    import numpy as np
    recs, imgs = [], {}
    for te in ("official", "unsloth"):
        engc.kill()
        S = base_state(z_te=te)
        recs.append(run(f"load TE {te}", load_blocking, S))
        r = run(f"gen TE {te}", generate_blocking, S)
        recs.append(r)
        if r.get("ok"):
            imgs[te] = np.asarray(Image.open(r["paths"][0]).convert("RGB"), dtype=np.float32)
    if len(imgs) == 2 and imgs["official"].shape == imgs["unsloth"].shape:
        recs.append({"label": "TE image difference", "mean_abs_diff_0_255": round(float(np.abs(
            imgs["official"] - imgs["unsloth"]).mean()), 2)})
    return recs


def _edits_for(src: str, kind: str, extra: dict, tag: str) -> list[dict]:
    recs = []
    multiple = 16 if kind == "Z-Image" else 8
    mpng = ellipse_mask(Image.open(src).size)
    mid = masks.store(CODE, mpng)
    common = dict(src=src, image=src, prompt="a dark green wool coat", feather=12, edit_count=1, **extra)
    plain = run(f"{tag} painted (plain)", generate_blocking, base_state(kind=kind, tool="area", mask_id=mid,
                                                                       strength=0.85 if multiple == 16 else 0.75,
                                                                       better_blend=False, **common))
    if plain.get("ok"):
        plain["outside_identical"] = outside_identical(src, plain["paths"][0], mpng, multiple, 12)
    recs.append(plain)
    if kind == "Z-Image":
        cn = run(f"{tag} painted (better blend)", generate_blocking, base_state(
            kind=kind, tool="area", mask_id=mid, strength=0.85, better_blend=True, **common))
        if cn.get("ok"):
            cn["outside_identical"] = outside_identical(src, cn["paths"][0], mpng, multiple, 12)
        recs.append(cn)
    recs.append(run(f"{tag} extend 25% each side", generate_blocking, base_state(
        kind=kind, tool="extend", extend=[25, 25, 25, 25], better_blend=False, **common)))
    recs.append(run(f"{tag} whole 0.5", generate_blocking, base_state(
        kind=kind, tool="whole", whole_strength=0.5, keep_size=False, **common)))
    recs.append(run(f"{tag} upscale 2x", generate_blocking, base_state(
        kind=kind, tool="upscale", up_scale=2.0, up_strength=0.3, tile_cn=False, **common)))
    if kind == "Z-Image":
        recs.append(run(f"{tag} upscale 2x (tile model)", generate_blocking, base_state(
            kind=kind, tool="upscale", up_scale=2.0, up_strength=0.3, tile_cn=True, **common)))
    return recs


def sec_edits(a) -> list[dict]:
    recs = []
    imgs = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))[:2]
    if not imgs:
        return [{"label": "edits", "ok": False, "error": f"no images in {a.images}"}]
    for i, p in enumerate(imgs):
        recs += _edits_for(put_source(p), "Z-Image", {}, f"Z img{i + 1}")
    return recs


def sec_big(a) -> list[dict]:
    import numpy as np
    p = OUT / "big_6000x4000.jpg"
    rng = np.random.default_rng(0)
    small = Image.fromarray(rng.integers(0, 255, (400, 600, 3), dtype=np.uint8)).resize((6000, 4000), Image.BICUBIC)
    small.save(p, quality=90)
    src = put_source(p)
    mid = masks.store(CODE, ellipse_mask((6000, 4000)))
    return [run("big 24 MP painted edit", generate_blocking, base_state(
        tool="area", src=src, image=src, mask_id=mid, strength=0.85, feather=12, edit_count=1, better_blend=False,
        prompt="a small red flower"))]


def sec_sdxl(a) -> list[dict]:
    recs = []
    engc.kill()
    if a.sdxl == "base":  # the public SDXL base checkpoint (6.9 GB, once)
        from sdxlr.fetch import fetch
        from sdxlr.paths import MODELS, SDXL_BASE
        a.sdxl = str(fetch(SDXL_BASE, MODELS))
    sd = dict(kind="SDXL", sdxl_model=a.sdxl, vae="fp16fix", negative="lowres, blurry")
    recs.append(run("load SDXL", load_blocking, base_state(**sd)))
    loras = []
    if a.loras:
        import shutil
        LORAS.mkdir(parents=True, exist_ok=True)
        for f in a.loras.split(","):
            shutil.copy2(f, LORAS / Path(f).name)
            loras.append([Path(f).name, 0.7])
        files._refresh(force=True)
    for n in sorted({0, min(1, len(loras)), min(3, len(loras))}):
        recs.append(run(f"SDXL gen {n} LoRAs", generate_blocking, base_state(**sd, loras=loras[:n])))
        recs.append(run(f"SDXL gen {n} LoRAs (warm)", generate_blocking, base_state(**sd, loras=loras[:n])))
    imgs = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))[:1]
    if imgs:
        recs += _edits_for(put_source(imgs[0]), "SDXL", {"sdxl_model": a.sdxl, "vae": "fp16fix"}, "SDXL img1")
    return recs


def sec_hog(a) -> list[dict]:
    recs = []
    t0 = time.time()
    rec = {"label": "RAM hog in the worker"}
    try:
        with Peaks() as pk:
            engc.call("hog", {"step_mb": 128, "pause": 0.25}, Task("generate", "hog"))
        rec.update(ok=False, error="the hog finished: the guard never acted", **pk.as_dict())
    except Exception as e:
        killed_by_guard = "low on memory" in str(e)  # a plain death (kernel OOM killer first) still means the session survived
        rec.update(ok=killed_by_guard, seconds=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}",
                   **pk.as_dict())
    rec["guard_notes"] = [m for _, m in guard.EVENTS[-3:]]
    print(json.dumps(rec), flush=True)
    recs.append(rec)
    recs.append(run("gen after the guard kill", generate_blocking, base_state()))
    return recs


SECTIONS = {"loads": sec_loads, "te": sec_te, "edits": sec_edits, "big": sec_big, "sdxl": sec_sdxl, "hog": sec_hog}


# ---------- report ----------
def sheet() -> None:
    if not RESULTS:
        return
    tw, th, cols = 256, 300, 6
    rows = -(-len(RESULTS) // cols)
    s = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
    d = ImageDraw.Draw(s)
    for i, (label, p) in enumerate(RESULTS):
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((tw - 8, th - 40))
            x, y = (i % cols) * tw, (i // cols) * th
            s.paste(im, (x + 4, y + 4))
            d.text((x + 4, y + th - 32), label[:40], fill=(230, 230, 230))
        except Exception:
            pass
    s.save(OUT / "sheet.jpg", quality=88)


def write_report() -> None:
    REPORT["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=1), encoding="utf-8")
    lines = [f"# T4 check {REPORT['started']}", ""]
    for name, recs in REPORT["sections"].items():
        lines += [f"## {name}", "", "| step | ok | s | worker RAM GB | VRAM alloc GB | min free RAM GB | notes |",
                  "|---|---|---|---|---|---|---|"]
        for r in recs:
            notes = r.get("error") or r.get("info") or ""
            for k in ("outside_identical", "first_preview_step", "mean_abs_diff_0_255"):
                if r.get(k) is not None:
                    notes = f"{k}={r[k]} " + notes
            lines.append(f"| {r.get('label')} | {'yes' if r.get('ok', True) else 'NO'} | {r.get('seconds', '')} | "
                         f"{r.get('worker_rss_gb', '')} | {r.get('vram_alloc_gb', '')} | {r.get('min_ram_free_gb', '')} | "
                         f"{str(notes)[:160].replace('|', '/')} |")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/content/t4_imgs")
    ap.add_argument("--sections", default="")
    ap.add_argument("--sdxl", default="")
    ap.add_argument("--loras", default="")
    a = ap.parse_args()
    names = [s for s in a.sections.split(",") if s] or \
        [s for s in ("loads", "te", "edits", "big", "sdxl", "hog") if s != "sdxl" or a.sdxl]
    if "hog" in names:
        if not os.environ.get("SDXLR_DEBUG_OPS"):
            sys.exit("hog needs SDXLR_DEBUG_OPS=1 (the worker only has the test op then)")
        names = [n for n in names if n != "hog"] + ["hog"]  # last: it ends with a killed worker
    OUT.mkdir(parents=True, exist_ok=True)
    guard.start(engc, jm)  # the same watchdog the app runs
    files.start_refresh()
    print(f"T4 check: {', '.join(names)} -> {OUT}", flush=True)
    try:
        for n in names:
            print(f"== {n}", flush=True)
            REPORT["sections"][n] = SECTIONS[n](a)
            write_report()
    finally:
        sheet()
        write_report()
        engc.kill()
    print(f"done: {OUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
