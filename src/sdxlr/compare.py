"""LoRA comparison grid. Runs inside the main venv.

  python -m sdxlr.compare --model PATH --loras GLOB --strengths 0.6 0.8 1.0 --prompt "..."
  python -m sdxlr.compare --model PATH --combos "a.safetensors:0.44+b.safetensors:0.44" ...

Rows = LoRAs (or combos), columns = strengths. Same seed everywhere, so only
the LoRA changes. Resume-safe: finished cells are skipped on re-run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .engine.models import Engine, Job
from .paths import Z_ENCODERS, Z_MODELS
from .fetch import LORAS, MODELS, WORK, fetch, list_loras

OUT = WORK / "compare"


def _font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _resolve(name: str) -> str:
    if Path(name).exists():
        return name
    for p in list_loras():
        if p.name == name:
            return str(p)
    raise FileNotFoundError(f"{name} (not in {LORAS} or this session's runs)")


def stitch(rows: list[str], cols: list[str], cells: dict, thumb: int = 320) -> Image.Image:
    font, pad, label_w, head_h = _font(18), 8, 260, 36
    any_img = Image.open(next(iter(cells.values())))
    th = int(thumb * any_img.height / any_img.width)
    W = label_w + len(cols) * (thumb + pad) + pad
    H = head_h + len(rows) * (th + pad) + pad
    grid = Image.new("RGB", (W, H), (248, 248, 250))
    d = ImageDraw.Draw(grid)
    for j, c in enumerate(cols):
        d.text((label_w + j * (thumb + pad) + thumb // 2 - 20, 8), c, fill=(20, 20, 20), font=font)
    for i, r in enumerate(rows):
        y = head_h + i * (th + pad)
        label = r if len(r) <= 24 else r[:23] + "…"
        d.text((pad, y + th // 2 - 10), label, fill=(20, 20, 20), font=font)
        for j, c in enumerate(cols):
            if (f := cells.get(f"{r}|{c}")) and Path(f).exists():
                grid.paste(Image.open(f).resize((thumb, th), Image.LANCZOS), (label_w + j * (thumb + pad), y))
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="SDXL checkpoint, or with --zimage a Z-Image preset/.gguf")
    ap.add_argument("--zimage", action="store_true")
    ap.add_argument("--te", default="official")
    ap.add_argument("--te-quant", default="nf4")
    ap.add_argument("--loras", default="*.safetensors", help="glob over downloaded + trained LoRAs")
    ap.add_argument("--combos", nargs="*", default=[], help="'a.safetensors:0.44+b.safetensors:0.44'; strength = multiplier")
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.78, 0.82, 0.88])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--sampler", default="dpmpp_2m_karras")
    ap.add_argument("--clip-skip", type=int, default=2)
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--height", type=int, default=1152)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="grid")
    ap.add_argument("--fresh", action="store_true", help="ignore earlier results")
    a = ap.parse_args()

    if a.combos:
        rows = {c: [(_resolve(n), float(w)) for n, w in (x.rsplit(":", 1) for x in c.split("+"))] for c in a.combos}
    else:
        rows = {p.name: [(str(p), 1.0)] for p in list_loras() if p.match(a.loras)}
    if not rows:
        raise SystemExit(f"no LoRAs match {a.loras}; fetch some first")
    cols = [f"{s:.2f}" for s in a.strengths]

    out = OUT / a.name
    out.mkdir(parents=True, exist_ok=True)
    state_f = out / "state.json"
    params = {k: v for k, v in vars(a).items() if k not in ("fresh", "name")}
    state = json.loads(state_f.read_text()) if state_f.exists() and not a.fresh else {}
    if state.get("params") != params:
        state = {"params": params, "cells": {}, "errors": {}}

    eng = Engine()
    if a.zimage:
        print(eng.load_zimage(Z_MODELS.get(a.model, a.model), Z_ENCODERS.get(a.te, a.te), a.te_quant), flush=True)
    else:
        print(eng.load(str(fetch(a.model, MODELS))), flush=True)
    todo = [(r, c, s) for r in rows for c, s in zip(cols, a.strengths) if f"{r}|{c}" not in state["cells"]]
    print(f"{len(rows)} rows x {len(cols)} strengths, {len(todo)} to render", flush=True)
    t0 = time.time()
    for i, (r, c, s) in enumerate(todo, 1):
        job = Job(prompt=a.prompt, negative=a.negative, steps=a.steps, cfg=a.cfg, sampler=a.sampler,
                  clip_skip=a.clip_skip, width=a.width, height=a.height, seed=a.seed,
                  loras=[(p, w * s) for p, w in rows[r]])
        try:
            imgs, _, info = eng.generate(job)
            f = out / f"{i:03d}_{Path(r).stem[:40]}_{c}.png".replace("+", "_").replace(":", "-")
            imgs[0].save(f)
            state["cells"][f"{r}|{c}"] = str(f)
            print(f"[{i}/{len(todo)}] {r} @ {c}: {info}", flush=True)
        except Exception as e:  # one bad LoRA must not stop the grid
            state["errors"][f"{r}|{c}"] = f"{type(e).__name__}: {e}"
            print(f"[{i}/{len(todo)}] {r} @ {c}: FAILED {state['errors'][f'{r}|{c}']}", flush=True)
        state_f.write_text(json.dumps(state, indent=1))

    if not state["cells"]:
        raise SystemExit(f"no images rendered; errors: {json.dumps(state['errors'], indent=1)}")
    grid = stitch(list(rows), cols, state["cells"])
    grid.save(out / "grid.png")
    print(f"grid: {out / 'grid.png'} ({len(state['cells'])} images, {len(state['errors'])} errors, "
          f"{time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
