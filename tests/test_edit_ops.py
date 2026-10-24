"""The edit ops (area, extend, whole) through the real client and worker with the stub engine.

Run: uv run --no-project -q --with psutil --with pillow --with numpy python -I tests/test_edit_ops.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, SRC)
TMP = Path(tempfile.mkdtemp(prefix="sdxlr_ops_test_"))
# setdefault: the spawned worker re-imports this file and must keep the parent's values
for _k, _v in {"SDXLR_FAKE_ENGINE": "1", "SDXLR_FAKE_FAST": "1", "SDXLR_WORK": str(TMP / "work")}.items():
    os.environ.setdefault(_k, _v)

from sdxlr.engine import client as client_mod  # noqa: E402
from sdxlr.engine import edit  # noqa: E402

MULTIPLE = 8  # the stub loads as SDXL


def make_source(w: int, h: int) -> Path:
    rng = np.random.default_rng(3)
    p = TMP / f"src_{w}x{h}.png"
    Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB").save(p)
    return p


def make_mask(size, rects, name) -> Path:
    p = TMP / name
    m = edit.mask_from_boxes(size, rects)
    m.save(p)
    return p


def run(job: dict, source: Path):
    job = {"prompt": "x", "steps": 10, "cfg": 5, "seed": 5, "count": 1, "blend": "plain", "image": str(source), **job}
    c = client_mod.EngineClient()
    try:
        c.call("load", {"kind": "sdxl", "path": "/m/a.safetensors"})
        r = c.call("generate", {"job": job, "meta": {"source": str(source), "tool": job.get("tool")}})
    finally:
        c.kill()
    return r


def blend_union(src_img: Image.Image, mask: Image.Image, feather: int = 12) -> np.ndarray:
    """Where the stitch may change pixels: the union of the per-region blends, as the worker computes them."""
    full = np.zeros((src_img.height, src_img.width), np.uint8)
    for rm in edit.split_regions(mask) or [mask]:
        plan = edit.prepare(src_img, rm, MULTIPLE, feather_px=feather)
        x0, y0, x1, y1 = plan.box
        full[y0:y1, x0:x1] = np.maximum(full[y0:y1, x0:x1], np.asarray(plan.blend))
    return full


def check_sidecars(r: dict, source: Path, tool: str) -> None:
    for p in r["paths"]:
        assert Path(p).exists()
        meta = json.loads(Path(p).with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["source"] == str(source) and meta["tool"] == tool, meta
        if tool in ("area", "extend"):
            assert Path(p[:-4] + "_mask.png").exists(), "the mask is saved next to the result"


def test_area_painted_mask_is_identical_outside() -> None:
    src = make_source(800, 600)
    # painted at half size (a display copy): the worker resizes it
    mask_file = make_mask((400, 300), [[100, 80, 180, 140]], "m1.png")
    r = run({"tool": "area", "mask": str(mask_file), "feather": 12}, src)
    out = Image.open(r["paths"][0]).convert("RGB")
    assert out.size == (800, 600) and r["size"] == [800, 600]
    check_sidecars(r, src, "area")
    a, b = np.asarray(Image.open(src).convert("RGB")), np.asarray(out)
    blend = blend_union(Image.open(src).convert("RGB"), edit.mask_from_painted(mask_file, (800, 600)))
    assert (blend > 0).any()
    outside = blend == 0
    assert (a[outside] == b[outside]).all(), "pixels outside the blend must be byte-identical"
    assert (a[blend == 255] != b[blend == 255]).any(), "the painted part must change"


def test_area_two_disjoint_regions() -> None:
    src = make_source(1000, 800)
    mask_file = make_mask((1000, 800), [[40, 40, 140, 120], [840, 660, 950, 750]], "m2.png")
    r = run({"tool": "area", "mask": str(mask_file), "feather": 8, "count": 2}, src)
    assert len(r["paths"]) == 2
    check_sidecars(r, src, "area")
    a = np.asarray(Image.open(src).convert("RGB"))
    mask = edit.mask_from_painted(mask_file, (1000, 800))
    assert len(edit.split_regions(mask) or []) == 2, "far apart regions are processed one by one"
    blend = blend_union(Image.open(src).convert("RGB"), mask, feather=8)
    for p in r["paths"]:
        b = np.asarray(Image.open(p).convert("RGB"))
        assert (a[blend == 0] == b[blend == 0]).all()


def test_mask_with_other_aspect_is_refused() -> None:
    src = make_source(800, 600)
    bad = make_mask((300, 300), [[10, 10, 50, 50]], "m3.png")
    try:
        run({"tool": "area", "mask": str(bad)}, src)
        raise AssertionError("expected the mask to be refused")
    except ValueError as e:
        assert "doesn't match this image" in str(e)


def test_empty_mask_and_blend_hook() -> None:
    src = make_source(640, 480)
    empty = make_mask((640, 480), [], "m4.png")
    try:
        run({"tool": "area", "mask": str(empty)}, src)
        raise AssertionError("expected 'nothing painted'")
    except ValueError as e:
        assert "Nothing is painted" in str(e)
    ok = make_mask((640, 480), [[100, 100, 200, 200]], "m5.png")
    try:
        run({"tool": "area", "mask": str(ok), "blend": "controlnet"}, src)
        raise AssertionError("expected the blend hook to refuse")
    except ValueError as e:
        assert "not available yet" in str(e)


def test_extend() -> None:
    src = make_source(640, 480)
    r = run({"tool": "extend", "extend": [0, 25, 0, 25], "feather": 12}, src)  # top, right, bottom, left (%)
    check_sidecars(r, src, "extend")
    out = Image.open(r["paths"][0]).convert("RGB")
    assert out.width > 640 and out.height == 480 and out.width % MULTIPLE == 0, out.size
    ox = int((out.width - 640) * 0.5)  # 25/25: centered
    inner = (ox + 80, 80, ox + 640 - 80, 480 - 80)  # deeper than the 32 px band + grow + feather
    a = np.asarray(Image.open(src).convert("RGB").crop((inner[0] - ox, inner[1], inner[2] - ox, inner[3])))
    assert (a == np.asarray(out.crop(inner))).all(), "the original stays inside the band"


def test_whole_keep_size_and_variations() -> None:
    src = make_source(2400, 1800)
    r = run({"tool": "whole", "strength": 0.5, "keep_size": True, "count": 2}, src)
    assert len(r["paths"]) == 2 and all(Image.open(p).size == (2400, 1800) for p in r["paths"])
    check_sidecars(r, src, "whole")
    r2 = run({"tool": "whole", "strength": 0.5, "keep_size": False}, src)
    w, h = Image.open(r2["paths"][0]).size
    assert w * h <= 1.25e6 and w % MULTIPLE == 0 and h % MULTIPLE == 0, (w, h)  # the ~1 MP work copy


def test_cancel_keeps_finished_variations() -> None:
    import threading
    import time

    from sdxlr.activity import Cancelled, Task
    src = make_source(640, 480)
    mask = make_mask((640, 480), [[100, 100, 200, 200]], "m6.png")
    out_dir = TMP / "work" / "outputs"
    before = set(out_dir.rglob("*.png")) if out_dir.exists() else set()
    os.environ["SDXLR_FAKE_SCALE"] = "1"  # 4 s per variation
    os.environ["SDXLR_FAKE_FAST"] = "0"
    try:
        c = client_mod.EngineClient()
        res = {}
        try:
            c.call("load", {"kind": "sdxl", "path": "/m/a.safetensors"})
            task = Task("generate", "v")

            def go():
                job = {"prompt": "x", "steps": 10, "cfg": 5, "seed": 5, "count": 3, "blend": "plain", "image": str(src),
                       "tool": "area", "mask": str(mask)}
                try:
                    c.call("generate", {"job": job, "meta": {"source": str(src)}}, task)
                except Cancelled:
                    res["r"] = "Cancelled"

            th = threading.Thread(target=go, daemon=True)
            th.start()
            time.sleep(6.0)  # inside the second variation
            task.cancel()
            th.join(15)
        finally:
            c.kill()
    finally:
        os.environ["SDXLR_FAKE_FAST"] = "1"
        os.environ.pop("SDXLR_FAKE_SCALE", None)
    made = [p for p in out_dir.rglob("*.png") if p not in before and "_mask" not in p.name]
    assert res.get("r") == "Cancelled" and len(made) >= 1, (res, made)


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"{name}: ok")
        except Exception as e:
            failed += 1
            print(f"{name}: FAILED {type(e).__name__}: {e}")
    print("edit op tests passed" if not failed else f"{failed} edit op tests FAILED")
    sys.exit(1 if failed else 0)
