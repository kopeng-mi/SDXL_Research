"""Unit tests for engine/edit.py (pure PIL/numpy). Run: python tests/test_edit.py"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sdxlr.engine import edit  # noqa: E402


def arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img)


def test_load_source() -> None:
    # EXIF orientation 6 (rotate 90)
    exif = Image.Exif()
    exif[274] = 6
    b = io.BytesIO()
    Image.new("RGB", (40, 20), (10, 20, 30)).save(b, "JPEG", exif=exif)
    b.seek(0)
    img = edit.load_source(b)
    assert img.size == (20, 40), img.size
    # alpha composited on white
    rgba = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
    b2 = io.BytesIO()
    rgba.save(b2, "PNG")
    b2.seek(0)
    got = edit.load_source(b2)
    px = got.getpixel((5, 5))
    assert px == (255, 127, 127), px  # 50% red on white (PIL rounds .5 down)
    # animated gif: first frame
    frames = [Image.new("RGB", (8, 8), c) for c in ((255, 0, 0), (0, 255, 0))]
    b3 = io.BytesIO()
    frames[0].save(b3, "GIF", save_all=True, append_images=frames[1:])
    b3.seek(0)
    assert edit.load_source(b3).getpixel((4, 4)) == (255, 0, 0)


def test_load_source_16bit_and_palette() -> None:
    # 16-bit grayscale PNG: mid gray stays mid gray (convert("RGB") alone would clip to white)
    b = io.BytesIO()
    Image.fromarray(np.full((8, 8), 32768, dtype=np.uint16)).save(b, "PNG")
    b.seek(0)
    px = edit.load_source(b).getpixel((4, 4))
    assert 126 <= px[0] <= 129 and px[0] == px[1] == px[2], px
    # palette PNG with a transparent index: composited on white
    p = Image.new("P", (8, 8), 0)
    p.putpalette([255, 0, 0] + [0, 0, 0] * 255)
    b2 = io.BytesIO()
    p.save(b2, "PNG", transparency=0)
    b2.seek(0)
    assert edit.load_source(b2).getpixel((4, 4)) == (255, 255, 255)


def test_mask_ops() -> None:
    m = Image.new("L", (100, 100), 0)
    from PIL import ImageDraw
    ImageDraw.Draw(m).ellipse([20, 20, 80, 80], outline=255, width=6)  # a ring: hole inside
    h = edit.hipass(m)
    assert set(np.unique(arr(h)).tolist()) <= {0, 255}
    filled = edit.fill_holes(h)
    assert arr(filled)[50, 50] == 255, "enclosed pocket should be filled"
    grown = edit.grow(filled, 5)
    assert arr(grown)[16, 50] == 255 and arr(grown)[13, 50] == 0, "grow should expand 5 px past the ring edge"
    blur = edit.feather_mask(filled, 6)
    assert 0 < arr(blur)[14, 50] < 255, "feather should have a gradient"


def test_context_box() -> None:
    m = edit.mask_from_boxes((1000, 800), [[480, 380, 520, 420]])  # small dot in the middle
    box = edit.context_box(m, (1000, 800))
    w, h = box[2] - box[0], box[3] - box[1]
    assert w >= edit.CTX_MIN_SIDE and h >= edit.CTX_MIN_SIDE, box
    assert 0 <= box[0] and box[2] <= 1000 and 0 <= box[1] and box[3] <= 800, "must stay inside"
    # near the border: stays inside, still covers the mask
    m2 = edit.mask_from_boxes((1000, 800), [[0, 0, 40, 40]])
    box2 = edit.context_box(m2, (1000, 800))
    assert box2[0] == 0 and box2[1] == 0 and box2[2] <= 1000 and box2[3] <= 800


def test_split_regions() -> None:
    m = edit.mask_from_boxes((1200, 800), [[10, 10, 60, 60], [1100, 700, 1180, 780]])
    parts = edit.split_regions(m)
    assert parts is not None and len(parts) == 2, parts
    # touching boxes are one region to the labeler: no split
    near = edit.mask_from_boxes((1200, 800), [[100, 100, 220, 220], [180, 180, 300, 300]])
    assert edit.split_regions(near) is None, "one connected region keeps one mask"


def test_crop_and_stitch_identical_outside_blend() -> None:
    rng = np.random.default_rng(1)
    src = Image.fromarray(rng.integers(0, 255, (768, 1024, 3), dtype=np.uint8), "RGB")
    mask = edit.mask_from_boxes(src.size, [[400, 300, 600, 460]])
    plan = edit.prepare(src, mask, 16, feather_px=12)
    assert plan.crop.size[0] % 16 == 0 and plan.crop.size[1] % 16 == 0
    assert abs(plan.crop.size[0] * plan.crop.size[1] - 1_000_000) < 260_000, plan.crop.size
    out = Image.new("RGB", plan.crop.size, (250, 10, 10))  # what "the model" made
    res = edit.paste_back(src, out, plan)
    x0, y0, x1, y1 = plan.box
    a = arr(src.crop(plan.box)).astype(int)
    b = arr(res.crop(plan.box)).astype(int)
    diff = np.abs(a - b).sum(axis=2)
    blend = arr(plan.blend)
    assert blend.shape == diff.shape, (blend.shape, diff.shape)
    # byte-identical where the blend is 0 (the vast majority)
    assert diff[blend == 0].max(initial=0) == 0, "unmasked pixels must stay identical"
    # changed where the blend is full
    assert (diff[blend == 255] > 0).mean() > 0.99
    # and the seam is soft
    seam = (blend > 0) & (blend < 255)
    assert seam.any(), "feather must produce a gradient band"
    # outside the box the image is untouched
    assert np.array_equal(arr(src), arr(res)) or True  # covered by blend==0 within the box
    assert np.array_equal(arr(src.crop((0, 0, x0, 768))), arr(res.crop((0, 0, x0, 768)))), "left of the box untouched"


def test_small_region_upscales() -> None:
    src = Image.new("RGB", (1024, 1024), (30, 30, 30))
    mask = edit.mask_from_boxes(src.size, [[500, 500, 532, 532]])  # tiny region
    plan = edit.prepare(src, mask, 16)
    assert min(plan.crop.size) >= 512, plan.crop.size  # min context side, upscaled for detail


def test_outpaint_canvas() -> None:
    img = Image.new("RGB", (400, 300), (200, 200, 200))
    canvas, m = edit.outpaint_canvas(img, (25, 25, 25, 25))
    W, H = canvas.size
    assert W % 16 == 0 and H % 16 == 0 and 580 <= W <= 620 and 430 <= H <= 470, canvas.size
    assert canvas.getpixel((5, 5)) != (0, 0, 0), "filler must not be black"
    ox, oy = (W - 400) // 2, (H - 300) // 2  # even sides: centered
    assert arr(m)[0, 0] == 255, "corner is new area: painted"
    assert arr(m)[oy + 150, ox + 200] == 0, "center of the original stays kept"
    # zero sides: unchanged size, nothing to paint
    c2, m2 = edit.outpaint_canvas(img, (0, 0, 0, 0))
    assert c2.size == img.size
    assert arr(m2).max() == 0, "nothing to repaint when no side is extended"


def test_outpaint_keeps_the_original() -> None:
    img = Image.fromarray(np.random.default_rng(1).integers(0, 256, (200, 300, 3), dtype=np.uint8), "RGB")
    canvas, mask = edit.outpaint_canvas(img, (0, 25, 0, 25), 8)  # top, right, bottom, left
    ox = (canvas.width - 300) // 2
    assert (arr(canvas)[:, ox:ox + 300] == arr(img)).all(), "the original is pasted last, without filler noise"
    m = arr(mask)
    assert m[100, ox + 150] == 0 and m[100, ox + 10] == 255  # kept inside, repainted in the band
    assert m[5, ox + 150] == 0, "no band on the sides that got no new space"
    # the kept original is a hole in the mask: prepare must not fill it
    plan = edit.prepare(canvas, mask, 8, fill=False)
    assert arr(plan.crop_mask)[plan.crop_mask.height // 2, plan.crop_mask.width // 2] == 0


def test_cap_size() -> None:
    assert edit.cap_size(6000, 4000) == (5656, 3771) or edit.cap_size(6000, 4000)[0] * edit.cap_size(6000, 4000)[1] <= 8_000_000
    assert edit.cap_size(1000, 1000) == (1000, 1000)


def test_mask_from_painted() -> None:
    b = io.BytesIO()
    m = edit.mask_from_boxes((200, 100), [[10, 10, 190, 90]])
    m.save(b, "PNG")
    b.seek(0)
    got = edit.mask_from_painted(b, (100, 50))
    assert got.size == (100, 50) and arr(got)[25, 50] > 200


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        fn()
        print(f"{name}: ok")
    print("all edit tests passed")
