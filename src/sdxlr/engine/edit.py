"""Pure image/mask operations for the editing tools (PIL + numpy only, no torch).

Crop-and-stitch inpaint: the mask says what to repaint; a context box around
it is cropped, resized to about 1 MP (small regions get upscaled - the main
quality win), generated, then blended back with a feathered mask so every
pixel outside the blend stays byte-identical. Outpaint builds a bigger
canvas with edge-extended filler and runs the same path.

All coordinates are pixels of the (already capped) work copy.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

GROW_PCT = 0.01  # grow the mask by max(GROW_MIN_PX, this fraction of the short side)
GROW_MIN_PX = 8
DEFAULT_FEATHER = 12
CTX_FACTOR = 1.75  # context box = mask bbox times this
CTX_MIN_SIDE = 512
TARGET_MP = 1.0  # the crop is resized to about a megapixel
SPLIT_RATIO = 2.2  # split disjoint regions when the shared box is this much bigger than them
OUTPAINT_BAND_PX = 32  # mask band reaching into the original image
MAX_OUT_MP = 8.0  # outpaint and upscale output cap


# ---------- sources ----------
def load_source(path: str | Path) -> Image.Image:
    """RGB, EXIF-rotated, alpha composited on white, first frame of animations."""
    img = Image.open(path)
    if getattr(img, "is_animated", False):
        img.seek(0)
    from PIL import ImageOps
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I;16", "I;16L", "I;16B", "I", "F"):  # 16-bit grayscale: scale to 8 bit (convert would clip to white)
        a = np.asarray(img, dtype=np.float64)
        top = 65535.0 if a.max() > 255 else 255.0
        img = Image.fromarray(np.clip(a / top * 255.0 + 0.5, 0, 255).astype(np.uint8), "L")
    if img.mode == "P" and "transparency" in img.info or img.mode in ("LA", "PA"):
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def mask_from_boxes(size: tuple[int, int], boxes: list) -> Image.Image:
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    for x0, y0, x1, y1 in boxes or []:
        d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=255)
    return m


def mask_from_painted(path: str | Path, size: tuple[int, int]) -> Image.Image:
    """The painter's PNG (8-bit L, natural size) resized to the work copy."""
    m = Image.open(path).convert("L")
    if abs(m.width / m.height - size[0] / size[1]) > 0.01 * size[0] / size[1]:
        raise ValueError("The mask doesn't match this image; clear it and paint again.")
    if m.size != size:
        m = m.resize(size, Image.BILINEAR)
    return m


# ---------- mask ops ----------
def hipass(mask: Image.Image, threshold: float = 0.1) -> Image.Image:
    """Binarize; values under the threshold are 'not painted'."""
    a = np.asarray(mask)
    return Image.fromarray(np.where(a < threshold * 255, 0, 255).astype(np.uint8), "L")


def fill_holes(mask: Image.Image) -> Image.Image:
    """Paint enclosed unpainted pockets (inside a drawn outline, say)."""
    a = np.asarray(mask)
    scale = max(1, math.ceil(max(a.shape) / 400))  # label at <= ~400 px: holes are macro features
    small = a[::scale, ::scale] == 0  # unpainted
    h, w = small.shape
    reach = np.zeros_like(small)
    q = deque((y, x) for y in (0, h - 1) for x in range(w) if small[y, x])
    q += deque((y, x) for x in (0, w - 1) for y in range(h) if small[y, x])
    while q:
        y, x = q.popleft()
        if reach[y, x]:
            continue
        reach[y, x] = True
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and small[ny, nx] and not reach[ny, nx]:
                q.append((ny, nx))
    holes = small & (~reach)  # unpainted, but not reachable from the border
    if not holes.any():
        return mask
    full = np.asarray(mask).copy()
    holes_full = np.kron(holes, np.ones((scale, scale), bool))[:full.shape[0], :full.shape[1]]
    full[holes_full] = 255
    return Image.fromarray(full, "L")


def grow(mask: Image.Image, px: int) -> Image.Image:
    return mask.filter(ImageFilter.MaxFilter(px * 2 + 1)) if px > 0 else mask


def feather_mask(mask: Image.Image, px: int) -> Image.Image:
    return mask.filter(ImageFilter.GaussianBlur(px)) if px > 0 else mask


def grow_px_for(size: tuple[int, int]) -> int:
    return max(GROW_MIN_PX, int(min(size) * GROW_PCT))


# ---------- geometry ----------
def mask_bbox(mask: Image.Image) -> tuple[int, int, int, int] | None:
    a = np.asarray(mask) > 127
    if not a.any():
        return None
    ys, xs = np.where(a)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def context_box(mask: Image.Image, size: tuple[int, int], factor: float = CTX_FACTOR,
                min_side: int = CTX_MIN_SIDE) -> tuple[int, int, int, int]:
    """Mask bbox grown by `factor` around its center, at least `min_side` wide, kept inside the image."""
    W, H = size
    x0, y0, x1, y1 = mask_bbox(mask) or (0, 0, W, H)
    w, h = x1 - x0, y1 - y0
    cw, ch = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = max(w * factor, min_side), max(h * factor, min_side)
    x0, y0 = int(cw - w / 2), int(ch - h / 2)
    x1, y1 = math.ceil(cw + w / 2), math.ceil(ch + h / 2)
    if x1 - x0 > W:  # wider than the image: center it
        x0, x1 = 0, W
    if y1 - y0 > H:
        y0, y1 = 0, H
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)


def region_boxes(mask: Image.Image, max_parts: int = 4) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of connected painted regions (labeled on a thumbnail)."""
    a = np.asarray(mask) > 127
    if not a.any():
        return []
    scale = max(1, math.ceil(max(a.shape) / 256))
    small = a[::scale, ::scale]
    h, w = small.shape
    seen = np.zeros_like(small)
    boxes = []
    for sy in range(h):
        for sx in range(w):
            if not small[sy, sx] or seen[sy, sx]:
                continue
            q, minx, miny, maxx, maxy = deque([(sy, sx)]), sx, sy, sx, sy
            seen[sy, sx] = True
            while q:
                y, x = q.popleft()
                minx, miny, maxx, maxy = min(minx, x), min(miny, y), max(maxx, x), max(maxy, y)
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and small[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            boxes.append((minx * scale, miny * scale, min((maxx + 1) * scale, a.shape[1]),
                          min((maxy + 1) * scale, a.shape[0])))
            if len(boxes) >= max_parts:
                break
    return boxes


def split_regions(mask: Image.Image) -> list[Image.Image] | None:
    """Disjoint regions that share a hugely oversized context box get a mask each."""
    a = np.asarray(mask) > 127
    if not a.any():
        return None
    boxes = region_boxes(mask)
    if len(boxes) < 2:
        return None
    box = context_box(mask, (a.shape[1], a.shape[0]), min_side=1)  # natural box, no min-side padding
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    mask_area = int(a.sum())
    if box_area < SPLIT_RATIO * max(mask_area, 1):
        return None
    out = []
    for x0, y0, x1, y1 in boxes:
        m = np.zeros_like(a)
        m[y0:y1, x0:x1] = a[y0:y1, x0:x1]
        out.append(Image.fromarray(np.where(m, 255, 0).astype(np.uint8), "L"))
    return out


# ---------- crop-and-stitch ----------
@dataclass
class EditPlan:
    box: tuple[int, int, int, int]  # crop in the work image
    crop: Image.Image  # what the pipeline sees (~TARGET_MP, multiple-aligned)
    crop_mask: Image.Image  # grown mask, same size as crop
    blend: Image.Image  # feathered blend at work-image scale, crop of the box


def _mult(n: int, m: int) -> int:
    return max(m, int(round(n / m)) * m)


def prepare(img: Image.Image, mask: Image.Image, multiple: int, feather_px: int = DEFAULT_FEATHER,
            grow_px: int | None = None, fill: bool = True) -> EditPlan:
    """Full mask pipeline up to the pipeline call; the result is stitched by paste_back.
    fill=False keeps holes (outpaint: the kept original is a hole in the mask)."""
    mask = hipass(mask)
    if fill:
        mask = fill_holes(mask)
    grown = grow(mask, grow_px if grow_px is not None else grow_px_for(img.size))
    box = context_box(grown, img.size)
    bx, by = box[2] - box[0], box[3] - box[1]
    f = math.sqrt(TARGET_MP * 1e6 / (bx * by))  # both ways: big crops shrink, small ones upscale to detail
    nw, nh = _mult(bx * f, multiple), _mult(by * f, multiple)
    blend = feather_mask(grown, feather_px).crop(box)
    return EditPlan(box=box, crop=img.crop(box).resize((nw, nh), Image.LANCZOS),
                    crop_mask=grown.crop(box).resize((nw, nh), Image.BILINEAR), blend=blend)


def paste_back(img: Image.Image, out_crop: Image.Image, plan: EditPlan) -> Image.Image:
    """Resize the generated crop back, blend it over the original crop, paste at the box.
    Pixels where the blend is 0 stay byte-identical."""
    x0, y0, x1, y1 = plan.box
    out = out_crop.resize((x1 - x0, y1 - y0), Image.LANCZOS) if out_crop.size != (x1 - x0, y1 - y0) else out_crop
    region = Image.composite(out, img.crop(plan.box), plan.blend)
    res = img.copy()
    res.paste(region, (x0, y0))
    return res


# ---------- outpaint ----------
def outpaint_canvas(img: Image.Image, sides: tuple[float, float, float, float],
                    multiple: int = 16) -> tuple[Image.Image, Image.Image]:
    """(canvas, mask): new area filled with edge-extended + blurred + noised pixels;
    the mask covers the new area plus a band into the original. sides = % per (top, right, bottom, left)."""
    t, r, b, l = (max(0.0, float(s)) for s in sides)
    W, H = img.size
    if not any((t, r, b, l)):
        return img.copy(), Image.new("L", img.size, 0)
    NW, NH = _mult(W * (1 + (l + r) / 100), multiple), _mult(H * (1 + (t + b) / 100), multiple)
    f = min(1.0, math.sqrt(MAX_OUT_MP * 1e6 / (NW * NH)))
    NW, NH = _mult(NW * f, multiple), _mult(NH * f, multiple)
    # the added space is split between the sides by their ratio (25/25 = centered)
    ox = int((NW - W) * (l / (l + r))) if l + r else 0
    oy = int((NH - H) * (t / (t + b))) if t + b else 0
    # stretch the border pixels outward, then blur the filler so it carries no sharp pattern
    big = img.resize((NW, NH), Image.LANCZOS)  # cheap edge extension: same content, scaled to fill
    filler = big.filter(ImageFilter.GaussianBlur(24))
    noise = (np.random.default_rng(7).normal(0, 5, (NH, NW, 1))).repeat(3, axis=2)  # light texture, filler only
    arr = np.asarray(filler).astype(np.int16) + noise.astype(np.int16)
    canvas = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    canvas.paste(img, (ox, oy))  # the original last: its pixels stay exactly as they were
    # repaint everything except the original area shrunk by the band (0 = keep)
    # the band reaches into the original only on sides that got new space
    m = Image.new("L", (NW, NH), 255)
    d = ImageDraw.Draw(m)
    bl, bt = OUTPAINT_BAND_PX if ox > 0 else 0, OUTPAINT_BAND_PX if oy > 0 else 0
    br, bb = OUTPAINT_BAND_PX if NW - (ox + W) > 0 else 0, OUTPAINT_BAND_PX if NH - (oy + H) > 0 else 0
    d.rectangle([ox + bl, oy + bt, ox + W - 1 - br, oy + H - 1 - bb], fill=0)
    return canvas, m


def cap_size(w: int, h: int, max_mp: float = MAX_OUT_MP) -> tuple[int, int]:
    f = min(1.0, math.sqrt(max_mp * 1e6 / (w * h)))
    return max(1, int(w * f)), max(1, int(h * f))
