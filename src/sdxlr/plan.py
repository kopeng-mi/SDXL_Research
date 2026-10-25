"""Training and captioning math for the notebooks. Pure stdlib, no I/O (tests/test_plan.py).

Auto repeats keep the proven number of image views V = N·R·E per subject type,
so any dataset size trains about as long as the measured 11-photo runs.
Numbers marked "est" are guesses until a T4 run measures them (docs/NOTEBOOKS_PLAN.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# subject -> caption style, target views (sdxl, zimage), flip augmentation, measured?
SUBJECTS = {
    "person": {"captions": "id", "views": {"sdxl": 990, "zimage": 440}, "flip": False, "est": False},
    "character/object": {"captions": "id", "views": {"sdxl": 990, "zimage": 440}, "flip": False, "est": True},
    "style": {"captions": "detailed", "views": {"sdxl": 1500, "zimage": 660}, "flip": True, "est": True},
    "concept/pose": {"captions": "detailed", "views": {"sdxl": 1200, "zimage": 530}, "flip": True, "est": True},
}

# seconds per optimizer step on T4: (kind, res, batch) -> s; the first entry per kind is the scaling base
STEP_TIME = {("sdxl", 1024, 3): 5.1, ("zimage", 512, 1): 5.7}
OVERHEAD = {"sdxl": 120, "zimage": 300}  # caching + model load, seconds
BUCKET_STEP = {"sdxl": 64, "zimage": 16}
MIN_PHOTOS = 5
MAX_REPEATS = 20
MIN_EPOCHS = 4


@dataclass
class TrainPlan:
    images: int
    repeats: int
    epochs: int
    batch: int
    steps: int
    seconds: float
    est: bool
    warnings: list[str]

    def summary(self) -> str:
        m = round(self.seconds / 60)
        line = (f"{self.images} images x {self.repeats} repeats x {self.epochs} epochs / batch {self.batch}"
                f" = {self.steps} steps, about {m} min{' (est)' if self.est else ''}")
        return "\n".join([line, *(f"warning: {w}" for w in self.warnings)])


def steps(n: int, repeats: int, epochs: int, batch: int) -> int:
    """S = ceil(N·R / B) · E"""
    return -(-n * repeats // max(batch, 1)) * epochs


def step_time(kind: str, res: int, batch: int) -> tuple[float, bool]:
    """(seconds per step, is_estimate): measured point, else t0·(res/res0)²·(B/B0)."""
    if (kind, res, batch) in STEP_TIME:
        return STEP_TIME[(kind, res, batch)], False
    (_, r0, b0), t0 = next((k, v) for k, v in STEP_TIME.items() if k[0] == kind)
    return t0 * (res / r0) ** 2 * (batch / b0), True


def auto(n: int, epochs: int, views: int) -> tuple[int, int]:
    """(repeats, epochs) so N·R·E stays near `views`; big sets also lose epochs."""
    r = min(max(round(views / (n * epochs)), 1), MAX_REPEATS)
    if r == 1 and n * epochs > 1.5 * views:
        epochs = min(max(round(views / n), MIN_EPOCHS), epochs)
    return r, epochs


def train(kind: str, subject: str, n: int, res: int, batch: int, epochs: int,
          repeats: int = 0, download_gb: float = 0.0) -> TrainPlan:
    """Plan a run; repeats 0 = auto (epochs may shrink for big sets)."""
    if n < 1:
        raise ValueError("the dataset has no images")
    sub = SUBJECTS[subject]
    if repeats <= 0:
        repeats, epochs = auto(n, epochs, sub["views"][kind])
    s = steps(n, repeats, epochs, batch)
    t, est = step_time(kind, res, batch)
    warn = []
    if n < MIN_PHOTOS:
        warn.append(f"only {n} photos: the LoRA learns weakly and overfits fast (5+ recommended)")
    if sub["est"]:
        warn.append(f"'{subject}' settings are a starting guess, not yet measured")
    secs = s * t + OVERHEAD[kind] + download_gb / 0.1
    return TrainPlan(n, repeats, epochs, batch, s, secs, est or sub["est"] or download_gb > 0, warn)


def bucket(w: int, h: int, res: int, step: int) -> tuple[int, int]:
    """Bucket of an image: same aspect, area ≈ res², sides rounded down to `step`."""
    scale = math.sqrt(res * res / (w * h))
    return max(step, int(w * scale) // step * step), max(step, int(h * scale) // step * step)


def buckets(sizes: list[tuple[int, int]], kind: str, res: int) -> dict[tuple[int, int], int]:
    """{bucket: count}; a bucket with one image is worth a look."""
    out: dict[tuple[int, int], int] = {}
    for w, h in sizes:
        b = bucket(w, h, res, BUCKET_STEP[kind])
        out[b] = out.get(b, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def too_small(w: int, h: int, res: int) -> bool:
    """Shortest side under half the training size comes out soft."""
    return min(w, h) < 0.5 * res


def lora_scale(alpha: float, dim: int, strength: float = 1.0) -> float:
    return alpha / dim * strength


# ---- captioning ----
FP16_MAX_GB = 12.5
BYTES = {"fp16": 2.0, "nf4": 0.56}
WORD_BUDGET = {"sdxl": 55, "zimage": 350}


def caption_vram(params_b: float, quant: str) -> float:
    """GB ≈ P·bytes·1.15 (vision tower included in P)."""
    return params_b * BYTES[quant] * 1.15


def caption_quant(params_b: float) -> str:
    return "fp16" if caption_vram(params_b, "fp16") <= FP16_MAX_GB else "nf4"


def caption_batch(quant: str) -> int:
    return 4 if quant == "fp16" else 2


def vision_tokens(w: int, h: int, max_side: int = 768) -> int:
    """Qwen3-VL: 32 px per token after merge, image fit inside max_side."""
    s = min(1.0, max_side / max(w, h))
    return math.ceil(w * s / 32) * math.ceil(h * s / 32)


def over_budget(text: str, target: str) -> bool:
    return len(text.split()) > WORD_BUDGET[target]
