"""A stub Engine for local UI tests (SDXLR_FAKE_ENGINE=1): no torch, no GPU, no downloads.

Same surface worker.py uses from models.Engine, and it runs inside the real
worker process, so a fake UI run exercises the real pipe protocol: reader
thread, heartbeats, cancel, the cancel kill, the death message. Generation
draws a labelled placeholder at the requested size.

Test switches (environment):
  SDXLR_FAKE_CRASH=1     the first generate (once per work dir) kills the worker with exit code 9
  SDXLR_FAKE_SLOWLOAD=1  the first load (once per work dir) sleeps 20 s without noticing cancel
  SDXLR_FAKE_FAST=1      no sleeps at all (unit tests); SDXLR_FAKE_SCALE=0.1 makes every sleep that much shorter
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from PIL import Image, ImageDraw

from ..paths import RUN, z_key


@dataclass
class Job:
    prompt: str
    negative: str = ""
    steps: int = 25
    cfg: float = 5.0
    width: int = 896
    height: int = 1152
    seed: int = -1
    count: int = 1
    sampler: str = "dpmpp_2m_karras"
    clip_skip: int = 2
    loras: list[tuple[str, float]] = field(default_factory=list)
    image: Image.Image | None = None
    mask: Image.Image | None = None
    strength: float = 0.75
    on_step: Callable[[int, int], None] | None = None
    on_preview: Callable[[int, object], None] | None = None


def _once(name: str) -> bool:
    """True the first time in this work dir (a restarted worker must not repeat a test switch)."""
    f = RUN / f"fake_{name}.done"
    if f.exists():
        return False
    f.parent.mkdir(parents=True, exist_ok=True)
    f.touch()
    return True


def _sleep(secs: float) -> None:
    scale = 0.0 if os.environ.get("SDXLR_FAKE_FAST", "0") not in ("", "0") else float(os.environ.get("SDXLR_FAKE_SCALE", "1"))
    if scale > 0:
        time.sleep(secs * scale)


def _img(w: int, h: int, text: str) -> Image.Image:
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    rnd = random.Random(w * h)
    for _ in range(max(w, h) // 4):
        x, y = rnd.randrange(w), rnd.randrange(h)
        r = rnd.randrange(2, 14)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)))
    for i, line in enumerate(text.split("|")):
        d.text((12, 12 + i * 16), line[:90], fill=(255, 255, 255))
    return im


class _Pipe:
    """Stands in for a diffusers pipeline: no unet/transformer/... attributes, so model_parts gives []."""


class Engine:
    def __init__(self) -> None:
        self.pipe = None
        self.kind: str | None = None
        self.model_path: str | None = None
        self.vae_mode: str | None = None
        self.adapters: dict[str, str] = {}

    # ---- sidebar numbers the beat reports instead of real VRAM parts ----
    def beat_extra(self) -> dict:
        if self.kind == "zimage":
            return {"parts": [("DiT", 7.2), ("text encoder", 2.5), ("VAE", 0.2)], "alloc": 9.9, "reserved": 11.0, "rss": 1.6}
        if self.kind == "sdxl":
            return {"parts": [("UNet", 5.1), ("text encoder", 0.6), ("text encoder 2", 1.2), ("VAE", 0.4)],
                    "alloc": 7.3, "reserved": 8.5, "rss": 1.4}
        return {"parts": [], "alloc": 0.3, "reserved": 0.5, "rss": 0.4}

    def unload(self) -> None:
        self.pipe = None
        self.kind = self.model_path = self.vae_mode = None
        self.adapters = {}

    def _load(self, kind: str, model_path: str, vae: str | None, status, label: str) -> str:
        t = time.time()
        for stage, secs in (("Getting the model files", 0.4), ("Loading the model", 0.7), ("Moving to GPU", 0.3)):
            status(stage)
            if stage == "Loading the model" and os.environ.get("SDXLR_FAKE_SLOWLOAD") and _once("slowload"):
                time.sleep(20)  # a load that cannot be interrupted: only the client's kill stops it
            _sleep(secs)
        self.pipe, self.kind, self.model_path, self.vae_mode = _Pipe(), kind, model_path, vae
        return f"{label} in {time.time() - t:.0f}s | fake"

    def load(self, path: str, vae: str = "fp16fix", status: Callable[[str], None] = lambda s: None) -> str:
        return self._load("sdxl", path, vae, status, "loaded SDXL (fake)")

    def load_zimage(self, transformer: str, text_encoder: str, te_quant: str,
                    status: Callable[[str], None] = lambda s: None) -> str:
        return self._load("zimage", z_key(transformer, text_encoder, te_quant), None, status, "loaded Z-Image (fake)")

    def _use(self, loras) -> None:
        """Remember the LoRAs a request used, like the real adapter cache (the sidebar and the picker flow see them)."""
        for p, w in loras or []:
            if p and w != 0:
                self.adapters[p] = "fake"
                self.adapters.pop(next(iter(self.adapters))) if len(self.adapters) > 12 else None

    def make_preview(self, latents, path, done) -> None:
        """A placeholder preview: a gradient that changes with the step (the real engine decodes the latents)."""
        self._prev_n = getattr(self, "_prev_n", 0) + 1
        im = Image.linear_gradient("L").resize((256, 192)).convert("RGB")
        ImageDraw.Draw(im).text((8, 8), f"preview {self._prev_n}", fill=(255, 255, 255))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        im.save(tmp, "JPEG", quality=70)
        os.replace(tmp, path)
        done(path)

    def _steps(self, steps: int, on_step, secs: float = 12.0, on_preview=None) -> None:
        for i in range(1, steps + 1):
            if on_step:
                on_step(i, steps)
            if on_preview and i % 2 == 0 and i < steps:
                on_preview(i, None)
            _sleep(secs / steps)

    def generate(self, job: Job) -> tuple[list[Image.Image], int, str]:
        if self.pipe is None:
            raise RuntimeError("no model loaded")
        if os.environ.get("SDXLR_FAKE_CRASH") and _once("crash"):
            os._exit(9)  # like the kernel's OOM kill: no goodbye on the pipe
        t0 = time.time()
        self._use(job.loras)
        self._steps(max(1, int(job.steps)), job.on_step, on_preview=job.on_preview)
        seed = job.seed if job.seed >= 0 else random.randrange(2**31 - 1)
        w, h = job.image.size if job.image is not None else (job.width, job.height)
        images = [_img(w, h, f"{job.prompt[:60]} | fake") for _ in range(max(1, job.count))]
        return images, seed, f"{time.time() - t0:.1f}s | fake | seed {seed}"

    def inpaint_crop(self, job: Job, crop: Image.Image, mask: Image.Image) -> Image.Image:
        if self.pipe is None:
            raise RuntimeError("no model loaded")
        self._use(job.loras)
        self._steps(max(1, int(job.steps)), job.on_step, secs=4.0, on_preview=job.on_preview)
        return _img(crop.width, crop.height, "edit fake")

    def upscale(self, job: Job, scale: float = 2.0, tile: int = 1024, overlap: int = 128,
                on_tile=None) -> tuple[Image.Image, int, str]:
        if self.pipe is None:
            raise RuntimeError("no model loaded")
        t0 = time.time()
        self._use(job.loras)
        w, h = int(job.image.width * scale), int(job.image.height * scale)
        tiles = max(1, (w // tile + 1) * (h // tile + 1))
        self._steps(tiles, on_tile, secs=8.0)
        seed = job.seed if job.seed >= 0 else random.randrange(2**31 - 1)
        return _img(w, h, f"upscale {scale:g}x | fake"), seed, f"{time.time() - t0:.1f}s | {w}x{h} fake | seed {seed}"
