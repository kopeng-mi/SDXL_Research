"""A fake engine for local UI tests (SDXLR_FAKE_ENGINE=1): no torch, no GPU.

Same surface as engine.client.EngineClient. Loads take about a second,
generation draws a labelled placeholder at the requested size, everything can
be cancelled, and results are saved exactly like the real worker saves them.
SDXLR_FAKE_CRASH=1 kills the "worker" on the first generate, to test the
death message and the restart on the next Generate.
"""

from __future__ import annotations

import os
import random
import time

from PIL import Image, ImageDraw

from .client import DEATH_MSG, WorkerDied
from .worker import open_source, save_images

FAKE_BEAT = {"type": "beat", "parts": [], "alloc": 0.3, "reserved": 0.5, "rss": 0.4}


class FakeEngineClient:
    needs_prefetch = False  # never download real model files from a fake run

    def __init__(self) -> None:
        self.kind = None
        self.model_path = None
        self.vae_mode = None
        self.adapters = {}
        self.beat = dict(FAKE_BEAT)
        self._busy = False
        self._dead = False
        self._crashed_once = False

    @property
    def alive(self) -> bool:
        return not self._dead

    @property
    def loaded(self) -> bool:
        return self.model_path is not None

    def _reset(self) -> None:
        self.kind = self.model_path = self.vae_mode = None
        self.adapters = {}
        self.beat = dict(FAKE_BEAT)

    # ---- process stand-ins ----
    def kill(self) -> None:
        self._dead = True
        self._reset()

    def unload(self) -> None:
        if self._busy:
            raise RuntimeError("the engine is busy")
        self._reset()

    def shutdown(self) -> None:
        self.kill()

    def _cancel_current(self) -> None:  # ops notice task.cancelled in their loops
        pass

    # ---- ops ----
    def call(self, op: str, args: dict | None = None, task=None) -> dict:
        args = args or {}
        if self._dead:  # a killed worker restarts on the next request
            self._dead = False
            self._reset()
        self._busy = True
        try:
            return {"load": self._load, "unload": self._unload, "status": self._status,
                    "generate": self._generate, "upscale": self._upscale}[op](args, task)
        finally:
            self._busy = False

    def _sleep(self, secs: float, task) -> None:
        end = time.time() + secs
        while time.time() < end:
            if task is not None:
                task.check()
            time.sleep(0.05)

    def _load(self, args, task):
        t = time.time()
        for stage, secs in (("Getting the model files", 0.4), ("Loading the model", 0.7), ("Moving to GPU", 0.3)):
            task.set_stage(stage)
            self._sleep(secs, task)
        if args.get("kind") == "zimage":
            self.kind, self.model_path = "zimage", "|".join(["zimage", args["transformer"],
                                                             args["text_encoder"], args["te_quant"]])
            self.beat = {**FAKE_BEAT, "kind": self.kind, "model_path": self.model_path,
                         "parts": [("DiT", 7.2), ("text encoder", 2.5), ("VAE", 0.2)],
                         "alloc": 9.9, "reserved": 11.0, "rss": 1.6}
            label = "loaded Z-Image (fake)"
        else:
            self.kind, self.model_path, self.vae_mode = "sdxl", args["path"], args.get("vae", "fp16fix")
            self.beat = {**FAKE_BEAT, "kind": self.kind, "model_path": self.model_path,
                         "parts": [("UNet", 5.1), ("text encoder", 0.6), ("text encoder 2", 1.2), ("VAE", 0.4)],
                         "alloc": 7.3, "reserved": 8.5, "rss": 1.4}
            label = "loaded SDXL (fake)"
        return {"label": f"{label} in {time.time() - t:.0f}s | fake"}

    def _unload(self, args, task):
        self._reset()
        return {}

    def _status(self, args, task):
        return {"kind": self.kind, "model_path": self.model_path, "vae_mode": self.vae_mode}

    def _img(self, w: int, h: int, text: str) -> Image.Image:
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

    def _generate(self, args, task):
        if os.environ.get("SDXLR_FAKE_CRASH") and not self._crashed_once:
            self._crashed_once = True
            self._dead = True
            self._reset()
            raise WorkerDied(DEATH_MSG)
        t0 = time.time()
        j = args["job"]
        steps = max(1, int(j.get("steps", 20)))
        task.set_stage("Generating")
        for i in range(1, steps + 1):
            task.step(i, steps)
            self._sleep(12 / steps, task)
        seed = int(j.get("seed", -1)) if int(j.get("seed", -1)) >= 0 else random.randrange(2**31 - 1)
        if j.get("image"):
            img, _ = open_source(j["image"], j.get("boxes") or [], 8)
            w, h = img.size
        else:
            w, h = int(j["width"]), int(j["height"])
        count = int(j.get("count", 1))
        meta = {**(args.get("meta") or {}), "seconds": round(time.time() - t0, 1)}
        paths = save_images([self._img(w, h, f"{j['prompt'][:60]} | fake") for _ in range(count)], meta, seed)
        return {"paths": paths, "seed": seed, "info": f"{meta['seconds']}s | fake | seed {seed}", "size": [w, h]}

    def _upscale(self, args, task):
        t0 = time.time()
        j = args["job"]
        img, _ = open_source(j["image"], [], 8)
        scale = float(args.get("scale", 2.0))
        w, h = int(img.width * scale), int(img.height * scale)
        tiles = max(1, (w // 1024 + 1) * (h // 1024 + 1))
        for i in range(1, tiles + 1):
            task.step(i, tiles)
            self._sleep(8 / tiles, task)
        seed = int(j.get("seed", -1)) if int(j.get("seed", -1)) >= 0 else random.randrange(2**31 - 1)
        big = self._img(w, h, f"upscale {scale:g}x | fake").resize((w, h))
        meta = {**(args.get("meta") or {}), "seconds": round(time.time() - t0, 1)}
        paths = save_images([big], meta, seed)
        return {"paths": paths, "seed": seed, "info": f"{meta['seconds']}s | {w}x{h} fake | seed {seed}", "size": [w, h]}
