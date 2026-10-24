"""SDXL generation engine with explicit resource handling for the Colab T4.

Rules (15 GB VRAM, ~12.7 GB RAM, no fast bf16):
  * one base model at a time; switching fully unloads the old one first
  * fp16 everywhere; the fp16-fix VAE avoids NaNs without an fp32 VAE
  * each LoRA file is loaded once and cached as a named adapter; changing a
    weight only calls set_adapters (no reload); least recently used adapters
    are dropped past MAX_ADAPTERS
  * prompt embeddings are cached, so seed/step changes skip re-encoding
  * out of memory: free cache, turn on VAE tiling, retry once
  * inpaint and image-to-image reuse the loaded weights (from_pipe, fp16:
    without torch_dtype from_pipe casts the shared UNet/VAE to fp32 in place
    and doubles VRAM)
  * after each image the CUDA cache is released, so the VRAM reading shows
    what the model really holds (about 5 GB of activations otherwise stay reserved)
  * upscale = enlarge, then refine overlapping tiles with image-to-image and
    blend them; every tile is about one normal image, so any size fits a T4
"""

from __future__ import annotations

import gc
import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from ..paths import FP16_VAE, SAMPLERS, Z_ENCODERS, Z_MODELS, Z_REPO, z_key
from ..prompt import encode_sdxl

MAX_ADAPTERS = 12


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
    loras: list[tuple[str, float]] = field(default_factory=list)  # (path, weight)
    image: Image.Image | None = None  # source: with mask = change an area, without = change the whole image
    mask: Image.Image | None = None  # white = repaint
    strength: float = 0.75
    on_step: Callable[[int, int], None] | None = None  # (done, total) after each denoising step


def _fix_text_encoder_lora_keys(pipe) -> None:
    """transformers 5 dropped the `text_model.` wrapper from CLIPTextModel, but
    diffusers 0.40 still maps kohya keys to `text_encoder.text_model.encoder...`,
    so text-encoder LoRA weights match no module and loading fails. Strip the
    wrapper from the converted keys (and alphas) for each encoder whose module
    names lack it. Checked per encoder: CLIPTextModel (te1) changed, while
    CLIPTextModelWithProjection (te2) still uses text_model.*; and by module
    names, not hasattr, because transformers 5 keeps a `text_model` alias."""
    flat = [prefix for prefix, enc in (("text_encoder", pipe.text_encoder), ("text_encoder_2", pipe.text_encoder_2))
            if not any(n.startswith("text_model.") for n, _ in enc.named_modules())]
    if not flat:
        return
    orig = pipe.lora_state_dict
    pat = re.compile(rf"^({'|'.join(flat)})\.text_model\.")

    def fixed(*args, **kwargs):
        sd, alphas, *rest = orig(*args, **kwargs)
        ren = lambda d: d if d is None else {pat.sub(r"\1.", k): v for k, v in d.items()}
        return (ren(sd), ren(alphas), *rest)

    pipe.lora_state_dict = fixed


def _step_cb(job: Job) -> dict:
    """callback_on_step_end that reports (done, total); total is the pipeline's own step count."""
    if job.on_step is None:
        return {}

    def cb(pipe, i, t, kw):
        job.on_step(i + 1, getattr(pipe, "_num_timesteps", None) or job.steps)
        return kw

    return {"callback_on_step_end": cb}


def vram() -> str:
    if not torch.cuda.is_available():
        return "no GPU"
    free, total = torch.cuda.mem_get_info()
    return f"VRAM {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GB"


def ram() -> str:
    import psutil
    m = psutil.virtual_memory()
    return f"RAM {(m.total - m.available) / 2**30:.1f}/{m.total / 2**30:.1f} GB"


class Engine:
    def __init__(self) -> None:
        self.pipe = None
        self.kind: str | None = None  # "sdxl" | "zimage"
        self.model_path: str | None = None
        self.vae_mode: str | None = None
        self.adapters: OrderedDict[str, str] = OrderedDict()  # path -> adapter name
        self._embeds: OrderedDict[tuple, dict] = OrderedDict()
        self._sampler: str | None = None
        self._inpaint = None
        self._img2img = None

    # ---- model ----
    def unload(self) -> None:
        self.pipe = self._inpaint = self._img2img = None
        self.kind = self.model_path = self._sampler = None
        self.adapters.clear()
        self._embeds.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, path: str, vae: str = "fp16fix", status: Callable[[str], None] = lambda s: None) -> str:
        if self.pipe is not None and self.model_path == path and self.vae_mode == vae:
            return f"already loaded: {Path(path).name}"
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline

        self.unload()  # free the old model before reading the new one
        t = time.time()
        kw = {"torch_dtype": torch.float16, "use_safetensors": True}
        if vae == "fp16fix":
            status("Getting the fp16 VAE")
            kw["vae"] = AutoencoderKL.from_pretrained(FP16_VAE, torch_dtype=torch.float16)
        status("Reading the checkpoint")
        pipe = StableDiffusionXLPipeline.from_single_file(path, **kw)
        if vae != "fp16fix":
            pipe.vae.config.force_upcast = True  # model VAE in fp16 NaNs; upcast only for decode
        pipe.set_progress_bar_config(disable=True)
        _fix_text_encoder_lora_keys(pipe)
        status("Moving to GPU")
        self.pipe = pipe.to("cuda")
        self.kind, self.model_path, self.vae_mode = "sdxl", path, vae
        return f"loaded {Path(path).name} in {time.time() - t:.0f}s | {vram()}"

    def load_zimage(self, transformer: str = Z_MODELS["turbo Q8"], text_encoder: str = Z_ENCODERS["official"],
                    te_quant: str = "nf4", status: Callable[[str], None] = lambda s: None) -> str:
        """Z-Image: GGUF (or safetensors) transformer + Qwen3 text encoder, fp16 with the T4 overflow fix.

        transformer:  org/repo/file, URL or path (.gguf recommended on T4)
        text_encoder: 'org/repo' or 'org/repo:subfolder'
        te_quant:     nf4 (~2.5 GB) | int8 (~4.4 GB) | fp16 (~8 GB, may not fit next to Q8)
        """
        key = z_key(transformer, text_encoder, te_quant)
        if self.pipe is not None and self.model_path == key:
            return "already loaded"
        from diffusers import GGUFQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel
        from transformers import AutoModel, BitsAndBytesConfig

        from ..fetch import MODELS, fetch
        from ..zimage_fp16 import patch

        self.unload()
        t = time.time()
        dt = torch.float16
        status("Getting the model file")
        src = fetch(transformer, MODELS / "zimage")
        status("Loading the model")
        kw = {"torch_dtype": dt, "config": Z_REPO, "subfolder": "transformer"}
        if src.suffix == ".gguf":
            kw["quantization_config"] = GGUFQuantizationConfig(compute_dtype=dt)
        tr = ZImageTransformer2DModel.from_single_file(str(src), **kw)

        repo, _, sub = text_encoder.partition(":")
        q = None
        if te_quant == "nf4":
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt)
        elif te_quant == "int8":
            q = BitsAndBytesConfig(load_in_8bit=True)
        status(f"Getting the text encoder ({te_quant})")
        te = AutoModel.from_pretrained(repo, subfolder=sub or "", dtype=dt, quantization_config=q)

        status("Getting VAE and configs")
        pipe = ZImagePipeline.from_pretrained(Z_REPO, transformer=tr, text_encoder=te, torch_dtype=dt)
        pipe.set_progress_bar_config(disable=True)
        status("Moving to GPU")
        self.pipe = pipe.to("cuda")
        patch(self.pipe.transformer)  # fp16 on T4 otherwise gives black images
        self.kind, self.model_path, self.vae_mode = "zimage", key, None
        return f"loaded Z-Image ({Path(str(src)).name}, TE {te_quant}) in {time.time() - t:.0f}s | {vram()}"

    def _set_sampler(self, name: str) -> None:
        if name == self._sampler:
            return
        import diffusers
        cls, kw = SAMPLERS[name]
        self.pipe.scheduler = getattr(diffusers, cls).from_config(self.pipe.scheduler.config, **kw)
        self._sampler = name

    # ---- LoRAs ----
    def _adapter(self, path: str) -> str:
        if path in self.adapters:
            self.adapters.move_to_end(path)
            return self.adapters[path]
        while len(self.adapters) >= MAX_ADAPTERS:
            _, old = self.adapters.popitem(last=False)
            self.pipe.delete_adapters([old])
        name = "l" + hashlib.md5(path.encode()).hexdigest()[:10]
        p = Path(path)
        try:
            self.pipe.load_lora_weights(str(p.parent), weight_name=p.name, adapter_name=name)
        except Exception:
            # a half-loaded adapter keeps its name taken; remove it so a retry works
            try:
                self.pipe.delete_adapters([name])
            except Exception:
                pass
            raise
        self.adapters[path] = name
        return name

    def set_loras(self, loras: list[tuple[str, float]]) -> None:
        active = [(self._adapter(p), float(w)) for p, w in loras if p and w != 0]
        if active:
            self.pipe.enable_lora()
            self.pipe.set_adapters([n for n, _ in active], adapter_weights=[w for _, w in active])
        elif self.adapters:
            self.pipe.disable_lora()

    # ---- generate ----
    def _encode(self, job: Job, lora_key: tuple) -> dict:
        # LoRAs also patch the text encoders, so they are part of the key
        key = (job.prompt, job.negative, job.clip_skip, lora_key)
        if key not in self._embeds:
            self._embeds[key] = encode_sdxl(self.pipe, job.prompt, job.negative, job.clip_skip)
            while len(self._embeds) > 8:
                self._embeds.popitem(last=False)
        self._embeds.move_to_end(key)
        return self._embeds[key]

    def _run_zimage(self, job: Job, seed: int) -> list[Image.Image]:
        # Qwen text encoder: no A1111 weights / clip skip; negative only matters with cfg > 1
        self.set_loras(job.loras)
        gen = [torch.Generator("cpu").manual_seed(seed + i) for i in range(job.count)]
        common = dict(prompt=[job.prompt] * job.count, num_inference_steps=job.steps, guidance_scale=job.cfg,
                      generator=gen, **_step_cb(job))
        if job.cfg > 1 and job.negative:
            common["negative_prompt"] = [job.negative] * job.count
        if job.image is not None and job.mask is not None:
            if self._inpaint is None:
                from diffusers import ZImageInpaintPipeline
                # GGUF/bnb weights can't be cast, which from_pipe tries; reuse the parts as they are
                self._inpaint = ZImageInpaintPipeline(**self.pipe.components)
            img = job.image.convert("RGB")
            w, h = (img.width // 16) * 16, (img.height // 16) * 16
            return self._inpaint(image=img.resize((w, h)), mask_image=job.mask.convert("L").resize((w, h)),
                                 width=w, height=h, strength=job.strength, **common).images
        if job.image is not None:
            if self._img2img is None:
                from diffusers import ZImageImg2ImgPipeline
                self._img2img = ZImageImg2ImgPipeline(**self.pipe.components)
            img = job.image.convert("RGB")
            w, h = (img.width // 16) * 16, (img.height // 16) * 16
            return self._img2img(image=img.resize((w, h)), width=w, height=h, strength=job.strength, **common).images
        return self.pipe(width=job.width, height=job.height, **common).images

    def _run(self, job: Job, seed: int) -> list[Image.Image]:
        if self.kind == "zimage":
            return self._run_zimage(job, seed)
        lora_key = tuple(sorted((p, round(w, 4)) for p, w in job.loras if p and w))
        self.set_loras(job.loras)
        self._set_sampler(job.sampler)
        emb = self._encode(job, lora_key)
        rep = {k: v.repeat(job.count, *[1] * (v.dim() - 1)) for k, v in emb.items()}
        gen = [torch.Generator("cpu").manual_seed(seed + i) for i in range(job.count)]
        common = dict(num_inference_steps=job.steps, guidance_scale=job.cfg, generator=gen, **rep, **_step_cb(job))
        if job.image is not None and job.mask is None:
            if self._img2img is None:
                from diffusers import StableDiffusionXLImg2ImgPipeline
                self._img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(self.pipe, torch_dtype=torch.float16)
            img = job.image.convert("RGB")
            return self._img2img(image=img.resize(((img.width // 8) * 8, (img.height // 8) * 8)),
                                 strength=job.strength, **common).images
        if job.image is not None:
            if self._inpaint is None:
                from diffusers import StableDiffusionXLInpaintPipeline
                self._inpaint = StableDiffusionXLInpaintPipeline.from_pipe(self.pipe, torch_dtype=torch.float16)
            img = job.image.convert("RGB")
            w, h = (img.width // 8) * 8, (img.height // 8) * 8
            return self._inpaint(image=img.resize((w, h)), mask_image=job.mask.convert("L").resize((w, h)),
                                 width=w, height=h, strength=job.strength, **common).images
        return self.pipe(width=job.width, height=job.height, **common).images

    def generate(self, job: Job) -> tuple[list[Image.Image], int, str]:
        if self.pipe is None:
            raise RuntimeError("no model loaded")
        seed = job.seed if job.seed >= 0 else int(torch.randint(0, 2**31 - 1, (1,)))
        t = time.time()
        try:
            images = self._run(job, seed)
        except torch.OutOfMemoryError:
            self._embeds.clear()
            gc.collect()
            torch.cuda.empty_cache()
            self.pipe.vae.enable_tiling()
            images = self._run(job, seed)  # a second OOM propagates to the caller
        info = f"{time.time() - t:.1f}s | seed {seed} | {vram()} | {ram()}"
        torch.cuda.empty_cache()  # give back the activations' memory between images
        return images, seed, info

    def upscale(self, job: Job, scale: float = 2.0, tile: int = 1024, overlap: int = 128,
                on_tile=None) -> tuple[Image.Image, int, str]:
        """Enlarge job.image by `scale`, then refine overlapping tiles with image-to-image (job.prompt,
        job.strength) and blend them with feathered edges. on_tile(done, total) reports progress."""
        import numpy as np
        if self.pipe is None:
            raise RuntimeError("no model loaded")
        t = time.time()
        seed = job.seed if job.seed >= 0 else int(torch.randint(0, 2**31 - 1, (1,)))
        m = 16 if self.kind == "zimage" else 8
        src = job.image.convert("RGB")
        W, H = (int(src.width * scale) // m) * m, (int(src.height * scale) // m) * m
        big = src.resize((W, H), Image.LANCZOS)
        if W <= tile and H <= tile:
            boxes = [(0, 0, W, H)]
        else:
            step = tile - overlap
            xs = sorted({*range(0, max(W - tile, 0) + 1, step), max(W - tile, 0)})
            ys = sorted({*range(0, max(H - tile, 0) + 1, step), max(H - tile, 0)})
            boxes = [(x, y, min(x + tile, W), min(y + tile, H)) for y in ys for x in xs]
        acc = np.zeros((H, W, 3), np.float32)
        wsum = np.zeros((H, W, 1), np.float32)
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            sub = Job(**{**job.__dict__, "image": big.crop((x0, y0, x1, y1)), "mask": None, "count": 1,
                         "seed": seed, "on_step": None})
            out = self._run(sub, seed)[0].resize((x1 - x0, y1 - y0), Image.LANCZOS)
            # weight falls off linearly over the overlap, so seams blend
            wx = np.minimum(np.arange(x1 - x0) + 1, np.arange(x1 - x0)[::-1] + 1).clip(max=overlap) / overlap
            wy = np.minimum(np.arange(y1 - y0) + 1, np.arange(y1 - y0)[::-1] + 1).clip(max=overlap) / overlap
            wgt = (wy[:, None] * wx[None, :])[..., None].astype(np.float32)
            acc[y0:y1, x0:x1] += np.asarray(out, np.float32) * wgt
            wsum[y0:y1, x0:x1] += wgt
            if on_tile:
                on_tile(i + 1, len(boxes))
        torch.cuda.empty_cache()
        res = Image.fromarray((acc / np.maximum(wsum, 1e-6)).clip(0, 255).astype(np.uint8))
        return res, seed, f"{time.time() - t:.1f}s | {W}x{H} from {len(boxes)} tiles | seed {seed} | {vram()}"
