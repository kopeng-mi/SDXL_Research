"""SDXL generation engine with explicit resource handling for the Colab T4.

Rules (15 GB VRAM, ~12.7 GB RAM, no fast bf16):
  * one base model at a time; switching fully unloads the old one first
  * fp16 everywhere; the fp16-fix VAE avoids NaNs without an fp32 VAE
  * each LoRA file is loaded once and cached as a named adapter; changing a
    weight only calls set_adapters (no reload); least recently used adapters
    are dropped past MAX_ADAPTERS
  * prompt embeddings are cached, so seed/step changes skip re-encoding
  * out of memory: free cache, turn on VAE tiling, retry once
  * inpaint reuses the loaded weights (from_pipe), no second copy
"""

from __future__ import annotations

import gc
import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

from .prompt import encode_sdxl

MAX_ADAPTERS = 12
FP16_VAE = "madebyollin/sdxl-vae-fp16-fix"

SAMPLERS = {
    "dpmpp_2m_karras": ("DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True}),
    "dpmpp_2m_sde_karras": ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True}),
    "dpmpp_sde_karras": ("DPMSolverSinglestepScheduler", {"use_karras_sigmas": True}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "euler": ("EulerDiscreteScheduler", {}),
    "lcm": ("LCMScheduler", {}),
}


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
    image: Image.Image | None = None  # inpaint source
    mask: Image.Image | None = None  # white = repaint
    strength: float = 0.75


def _fix_text_encoder_lora_keys(pipe) -> None:
    """transformers 5 dropped the `text_model.` wrapper from CLIPTextModel, but
    diffusers 0.40 still maps kohya keys to `text_encoder.text_model.encoder...`,
    so text-encoder LoRA weights match no module and loading fails. Strip the
    wrapper from the converted keys (and alphas) when the module names lack it.
    (Module names, not hasattr: transformers 5 keeps a `text_model` alias.)"""
    if any(n.startswith("text_model.") for n, _ in pipe.text_encoder.named_modules()):
        return
    orig = pipe.lora_state_dict
    pat = re.compile(r"^(text_encoder(?:_2)?)\.text_model\.")

    def fixed(*args, **kwargs):
        sd, alphas, *rest = orig(*args, **kwargs)
        ren = lambda d: d if d is None else {pat.sub(r"\1.", k): v for k, v in d.items()}
        return (ren(sd), ren(alphas), *rest)

    pipe.lora_state_dict = fixed


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
        self.model_path: str | None = None
        self.vae_mode: str | None = None
        self.adapters: OrderedDict[str, str] = OrderedDict()  # path -> adapter name
        self._embeds: OrderedDict[tuple, dict] = OrderedDict()
        self._sampler: str | None = None
        self._inpaint = None

    # ---- model ----
    def unload(self) -> None:
        self.pipe = self._inpaint = None
        self.model_path = self._sampler = None
        self.adapters.clear()
        self._embeds.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, path: str, vae: str = "fp16fix") -> str:
        if self.pipe is not None and self.model_path == path and self.vae_mode == vae:
            return f"already loaded: {Path(path).name}"
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline

        self.unload()  # free the old model before reading the new one
        t = time.time()
        kw = {"torch_dtype": torch.float16, "use_safetensors": True}
        if vae == "fp16fix":
            kw["vae"] = AutoencoderKL.from_pretrained(FP16_VAE, torch_dtype=torch.float16)
        pipe = StableDiffusionXLPipeline.from_single_file(path, **kw)
        if vae != "fp16fix":
            pipe.vae.config.force_upcast = True  # model VAE in fp16 NaNs; upcast only for decode
        pipe.set_progress_bar_config(disable=True)
        _fix_text_encoder_lora_keys(pipe)
        self.pipe = pipe.to("cuda")
        self.model_path, self.vae_mode = path, vae
        return f"loaded {Path(path).name} in {time.time() - t:.0f}s | {vram()}"

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
        self.pipe.load_lora_weights(str(p.parent), weight_name=p.name, adapter_name=name)
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

    def _run(self, job: Job, seed: int) -> list[Image.Image]:
        lora_key = tuple(sorted((p, round(w, 4)) for p, w in job.loras if p and w))
        self.set_loras(job.loras)
        self._set_sampler(job.sampler)
        emb = self._encode(job, lora_key)
        rep = {k: v.repeat(job.count, *[1] * (v.dim() - 1)) for k, v in emb.items()}
        gen = [torch.Generator("cpu").manual_seed(seed + i) for i in range(job.count)]
        common = dict(num_inference_steps=job.steps, guidance_scale=job.cfg, generator=gen, **rep)
        if job.image is not None:
            if self._inpaint is None:
                from diffusers import StableDiffusionXLInpaintPipeline
                self._inpaint = StableDiffusionXLInpaintPipeline.from_pipe(self.pipe)
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
        return images, seed, f"{time.time() - t:.1f}s | seed {seed} | {vram()} | {ram()}"
