"""SDXL generation engine with explicit resource handling for the Colab T4.

Rules (15 GB VRAM, ~12.7 GB RAM, no fast bf16):
  * one base model at a time; switching fully unloads the old one first
  * fp16 everywhere; the fp16-fix VAE avoids NaNs without an fp32 VAE
  * each LoRA file is loaded once and cached as a named adapter; changing a
    weight only calls set_adapters (no reload); least recently used adapters
    are dropped past LORA_BUDGET_MB (sized from the file, not counted)
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
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from ..paths import CONTROLNETS, FP16_VAE, MODELS, PREVIEWS, SAMPLERS, Z_ENCODERS, Z_MODELS, Z_REPO, te_prequantized, z_key
from .keys import needs_tiling, reuse_plan, z_embed_key
from .worker import UserError  # a plain sentence for the user, not an engine error
from .. import loraformat
from ..prompt import encode_sdxl

LORA_BUDGET_MB = 1536  # VRAM the cached LoRA adapters may hold (a count of 12 ignores rank)


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
    on_preview: Callable[[int, object], None] | None = None  # (step, latents) every 2nd step, for live previews


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
    """callback_on_step_end that reports (done, total) and hands every 2nd step's latents to the preview;
    total is the pipeline's own step count."""
    if job.on_step is None and job.on_preview is None:
        return {}

    def cb(pipe, i, t, kw):
        total = getattr(pipe, "_num_timesteps", None) or job.steps
        if job.on_step:
            job.on_step(i + 1, total)
        if job.on_preview and "latents" in kw and (i + 1) % 2 == 0 and i + 1 < total:
            job.on_preview(i + 1, kw["latents"])
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
        self.adapter_mb: dict[str, float] = {}  # path -> size from the file header
        self._embeds: OrderedDict[tuple, dict] = OrderedDict()
        self._sampler: str | None = None
        self._inpaint = None
        self._img2img = None
        self._prev = None  # live preview decoder (engine/preview.py), made on first use
        self._cn = None  # (name, ZImageControlNetModel): one ControlNet in VRAM at a time
        self._cn_pipes: dict = {}
        self._fused: tuple | None = None  # SDXL LoRA set merged into the weights (default; SDXLR_FUSE=0 = off)

    # ---- model ----
    def unload(self) -> None:
        self.pipe = self._inpaint = self._img2img = None
        self._cn = None
        self._cn_pipes = {}
        self._fused = None
        self.kind = self.model_path = self._sampler = None
        self.adapters.clear()
        self.adapter_mb.clear()
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
        if is_vpred(path):  # v-pred models (NoobAI v-pred) also need zero-terminal-SNR timesteps
            pipe.scheduler = pipe.scheduler.from_config(pipe.scheduler.config, prediction_type="v_prediction",
                                                        rescale_betas_zero_snr=True, timestep_spacing="trailing")
        pipe.set_progress_bar_config(disable=True)
        _fix_text_encoder_lora_keys(pipe)
        status("Moving to GPU")
        self.pipe = pipe.to("cuda")
        self.kind, self.model_path, self.vae_mode = "sdxl", path, vae
        return f"loaded {Path(path).name} in {time.time() - t:.0f}s | {vram()}"

    def load_zimage(self, transformer: str = Z_MODELS["turbo 4-bit (unsloth)"], text_encoder: str = Z_ENCODERS["official"],
                    te_quant: str = "nf4", status: Callable[[str], None] = lambda s: None) -> str:
        """Z-Image: GGUF (or safetensors) transformer + Qwen3 text encoder, fp16 with the T4 overflow fix.

        transformer:  org/repo/file, URL or path (.gguf recommended on T4)
        text_encoder: 'org/repo' or 'org/repo:subfolder'; the unsloth bnb-4bit one is pre-quantized
        te_quant:     nf4 (~2.5 GB) | int8 (~4.4 GB) | fp16 (~8 GB, may not fit next to Q8); ignored when pre-quantized
        Changing only the transformer or only the text encoder keeps the other loaded part (no re-quantizing, no re-read).
        """
        key = z_key(transformer, text_encoder, te_quant)
        if self.pipe is not None and self.model_path == key:
            return "already loaded"
        from diffusers import GGUFQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel
        from transformers import AutoModel, BitsAndBytesConfig

        from ..fetch import MODELS, fetch
        from ..zimage_fp16 import patch

        keep_tr, keep_te = reuse_plan(self.model_path if self.kind == "zimage" else None, transformer, text_encoder, te_quant)
        old = self.pipe
        old_tr = old.transformer if old is not None and keep_tr else None
        old_te = old.text_encoder if old is not None and keep_te else None
        if old_tr is not None and self.adapters:  # adapters sit inside the transformer: drop them, they reload when used
            try:
                old.delete_adapters(list(self.adapters.values()))
            except Exception:
                pass
        embeds = self._embeds if old_te is not None else None  # the embeddings depend on the text encoder only
        self.unload()  # frees everything except the parts held by old_tr / old_te
        del old
        if embeds is not None:
            self._embeds = embeds
        t = time.time()
        dt = torch.float16
        if old_tr is not None:
            status("Keeping the loaded model")
            tr, name = old_tr, "kept"
        elif ":" in transformer and not Path(transformer).exists():  # 'org/repo:subfolder', e.g. unsloth's bnb-4bit DiT
            repo, _, sub = transformer.partition(":")
            status("Loading the model (pre-quantized)")
            tr = ZImageTransformer2DModel.from_pretrained(repo, subfolder=sub, torch_dtype=dt)
            self._fp16_compute(tr)  # stored with bf16 compute, which the T4 can't do fast
            name = f"{repo.split('/')[-1]} ({sub})"
        else:
            status("Getting the model file")
            src = fetch(transformer, MODELS / "zimage")
            status("Loading the model")
            kw = {"torch_dtype": dt, "config": Z_REPO, "subfolder": "transformer"}
            cache = src.parent / f"{src.stem}-nf4"  # a custom bf16/fp16/fp8 checkpoint, quantized once
            if src.suffix == ".gguf":
                kw["quantization_config"] = GGUFQuantizationConfig(compute_dtype=dt)
                tr = ZImageTransformer2DModel.from_single_file(str(src), **kw)
            elif (cache / "config.json").exists():
                tr = ZImageTransformer2DModel.from_pretrained(cache, torch_dtype=dt)
            else:  # full size (12 GB bf16) does not fit next to the text encoder: nf4 ≈ 3.4 GB
                from diffusers import BitsAndBytesConfig as DiffusersBnb
                status("Quantizing the model to 4-bit (once; cached next to the file)")
                kw["quantization_config"] = DiffusersBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                         bnb_4bit_compute_dtype=dt)
                tr = ZImageTransformer2DModel.from_single_file(str(src), **kw)
                try:
                    tr.save_pretrained(cache)
                except Exception as e:  # a full disk only costs the cache
                    print(f"could not cache the 4-bit model ({type(e).__name__}: {e})", flush=True)
            status("Moving the model to GPU")
            if src.suffix == ".gguf":  # bnb models are placed on the GPU while loading
                tr = tr.to("cuda")  # its RAM is free before the text encoder loads (Q8 peaked at 9.6 of 12.7 GB on T4)
            gc.collect()
            name = Path(str(src)).name
        if old_te is not None:
            te = old_te
        else:
            repo, _, sub = text_encoder.partition(":")
            prequant = te_prequantized(text_encoder)
            q = None
            if prequant:
                pass  # ships its own bnb config; its modules stay as saved
            elif te_quant == "nf4":
                q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt)
            elif te_quant == "int8":
                q = BitsAndBytesConfig(load_in_8bit=True)
            status(f"Getting the text encoder ({'pre-quantized' if prequant else te_quant})")
            # the unsloth file is a Qwen3ForCausalLM checkpoint (every key starts with "model."); AutoModel builds the
            # base Qwen3Model, and transformers strips the base-model prefix when loading a head checkpoint into it
            te = AutoModel.from_pretrained(repo, subfolder=sub or "", dtype=dt, quantization_config=q)
            if prequant:
                self._fp16_compute(te)

        status("Getting VAE and configs")
        pipe = ZImagePipeline.from_pretrained(Z_REPO, transformer=tr, text_encoder=te, torch_dtype=dt)
        pipe.set_progress_bar_config(disable=True)
        status("Moving to GPU")
        self.pipe = pipe.to("cuda")
        if old_tr is None:
            patch(self.pipe.transformer)  # fp16 on T4 otherwise gives black images
        self.kind, self.model_path, self.vae_mode = "zimage", key, None
        kept = " (kept " + " and ".join(n for n, k in (("transformer", old_tr), ("text encoder", old_te)) if k is not None) + ")" \
            if old_tr is not None or old_te is not None else ""
        return f"loaded Z-Image ({name}, TE {'pre-quantized' if te_prequantized(text_encoder) else te_quant}){kept} in {time.time() - t:.0f}s | {vram()}"

    @staticmethod
    def _fp16_compute(te) -> None:
        """unsloth's pre-quantized encoder and DiT store bnb_4bit_compute_dtype bfloat16, which a T4 cannot do fast
        (or at all): run every 4-bit Linear in fp16."""
        try:
            import bitsandbytes as bnb
        except ImportError:
            return
        for m in te.modules():
            if isinstance(m, bnb.nn.Linear4bit):
                m.compute_dtype = torch.float16
                qs = getattr(m.weight, "quant_state", None)
                if qs is not None:
                    qs.dtype = torch.float16

    def _set_sampler(self, name: str) -> None:
        if name == self._sampler:
            return
        import diffusers
        cls, kw = SAMPLERS[name]
        self.pipe.scheduler = getattr(diffusers, cls).from_config(self.pipe.scheduler.config, **kw)
        self._sampler = name

    # ---- LoRAs ----
    def _adapter(self, path: str, pinned: tuple = ()) -> str:
        if path in self.adapters:
            self.adapters.move_to_end(path)
            return self.adapters[path]
        mb = loraformat.inspect(path).size_mb
        # drop the least recently used adapters (never one this request needs) until the new one fits the budget
        for old in [p for p in self.adapters if p not in pinned]:
            if sum(self.adapter_mb.values()) + mb <= LORA_BUDGET_MB:
                break
            self.pipe.delete_adapters([self.adapters.pop(old)])
            self.adapter_mb.pop(old, None)
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
        self.adapter_mb[path] = mb
        return name

    def set_loras(self, loras: list[tuple[str, float]]) -> None:
        want = tuple((p, round(float(w), 4)) for p, w in loras if p and w != 0)
        # fused by default: T4, 3 LoRAs 34.7 -> 20.8 s, 1 LoRA 25.0 -> 19.5 s; SDXLR_FUSE=0 turns it off.
        # GGUF/bnb weights (Z-Image) can't be fused
        fuse = self.kind == "sdxl" and os.environ.get("SDXLR_FUSE", "1") == "1"
        if self._fused is not None and self._fused != want:
            self.pipe.unfuse_lora()  # back to the base weights before the set changes (or an adapter is evicted)
            self._fused = None
        if fuse and want and self._fused == want:
            return  # already merged into the weights: nothing to do per image
        pinned = tuple(p for p, _ in want)
        active = [(self._adapter(p, pinned), w) for p, w in want]
        if active:
            self.pipe.enable_lora()
            self.pipe.set_adapters([n for n, _ in active], adapter_weights=[w for _, w in active])
            if fuse:  # merged weights: no per-step LoRA math (fp16 drift after many switches; measured on T4)
                self.pipe.fuse_lora(adapter_names=[n for n, _ in active], lora_scale=1.0)
                self._fused = want
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

    def _tiling(self, width: int, height: int) -> None:
        """VAE tiling only for decodes over 1.5 MP; it is switched per call, so a small one never pays for it."""
        vae = getattr(self.pipe, "vae", None)
        if vae is not None:
            (vae.enable_tiling if needs_tiling(width, height) else vae.disable_tiling)()

    def _z_embeds(self, job: Job, count: int) -> dict:
        """Z-Image prompt embeddings, encoded once per (prompt, negative, text encoder, TE-touching LoRAs) and reused by
        generate, edits and every upscale tile. Returned as the pipeline's `prompt_embeds` arguments (one per image)."""
        key = z_embed_key(job.prompt, job.negative, job.cfg, self.model_path.split("|")[2:4], job.loras)
        if key not in self._embeds:
            cfg_on = job.cfg > 0  # diffusers 0.40 Z-Image: guidance (and its negative embeds) for any cfg > 0
            with torch.no_grad():
                pe, ne = self.pipe.encode_prompt(job.prompt, device="cuda", do_classifier_free_guidance=cfg_on,
                                                 negative_prompt=(job.negative or None) if cfg_on else None)
            self._embeds[key] = (pe[0], ne[0] if ne else None)
            while len(self._embeds) > 8:
                self._embeds.popitem(last=False)
        self._embeds.move_to_end(key)
        pe, ne = self._embeds[key]
        out = {"prompt_embeds": [pe] * count}
        if ne is not None:
            out["negative_prompt_embeds"] = [ne] * count
        return out

    # ---- ControlNet lite (GPU, untested; see engine/zcontrol.py) ----
    def _cn_pipe(self, name: str, zero_calls: tuple):
        """The control pipeline for "union" (better inpaint blend) or "tile" (upscale tiles). Loads the ControlNet on first
        use (a few seconds, ~2 GB VRAM), keeps one at a time, and refuses cleanly when the GPU has no room for it."""
        from . import zcontrol
        if self._cn is None or self._cn[0] != name:
            self._cn, self._cn_pipes = None, {}
            gc.collect()
            torch.cuda.empty_cache()
            path = MODELS / "zimage_cn" / CONTROLNETS[name][1]
            if not path.exists():
                raise UserError("The ControlNet file is not downloaded yet. Turn the option off and on again to fetch it.")
            free0 = torch.cuda.mem_get_info()[0]
            cn = zcontrol.load_controlnet(str(path)).to("cuda")
            free1 = torch.cuda.mem_get_info()[0]
            print(f"engine worker: ControlNet {name} loaded, VRAM free {free0 / 2**30:.1f} -> {free1 / 2**30:.1f} GB, "
                  f"{ram()}", file=sys.stderr, flush=True)
            if free1 < 0.8 * 2**30:  # no room left for activations: give it back instead of failing mid-image
                del cn
                gc.collect()
                torch.cuda.empty_cache()
                raise UserError("Not enough GPU memory for this option next to the loaded model. Use the plain version.")
            self._cn = (name, cn)
        key = (name, zero_calls)
        if key not in self._cn_pipes:
            self._cn_pipes[key] = zcontrol.make_pipeline(self.pipe, self._cn[1], zero_calls)
        torch.cuda.reset_peak_memory_stats()
        return self._cn_pipes[key]

    def _drop_cn(self) -> None:
        """The ControlNet (~2 GB VRAM) stays only while ops use it: next to Q8 it pushed the next upscale into OOM retries."""
        if self._cn is not None:
            self._cn, self._cn_pipes = None, {}
            gc.collect()
            torch.cuda.empty_cache()

    def _log_peak(self, what: str) -> None:
        print(f"engine worker: {what}: peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, {ram()}",
              file=sys.stderr, flush=True)

    def _inpaint_cn(self, job: Job, img: Image.Image, m: Image.Image) -> Image.Image:
        """Painted edit with the ControlNet lite inpaint inputs (8 steps, scale 0.9): a better blend than plain inpaint."""
        pipe = self._cn_pipe("union", (1,))  # the control latents are zeros: no pose or depth hint
        w, h = (img.width // 16) * 16, (img.height // 16) * 16
        if (w, h) != img.size:
            img = img.resize((w, h))
        self.set_loras(job.loras)
        gen = torch.Generator("cpu").manual_seed(job.seed if job.seed >= 0 else 0)
        with torch.no_grad():
            out = pipe(height=h, width=w, image=img, mask_image=m.resize((w, h)), control_image=img,
                       num_inference_steps=8, guidance_scale=job.cfg if job.cfg > 1 else 0.0,
                       controlnet_conditioning_scale=0.9, generator=gen,
                       **self._z_embeds(job, 1), **_step_cb(job)).images[0]
        self._log_peak("ControlNet inpaint")
        return out

    def _tile_cn(self, job: Job, tile: Image.Image, seed: int) -> Image.Image:
        """Refine one upscale tile with the Tile ControlNet lite (8 steps, scale 0.8); the whole tile is the control image."""
        pipe = self._cn_pipe("tile", (2,))  # all-white mask, zero masked-image latents
        w, h = (tile.width // 16) * 16, (tile.height // 16) * 16
        tile = tile.convert("RGB").resize((w, h))
        gen = torch.Generator("cpu").manual_seed(seed)
        with torch.no_grad():
            return pipe(height=h, width=w, image=tile, mask_image=Image.new("L", (w, h), 255), control_image=tile,
                        num_inference_steps=8, guidance_scale=job.cfg if job.cfg > 1 else 0.0,
                        controlnet_conditioning_scale=0.8, generator=gen, **self._z_embeds(job, 1)).images[0]

    def make_preview(self, latents, path, done) -> None:
        """Save a live preview of these latents to `path` (cheap, skipped while the last one is being written)."""
        from .preview import Previewer
        if self._prev is None:
            self._prev = Previewer()
        self._prev.make(self.kind, self.pipe, latents, path, done)

    def _run_zimage(self, job: Job, seed: int) -> list[Image.Image]:
        # Qwen text encoder: no A1111 weights / clip skip; negative only matters with cfg > 1
        self.set_loras(job.loras)
        gen = [torch.Generator("cpu").manual_seed(seed + i) for i in range(job.count)]
        self._tiling(*(job.image.size if job.image is not None else (job.width, job.height)))
        common = dict(num_inference_steps=job.steps, guidance_scale=job.cfg, generator=gen,
                      **self._z_embeds(job, job.count), **_step_cb(job))
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

    def _run(self, job: Job, seed: int, tim: dict | None = None) -> list[Image.Image]:
        if self.kind == "zimage":
            return self._run_zimage(job, seed)
        lora_key = tuple(sorted((p, round(w, 4)) for p, w in job.loras if p and w))
        self.set_loras(job.loras)
        self._set_sampler(job.sampler)
        self._tiling(*(job.image.size if job.image is not None else (job.width, job.height)))
        t0 = time.time()
        emb = self._encode(job, lora_key)
        if tim is not None:
            tim["enc"] = time.time() - t0  # Z-Image encodes inside the pipeline; this stays 0 there
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
        tim = {"enc": 0.0, "first": None, "last": None}
        orig_step = job.on_step

        def timed_step(done: int, total: int) -> None:
            now = time.time()
            if tim["first"] is None:
                tim["first"] = now
            tim["last"] = now
            if orig_step:
                orig_step(done, total)

        job.on_step = timed_step
        try:
            images = self._run(job, seed, tim)
        except torch.OutOfMemoryError:
            self._embeds.clear()
            gc.collect()
            torch.cuda.empty_cache()
            self.pipe.vae.enable_tiling()
            tim.update(enc=0.0, first=None, last=None)  # the timings of the failed try are meaningless
            images = self._run(job, seed, tim)  # a second OOM propagates to the caller
        total = time.time() - t
        parts = []
        if tim["enc"] > 0.05:
            parts.append(f"encode {tim['enc']:.1f}s")
        if tim["last"]:
            parts.append(f"denoise {tim['last'] - tim['first']:.1f}s")
            parts.append(f"decode {max(total - (tim['last'] - t), 0):.1f}s")  # past the last step: decode + copy
        stages = f" ({' · '.join(parts)})" if parts else ""
        info = f"{total:.1f}s{stages} | seed {seed} | {vram()} | {ram()}"
        torch.cuda.empty_cache()  # give back the activations' memory between images
        return images, seed, info

    def inpaint_crop(self, job: Job, crop: Image.Image, mask: Image.Image, blend: str = "plain") -> Image.Image:
        """Inpaint one prepared crop (crop-and-stitch): the crop is ~1 MP and multiple-aligned,
        the mask is already grown; blending back is the caller's (edit.paste_back) job."""
        if self.pipe is None:
            raise RuntimeError("no model loaded")
        img = crop.convert("RGB")
        m = mask.convert("L")
        gen = torch.Generator("cpu").manual_seed(job.seed if job.seed >= 0 else 0)
        if blend != "controlnet":
            self._drop_cn()
        if blend == "controlnet":
            if self.kind != "zimage":
                raise UserError("The better blend is for Z-Image only.")
            return self._inpaint_cn(job, img, m)
        if self.kind == "zimage":
            if self._inpaint is None:
                from diffusers import ZImageInpaintPipeline
                # GGUF/bnb weights can't be cast, which from_pipe tries; reuse the parts as they are
                self._inpaint = ZImageInpaintPipeline(**self.pipe.components)
            self.set_loras(job.loras)
            w, h = (img.width // 16) * 16, (img.height // 16) * 16
            if (w, h) != img.size:
                img = img.resize((w, h))
            kw = dict(num_inference_steps=job.steps, guidance_scale=job.cfg, generator=gen,
                      **self._z_embeds(job, 1), **_step_cb(job))
            return self._inpaint(image=img, mask_image=m.resize(img.size), width=w, height=h,
                                 strength=job.strength, **kw).images[0]
        if self._inpaint is None:
            from diffusers import StableDiffusionXLInpaintPipeline
            self._inpaint = StableDiffusionXLInpaintPipeline.from_pipe(self.pipe, torch_dtype=torch.float16)
        lora_key = tuple(sorted((p, round(w, 4)) for p, w in job.loras if p and w))
        self.set_loras(job.loras)
        self._set_sampler(job.sampler)
        emb = self._encode(job, lora_key)
        w, h = (img.width // 8) * 8, (img.height // 8) * 8
        if (w, h) != img.size:
            img = img.resize((w, h))
        return self._inpaint(image=img, mask_image=m.resize(img.size), width=w, height=h,
                             strength=job.strength, num_inference_steps=job.steps,
                             guidance_scale=job.cfg, generator=gen, **emb, **_step_cb(job)).images[0]

    def upscale(self, job: Job, scale: float = 2.0, tile: int = 1280, overlap: int = 128,
                on_tile=None, tile_cn: bool = False) -> tuple[Image.Image, int, str]:
        """Enlarge job.image by `scale`, then refine overlapping tiles with image-to-image (job.prompt,
        job.strength) and blend them with feathered edges. on_tile(done, total) reports progress."""
        import numpy as np
        if self.pipe is None:
            raise RuntimeError("no model loaded")
        if not tile_cn:
            self._drop_cn()
        t = time.time()
        seed = job.seed if job.seed >= 0 else int(torch.randint(0, 2**31 - 1, (1,)))
        m = 16 if self.kind == "zimage" else 8
        src = job.image.convert("RGB")
        from .edit import cap_size
        W, H = (int(src.width * scale) // m) * m, (int(src.height * scale) // m) * m
        cw, ch = cap_size(W, H)  # 8 MP output cap
        W, H = (cw // m) * m, (ch // m) * m
        big = src.resize((W, H), Image.LANCZOS)
        from .keys import tile_starts
        if tile_cn:
            tile = min(tile, 1024)  # the ControlNet's 2 GB next to Q8: 1.2 MP tiles ran out of VRAM on a T4
        xs, tw = tile_starts(W, tile, overlap, m)
        ys, th = tile_starts(H, tile, overlap, m)
        boxes = [(x, y, min(x + tw, W), min(y + th, H)) for y in ys for x in xs]
        acc = np.zeros((H, W, 3), np.float32)
        wsum = np.zeros((H, W, 1), np.float32)
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            sub = Job(**{**job.__dict__, "image": big.crop((x0, y0, x1, y1)), "mask": None, "count": 1,
                         "seed": seed, "on_step": None, "on_preview": None})
            try:
                piece = self._tile_cn(sub, sub.image, seed) if tile_cn and self.kind == "zimage" else self._run(sub, seed)[0]
            except torch.OutOfMemoryError:
                self._drop_cn()
                gc.collect()
                torch.cuda.empty_cache()
                raise UserError("Not enough GPU memory for this upscale"
                                   + (" with the tile detail model. Turn it off, or try 1.5x." if tile_cn else
                                      ". Try 1.5x.")) from None
            out = piece.resize((x1 - x0, y1 - y0), Image.LANCZOS)
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
