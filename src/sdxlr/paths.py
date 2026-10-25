"""Work dirs and the model catalog.

Imported by everything (fetch, UI, jobs, the engine worker): stdlib only,
never torch, so the web process and the download prefetch subprocesses stay
small. Local runs (no /content) work under ./work; SDXLR_WORK overrides.
"""

from __future__ import annotations

import os
from pathlib import Path


def _work_root() -> Path:
    env = os.environ.get("SDXLR_WORK")
    if env:
        return Path(env)
    if Path("/content").exists():  # Colab
        return Path("/content/work")
    return Path.cwd() / "work"


WORK = _work_root()
MODELS = WORK / "models"
LORAS = WORK / "loras"
ZIMAGE = MODELS / "zimage_custom"  # user Z-Image checkpoints (built-in ones go to models/zimage)
DATASETS = WORK / "datasets"
OUTPUTS = WORK / "outputs"
UPLOADS = WORK / "uploads"
THUMBS = WORK / "thumbs"
RUNS = WORK / "runs"
LOGS = WORK / "logs"
PREVIEWS = WORK / "previews"  # live step previews (JPEG per request), served to the page
COMPARE = WORK / "compare"
RUN = WORK / "run"  # pidfiles: the engine worker and job processes
SESS = WORK / "sessions"
WEIGHTS = {".safetensors", ".gguf", ".ckpt"}

FP16_VAE = "madebyollin/sdxl-vae-fp16-fix"
SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"
# training bases for the SDXL notebook (v-pred is read from the file header, see loraformat.is_vpred)
SDXL_BASES = {
    "sdxl-base": SDXL_BASE,
    "pony-v6": "LyliaEngine/Pony_Diffusion_V6_XL/ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
    "illustrious-0.1": "OnomaAIResearch/Illustrious-xl-early-release-v0/Illustrious-XL-v0.1.safetensors",
    "noobai-eps-1.1": "Laxhar/noobai-XL-1.1/NoobAI-XL-v1.1.safetensors",
    "noobai-vpred-1.0": "Laxhar/noobai-XL-Vpred-1.0/NoobAI-XL-Vpred-v1.0.safetensors",
}

# Z-Image choices (any org/repo/file, URL or path also works)
Z_REPO = "Tongyi-MAI/Z-Image-Turbo"  # tokenizer, VAE, scheduler, configs
Z_MODELS = {
    # default: bnb nf4 DiT, 3.3 GB. T4: 35.9 s Speed (Q8 35.0), the same image with and without a LoRA, 6 GB VRAM
    "turbo 4-bit (unsloth)": "unsloth/Z-Image-Turbo-unsloth-bnb-4bit:transformer",
    "turbo Q8": "unsloth/Z-Image-Turbo-GGUF/z-image-turbo-Q8_0.gguf",  # 7.2 GB, 10 GB VRAM
    "turbo Q6": "unsloth/Z-Image-Turbo-GGUF/z-image-turbo-Q6_K.gguf",  # 5.9 GB; Q6/Q4 are slower than Q8 on a T4
    "turbo Q5": "unsloth/Z-Image-Turbo-GGUF/z-image-turbo-Q5_K_M.gguf",  # 5.6 GB
    "turbo Q4": "unsloth/Z-Image-Turbo-GGUF/z-image-turbo-Q4_K_M.gguf",  # 5.0 GB
}
Z_SIZES_GB = {"turbo Q8": 7.2, "turbo Q6": 5.9, "turbo Q5": 5.6, "turbo Q4": 5.0, "turbo 4-bit (unsloth)": 3.3}  # download sizes of the built-in models
Z_ENCODERS = {
    "official": "Tongyi-MAI/Z-Image-Turbo:text_encoder",
    "abliterated": "BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1",  # fewer refusals in prompt following
    # unsloth's pre-quantized encoder was tried (T4, 2026-10-24): its packed 4-bit weights do not load through AutoModel
    # (size mismatch on every Linear), so it is not offered; the official encoder quantized at load stays
}
TE_GB = {"nf4": 2.5, "int8": 4.4, "fp16": 7.9, "prequant": 3.4}  # download size of the text encoder variants
# ControlNet lite files (8 steps, ~1.9 GB each): "union" gives the better inpaint blend, "tile" refines upscale tiles
# captioners: key -> (repo, params in billions (fp16 vs nf4 is picked from this), loader)
CAPTIONERS = {
    "qwen3vl-2b": ("Qwen/Qwen3-VL-2B-Instruct", 2.1, "qwen"),
    "qwen3vl-4b": ("Qwen/Qwen3-VL-4B-Instruct", 4.4, "qwen"),
    "qwen3vl-4b-abl": ("huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated", 4.4, "qwen"),
    "qwen3vl-8b-abl": ("huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated", 8.8, "qwen"),
    "qwen35-4b-abl (exp)": ("huihui-ai/Huihui-Qwen3.5-4B-abliterated", 4.5, "qwen"),
    "qwen35-9b-abl (exp)": ("huihui-ai/Huihui-Qwen3.5-9B-abliterated", 9.6, "qwen"),
    "joy": ("fancyfeast/llama-joycaption-beta-one-hf-llava", 8.5, "joy"),  # uncensored by design
    "wd": ("SmilingWolf/wd-eva02-large-tagger-v3", 0.3, "wd"),
}
CAPTIONER_ALIASES = {"qwen": "qwen3vl-4b", "qwen-2b": "qwen3vl-2b"}

CONTROLNETS = {
    "union": ("alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1", "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps.safetensors"),
    "tile": ("alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1", "Z-Image-Turbo-Fun-Controlnet-Tile-2.1-lite-2601-8steps.safetensors"),
}
PREVIEW_VAES = {"zimage": "madebyollin/taef1", "sdxl": "madebyollin/taesdxl"}  # tiny VAEs for live previews


def te_prequantized(text_encoder: str) -> bool:
    """True for a text encoder that ships its own quantization (the unsloth bnb-4bit one)."""
    return text_encoder.startswith("unsloth/") and "bnb" in text_encoder

# name -> (diffusers scheduler class, kwargs); diffusers itself is imported only where it runs
SAMPLERS = {
    "dpmpp_2m_karras": ("DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True}),
    "dpmpp_2m_sde_karras": ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True}),
    "dpmpp_sde_karras": ("DPMSolverSinglestepScheduler", {"use_karras_sigmas": True}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_trailing": ("EulerDiscreteScheduler", {"timestep_spacing": "trailing"}),  # Lightning / few-step
    "lcm": ("LCMScheduler", {}),
}


def z_key(transformer: str, text_encoder: str, te_quant: str) -> str:
    """The loaded-model identity of a Z-Image setup (compared against the UI's pick)."""
    return f"zimage|{transformer}|{text_encoder}|{te_quant}"
