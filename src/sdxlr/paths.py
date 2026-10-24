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
COMPARE = WORK / "compare"
RUN = WORK / "run"  # pidfiles: the engine worker and job processes
SESS = WORK / "sessions"
WEIGHTS = {".safetensors", ".gguf", ".ckpt"}

FP16_VAE = "madebyollin/sdxl-vae-fp16-fix"
SDXL_BASE = "stabilityai/stable-diffusion-xl-base-1.0/sd_xl_base_1.0.safetensors"

# Z-Image choices (any org/repo/file, URL or path also works)
Z_REPO = "Tongyi-MAI/Z-Image-Turbo"  # tokenizer, VAE, scheduler, configs
Z_MODELS = {
    "turbo Q8": "unsloth/Z-Image-Turbo-GGUF/z-image-turbo-Q8_0.gguf",  # 7.2 GB, best quality
    "turbo Q6": "unsloth/Z-Image-Turbo-GGUF/z-image-turbo-Q6_K.gguf",  # 5.9 GB
    "turbo Q5": "unsloth/Z-Image-Turbo-GGUF/z-image-turbo-Q5_K_M.gguf",  # 5.6 GB
    "turbo Q4": "unsloth/Z-Image-Turbo-GGUF/z-image-turbo-Q4_K_M.gguf",  # 5.0 GB
}
Z_SIZES_GB = {"turbo Q8": 7.2, "turbo Q6": 5.9, "turbo Q5": 5.6, "turbo Q4": 5.0}  # file sizes of the built-in GGUFs
Z_ENCODERS = {
    "official": "Tongyi-MAI/Z-Image-Turbo:text_encoder",
    "abliterated": "BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1",  # fewer refusals in prompt following
}
TE_GB = {"nf4": 2.5, "int8": 4.4, "fp16": 7.9}  # download size of the text encoder variants

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
