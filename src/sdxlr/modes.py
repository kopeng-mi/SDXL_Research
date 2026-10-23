"""UI choices: speed modes, aspect ratios, styles, and training-time estimates.

Seconds-per-step numbers are measured on the Colab T4 unless marked "est".
"""

from __future__ import annotations

# mode -> settings applied on top of the form (per model kind)
MODES = {
    "sdxl": {
        "Quality": {"steps": 30, "cfg": 5.0, "sampler": "dpmpp_2m_karras"},
        "Speed": {"steps": 20, "cfg": 5.0, "sampler": "dpmpp_2m_karras"},
        # SDXL-Lightning 8-step LoRA, added automatically (est: ~3x faster than Speed)
        "Fast": {"steps": 8, "cfg": 1.5, "sampler": "euler_trailing",
                 "lora": ("ByteDance/SDXL-Lightning/sdxl_lightning_8step_lora.safetensors", 1.0)},
    },
    "zimage": {
        "Quality": {"steps": 12, "cfg": 0.0},
        "Speed": {"steps": 9, "cfg": 0.0},  # Z-Image Turbo default
        "Fast": {"steps": 6, "cfg": 0.0},
    },
}

ASPECTS = {
    "Portrait 896x1152": (896, 1152),
    "Tall 832x1216": (832, 1216),
    "Square 1024x1024": (1024, 1024),
    "Landscape 1152x896": (1152, 896),
    "Wide 1216x832": (1216, 832),
    "Small portrait 768x1024": (768, 1024),
}

# appended to the prompt; SDXL Pony models also get their score tags
STYLES = {
    "None": "",
    "Photo": "photo, natural skin texture, realistic",
    "Studio portrait": "studio portrait, softbox lighting, plain background, sharp focus",
    "Candid": "candid photo, natural light, everyday setting",
    "Cinematic": "cinematic still, dramatic lighting, shallow depth of field, film grain",
    "Fashion": "fashion editorial, full body, clean background, professional lighting",
}
PONY_TAGS = "score_9, score_8_up, score_7_up, rating_safe"
PONY_NEG = "nsfw, nude, lingerie, underwear, score_6, score_5, score_4"

# training seconds per optimizer step on T4
TRAIN_SPEED = {
    ("sdxl", 1024, 3): 5.1,
    ("zimage", 512, 1): 5.7,
    ("zimage", 768, 1): 13.0,  # est (~2.25x the pixels of 512)
}
TRAIN_OVERHEAD = {"sdxl": 120, "zimage": 300}  # caching + model load, seconds (first run adds downloads)


def train_eta(kind: str, images: int, repeats: int, epochs: int, batch: int, resolution: int) -> tuple[int, float, bool]:
    """-> (steps, seconds, is_estimate)"""
    steps = -(-images * repeats // max(batch, 1)) * epochs
    key = (kind, resolution, batch)
    if key in TRAIN_SPEED:
        per, est = TRAIN_SPEED[key], kind == "zimage" and resolution != 512
    else:  # scale the closest measured point by pixels x batch
        base = (kind, 1024, 3) if kind == "sdxl" else (kind, 512, 1)
        per = TRAIN_SPEED[base] * (resolution / base[1]) ** 2 * (batch / base[2])
        est = True
    return steps, steps * per + TRAIN_OVERHEAD[kind], est
