"""Names and listings under WORK: models, LoRAs, datasets.

Today these scan the disk on each call (fine for user actions; the pages call
them when building widgets). Timers must go through the cached versions
(added next) so the event loop never touches the disk.
"""

from __future__ import annotations

from pathlib import Path

from ..fetch import list_loras, list_models, list_zimage
from ..paths import DATASETS, Z_MODELS

IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def lora_names() -> list[str]:
    return [p.name for p in list_loras()]


def lora_path(name: str) -> str | None:
    return next((str(p) for p in list_loras() if p.name == name), None)


def model_names() -> list[str]:
    return [p.name for p in list_models()]


def model_path(name: str) -> str:
    return next((str(p) for p in list_models() if p.name == name), name)


def z_names() -> list[str]:
    return [*Z_MODELS, *(p.name for p in list_zimage())]


def z_path(name: str) -> str:
    return Z_MODELS.get(name) or next((str(p) for p in list_zimage() if p.name == name), name)


def datasets() -> list[str]:
    return sorted(p.name for p in DATASETS.iterdir() if p.is_dir()) if DATASETS.exists() else []


def images(ds: str) -> list[Path]:
    d = DATASETS / ds if ds else None
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG) if d and d.exists() else []
