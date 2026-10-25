"""Fooocus 2.5.5 on Colab, in its own Python 3.11 venv. Stdlib only (runs in the Colab kernel).

Fooocus is unmaintained since 2.5.5 and breaks on Colab's Python 3.12 / new numpy,
so it gets a uv-managed 3.11 venv with the pins that are known to work, and torch
2.1.0 cu121 installed up front (launch.py then skips its own install).
config.txt points Fooocus at the local model folders plus the Drive ones
(Fooocus accepts lists), and turns off its default model and LoRA downloads.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

TAG = "v2.5.5"
URL = "https://github.com/lllyasviel/Fooocus.git"
DIR = Path("/content/Fooocus")
VENV = Path("/content/fooocus-venv")
PY = VENV / "bin" / "python"
PORT = 7865
STAMP = VENV / ".sdxlr-stamp"
PINS = ["pygit2==1.15.1", "numpy==1.26.4"]
WEB_PINS = ["jinja2==3.0.3", "fastapi==0.95.2", "starlette==0.27.0"]  # gradio 3.41 breaks on newer ones
TORCH = ["torch==2.1.0", "torchvision==0.16.0", "--extra-index-url", "https://download.pytorch.org/whl/cu121"]
CKPTS, LORAS, OUTPUTS = DIR / "models" / "checkpoints", DIR / "models" / "loras", DIR / "outputs"


def _uv() -> str:
    from .bootstrap import _uv as uv
    return uv()


def setup() -> None:
    """Clone Fooocus at TAG and build its venv (skipped when the stamp matches)."""
    want = f"{TAG} {' '.join(PINS + WEB_PINS + TORCH[:2])}"
    if STAMP.exists() and STAMP.read_text() == want and DIR.exists():
        print(f"fooocus ok: {DIR} ({TAG})")
        return
    if not DIR.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", TAG, URL, str(DIR)], check=True)
    uv = _uv()
    if not PY.exists():
        subprocess.run([uv, "venv", str(VENV), "--python", "3.11", "--managed-python", "--seed"], check=True)

    def pip(*args: str) -> None:
        subprocess.run([uv, "pip", "install", "-q", "--python", str(PY), *args], cwd=DIR, check=True)

    pip(*TORCH)
    pip(*PINS)
    pip("-r", "requirements_versions.txt")
    pip("--reinstall", *WEB_PINS)
    subprocess.run([uv, "pip", "uninstall", "-q", "--python", str(PY), "cupy-cuda12x"], check=False)
    STAMP.write_text(want)
    print(f"fooocus ready: {DIR} ({TAG})")


def write_config(model: str, ckpt_dirs: list[Path], lora_dirs: list[Path], outputs: Path) -> Path:
    """config.txt: our folders, `model` as the default, no downloads, no default LoRA."""
    cfg = {
        "path_checkpoints": [str(p) for p in ckpt_dirs],
        "path_loras": [str(p) for p in lora_dirs],
        "path_outputs": str(outputs),
        "default_model": model,
        "default_refiner": "None",
        "default_loras": [[True, "None", 1.0] for _ in range(5)],
        "checkpoint_downloads": {},
        "lora_downloads": {},
        "embeddings_downloads": {},
        "previous_default_models": [],
    }
    for p in [*ckpt_dirs, *lora_dirs, outputs]:
        Path(p).mkdir(parents=True, exist_ok=True)
    f = DIR / "config.txt"
    f.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return f


def checkpoints(dirs: list[Path]) -> list[str]:
    return sorted({p.name for d in dirs if Path(d).exists() for p in Path(d).rglob("*.safetensors")})


def launch_args(port: int = PORT, share: bool = False, extra: str = "") -> list[str]:
    args = ["launch.py", "--listen", "--port", str(port), "--always-high-vram", "--disable-analytics",
            "--disable-preset-download"]
    return args + (["--share"] if share else []) + extra.split()


def wipe_outputs(path: Path = OUTPUTS) -> int:
    """Delete local outputs (not the Drive copy); returns how many files."""
    n = sum(1 for p in Path(path).rglob("*") if p.is_file()) if Path(path).exists() else 0
    shutil.rmtree(path, ignore_errors=True)
    return n
