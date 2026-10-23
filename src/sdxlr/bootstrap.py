"""Create the shared uv venv on Colab. Stdlib only: runs on the system Python.

Usage (from the notebook):  python -m sdxlr.bootstrap  (with src/ on PYTHONPATH)

Idempotent: a stamp file records a hash of the install inputs, so a re-run in
the same session is a no-op unless pyproject.toml or the kohya pin changed.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KOHYA = REPO / "vendor" / "sd-scripts"
VENV = Path(os.environ.get("SDXLR_VENV", "/content/venv"))
VENV_PY = VENV / "bin" / "python"
PYTHON = "3.12"  # uv-managed, never the Colab system Python (it changes often)
STAMP = VENV / ".sdxlr-stamp"


def _uv() -> str:
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if not Path(uv).exists():
        subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True, check=True)
    assert Path(uv).exists(), f"uv not found at {uv}"
    return uv


def _inputs_hash() -> str:
    h = hashlib.sha256()
    for p in (REPO / "pyproject.toml", KOHYA / "requirements.txt"):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def ensure_venv() -> Path:
    want = _inputs_hash()
    if STAMP.exists() and STAMP.read_text().strip() == want:
        print(f"venv ok: {VENV} ({want})")
        return VENV_PY

    uv = _uv()
    if not VENV_PY.exists():
        subprocess.run([uv, "venv", str(VENV), "--python", PYTHON, "--managed-python", "--seed"], check=True)

    def pip(*args: str, cwd: Path = REPO) -> None:
        subprocess.run([uv, "pip", "install", "--python", str(VENV_PY), *args], cwd=cwd, check=True)

    # kohya first (its -e . must resolve inside its own dir), then ours, which
    # pins the shared versions on top.
    pip("-r", "requirements.txt", cwd=KOHYA)
    pip("-e", str(REPO))

    STAMP.write_text(want)
    print(f"venv ready: {VENV} ({want})")
    return VENV_PY


if __name__ == "__main__":
    ensure_venv()
    subprocess.run([str(VENV_PY), "-c",
                    "import torch, diffusers, transformers; "
                    "print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), "
                    "'| diffusers', diffusers.__version__, '| transformers', transformers.__version__)"],
                   check=True)
    sys.exit(0)
