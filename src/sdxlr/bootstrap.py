"""Create the shared uv venv on Colab. Stdlib only: runs on the system Python.

Usage (from the notebook):  python -m sdxlr.bootstrap --groups caption,train-sdxl  (src/ on PYTHONPATH)

Groups only add packages on top of core (our pinned deps), so one VM can run
several notebooks with one venv:  caption (core only), train-sdxl (kohya),
train-z (musubi), app (nicegui). Idempotent: the stamp records a hash of the
install inputs and the groups installed; a re-run is a no-op unless an input
changed or a new group is asked for.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KOHYA = REPO / "vendor" / "sd-scripts"
MUSUBI = REPO / "vendor" / "musubi-tuner"
VENV = Path(os.environ.get("SDXLR_VENV", "/content/venv"))
VENV_PY = VENV / "bin" / "python"
PYTHON = "3.12"  # uv-managed, never the Colab system Python (it changes often)
STAMP = VENV / ".sdxlr-stamp"
GROUPS = ("caption", "train-sdxl", "train-z", "app")


def _uv() -> str:
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if not Path(uv).exists():
        subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True, check=True)
    assert Path(uv).exists(), f"uv not found at {uv}"
    return uv


def _inputs_hash() -> str:
    h = hashlib.sha256()
    for p in (REPO / "pyproject.toml", KOHYA / "requirements.txt", MUSUBI / "pyproject.toml"):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _stamp() -> tuple[str, set[str]]:
    try:
        h, _, gs = STAMP.read_text().strip().partition(" ")
        return h, set(filter(None, gs.split(",")))
    except OSError:
        return "", set()


def ensure_venv(groups: set[str] = frozenset()) -> Path:
    want = _inputs_hash()
    have_hash, have = _stamp()
    if have_hash == want and groups <= have:
        print(f"venv ok: {VENV} ({want}, groups: {','.join(sorted(have)) or 'core'})")
        return VENV_PY
    groups = set(groups) | (have if have_hash == want else set())

    uv = _uv()
    if not VENV_PY.exists():
        subprocess.run([uv, "venv", str(VENV), "--python", PYTHON, "--managed-python", "--seed"], check=True)

    def pip(*args: str, cwd: Path = REPO) -> None:
        subprocess.run([uv, "pip", "install", "--python", str(VENV_PY), *args], cwd=cwd, check=True)

    # kohya first (its -e . must resolve inside its own dir), then ours, which
    # pins the shared versions on top.
    if "train-sdxl" in groups:
        pip("-r", "requirements.txt", cwd=KOHYA)
    if "train-z" in groups:
        pip("-e", str(MUSUBI))  # same diffusers/transformers/accelerate pins as kohya
    pip("-e", f"{REPO}[app]" if "app" in groups else str(REPO))

    STAMP.write_text(f"{want} {','.join(sorted(groups))}")
    print(f"venv ready: {VENV} ({want}, groups: {','.join(sorted(groups)) or 'core'})")
    return VENV_PY


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="", help=f"comma list of {', '.join(GROUPS)}")
    gs = {g.strip() for g in ap.parse_args().groups.split(",") if g.strip()}
    if bad := gs - set(GROUPS):
        sys.exit(f"unknown groups: {', '.join(sorted(bad))} (choose from {', '.join(GROUPS)})")
    ensure_venv(gs)
    subprocess.run([str(VENV_PY), "-c",
                    "import torch, diffusers, transformers; "
                    "print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), "
                    "'| diffusers', diffusers.__version__, '| transformers', transformers.__version__)"],
                   check=True)
    sys.exit(0)
