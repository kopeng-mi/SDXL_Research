"""Build the notebooks from cells/ (sources of truth).

notebooks/<nb>.ipynb = the shared cells/_setup.py (title, intro, venv groups and
Drive default filled in from NOTEBOOKS) + cells/<nb>/NN_name.py in filename order.
A file may start with a markdown block fenced by `# %% [md]` ... `# %%`; that block
becomes a markdown cell placed before the code cell.

  python tools/build_notebook.py          write the notebooks
  python tools/build_notebook.py --check  exit 1 if any notebook is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "cells"
OUT_DIR = ROOT / "notebooks"

# name -> title, intro line, venv groups (None = no shared venv), Drive on by default
NOTEBOOKS = {
    "captioning": ("Captioning", "Caption a dataset on Drive with Qwen3-VL (abliterated options), JoyCaption or WD tags.",
                   "caption", True),
    "train-test-sdxl": ("Train + test SDXL LoRA", "Train an SDXL LoRA from a Drive dataset, then compare epochs.",
                        "caption,train-sdxl", True),
    "train-test-zimage": ("Train + test Z-Image LoRA", "Train a Z-Image LoRA from a Drive dataset, then compare epochs.",
                          "caption,train-z", True),
    "sdxl": ("SDXL (Fooocus)", "Fooocus 2.5.5 with your checkpoints and LoRAs, at http://sdxl:7865 on Tailscale.",
             None, False),
    "zimage": ("Z-Image", "Generate, edit and upscale with Z-Image, at http://zimage:7860 on Tailscale.", "app", False),
}


def _lines(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def _split(src: str) -> tuple[str | None, str]:
    if not src.startswith("# %% [md]\n"):
        return None, src
    head, _, code = src[len("# %% [md]\n"):].partition("\n# %%\n")
    md = "\n".join(line.removeprefix("# ").removeprefix("#") for line in head.split("\n"))
    return md, code


def _cells(sources: list[tuple[str, str]]) -> list[dict]:
    cells = []
    for cell_id, src in sources:
        md, code = _split(src)
        if md:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(md)})
        cells.append({
            "cell_type": "code", "execution_count": None,
            "metadata": {"id": cell_id}, "outputs": [], "source": _lines(code),
        })
    return cells


def _nb(cells: list[dict]) -> str:
    nb = {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def build() -> dict[Path, str]:
    out = {}
    setup = (CELLS / "_setup.py").read_text(encoding="utf-8")
    for name, (title, intro, groups, drive) in NOTEBOOKS.items():
        head = (setup.replace("__TITLE__", title).replace("__INTRO__", intro)
                .replace("__GROUPS__", repr(groups)).replace("__DRIVE__", str(drive)))
        srcs = [("setup", head)] + [(p.stem, p.read_text(encoding="utf-8"))
                                    for p in sorted((CELLS / name).glob("*.py"))]
        out[OUT_DIR / f"{name}.ipynb"] = _nb(_cells(srcs))
    return out


if __name__ == "__main__":
    built = build()
    if "--check" in sys.argv:
        stale = [p.name for p, t in built.items() if not p.exists() or p.read_text(encoding="utf-8") != t]
        print(f"stale: {', '.join(stale)} (run tools/build_notebook.py)" if stale else "notebooks up to date")
        sys.exit(int(bool(stale)))
    OUT_DIR.mkdir(exist_ok=True)
    for p, text in built.items():
        p.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {p.relative_to(ROOT).as_posix()} ({text.count('\"cell_type\": \"code\"')} code cells)")
