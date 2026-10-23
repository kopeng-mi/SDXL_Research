"""Build notebook.ipynb from cells/*.py (sources of truth).

Each cells/NN_name.py becomes one code cell, in filename order. A file may
start with a markdown block fenced by `# %% [md]` ... `# %%`; that block
becomes a markdown cell placed before the code cell.

  python tools/build_notebook.py          write notebook.ipynb
  python tools/build_notebook.py --check  exit 1 if notebook.ipynb is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "cells"
OUT = ROOT / "notebook.ipynb"


def _lines(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def _split(src: str) -> tuple[str | None, str]:
    if not src.startswith("# %% [md]\n"):
        return None, src
    head, _, code = src[len("# %% [md]\n"):].partition("\n# %%\n")
    md = "\n".join(line.removeprefix("# ").removeprefix("#") for line in head.split("\n"))
    return md, code


def build() -> str:
    cells = []
    for path in sorted(CELLS.glob("*.py")):
        md, code = _split(path.read_text(encoding="utf-8"))
        if md:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(md)})
        cells.append({
            "cell_type": "code", "execution_count": None,
            "metadata": {"id": path.stem}, "outputs": [], "source": _lines(code),
        })
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


if __name__ == "__main__":
    text = build()
    if "--check" in sys.argv:
        stale = not OUT.exists() or OUT.read_text(encoding="utf-8") != text
        print("notebook.ipynb is stale, run tools/build_notebook.py" if stale else "notebook.ipynb up to date")
        sys.exit(int(stale))
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.name} ({text.count('\"cell_type\": \"code\"')} code cells)")
