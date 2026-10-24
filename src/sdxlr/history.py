"""Image history: every generated PNG gets a JSON file with the same name.

The JSON holds everything needed to make the image again (prompt, model,
LoRAs + weights, mode, steps, size, seed, ...). It is also embedded in the
PNG ("sdxlr" text chunk), so either file can be loaded back into the form.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .paths import OUTPUTS


def save(png: Path, meta: dict) -> None:
    png.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def items(limit: int = 200) -> list[tuple[Path, dict]]:
    """Newest first: (png, meta)."""
    out = []
    for j in sorted(OUTPUTS.glob("*/*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        png = j.with_suffix(".png")
        if png.exists():
            try:
                out.append((png, json.loads(j.read_text(encoding="utf-8"))))
            except json.JSONDecodeError:
                pass
    return out


def read(data: bytes, name: str) -> dict:
    """Settings from an uploaded .json or a PNG made by this app."""
    if name.lower().endswith(".json"):
        return json.loads(data.decode("utf-8"))
    import io
    txt = Image.open(io.BytesIO(data)).text.get("sdxlr")
    if not txt:
        raise ValueError("this PNG has no settings (it was not made by this app)")
    return json.loads(txt)


def prompts(limit: int = 30) -> list[str]:
    """Recent unique prompts, newest first."""
    seen, out = set(), []
    for _, m in items(300):
        p = m.get("prompt_raw") or m.get("prompt", "")
        if p and p not in seen:
            seen.add(p)
            out.append(p)
        if len(out) >= limit:
            break
    return out
