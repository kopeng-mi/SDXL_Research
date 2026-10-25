"""Caption review for the notebooks. Stdlib only (runs in the kernel).

`report` flags captions that are missing, over the word budget, or lack the
trigger word; `replace` does a find/replace pass over every caption (and its
Drive copy); `zip_dataset` packs images + captions for download.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from sdxlr import drive, plan

IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _images(ds: Path) -> list[Path]:
    return sorted(p for p in Path(ds).iterdir() if p.suffix.lower() in IMG)


def report(ds: str | Path, target: str, trigger: str = "") -> str:
    imgs = _images(Path(ds))
    words, missing, over, no_trig = [], [], [], []
    for p in imgs:
        t = p.with_suffix(".txt")
        if not t.exists():
            missing.append(p.name)
            continue
        text = t.read_text(encoding="utf-8")
        words.append(len(text.split()))
        if plan.over_budget(text, target):
            over.append(f"{t.name} ({words[-1]} words)")
        if trigger and not text.startswith(trigger):
            no_trig.append(t.name)
    lines = [f"{len(imgs)} images, {len(words)} captions"]
    if words:
        lines.append(f"words: min {min(words)}, avg {sum(words) / len(words):.0f}, max {max(words)} "
                     f"(budget {plan.WORD_BUDGET[target]} for {target})")
    for label, items in (("no caption", missing), ("over budget", over), (f"not starting with '{trigger}'", no_trig)):
        if items:
            lines.append(f"{label}: {', '.join(items)}")
    return "\n".join(lines)


def replace(ds: str | Path, find: str, repl: str, mirror: str | Path | None = None) -> int:
    """Replace text in every caption; returns how many files changed."""
    n = 0
    for t in sorted(Path(ds).glob("*.txt")):
        if t.name.startswith("_"):
            continue
        text = t.read_text(encoding="utf-8")
        if find in text:
            t.write_text(text.replace(find, repl), encoding="utf-8")
            n += 1
            if mirror:
                drive.mirror(t, mirror)
    return n


def zip_dataset(ds: str | Path, out: str | Path) -> Path:
    ds, out = Path(ds), Path(out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in _images(ds):
            z.write(p, p.name)
            if p.with_suffix(".txt").exists():
                z.write(p.with_suffix(".txt"), p.with_suffix(".txt").name)
    return out
