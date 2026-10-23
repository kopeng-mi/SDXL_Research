"""Import a dataset into one flat folder. Runs inside the main venv.

  python -m sdxlr.data --dest DIR SRC [SRC ...]

SRC can be: 'test' (5 public photos), a .zip / folder / image path (uploads,
Drive), an https URL, or an HF file 'org/repo/file.zip' (dataset or model repo).
Zips may contain captions (.txt next to each image) or not; nested folders are
flattened. Images are never resized here (the trainers bucket them).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
TEST_REPO = "diffusers/dog-example"  # 5 public DreamBooth photos, for pipeline tests


def _hf_file(spec: str, dst: Path) -> Path:
    from huggingface_hub import hf_hub_download
    org, repo, file = spec.split("/", 2)
    for kind in ("dataset", "model"):
        try:
            return Path(hf_hub_download(f"{org}/{repo}", file, repo_type=kind, local_dir=dst))
        except Exception as e:  # wrong repo type -> try the other
            err = e
    raise err


def _materialize(src: str, tmp: Path) -> Path:
    """Return a local file or folder for SRC."""
    if src == "test":
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(TEST_REPO, repo_type="dataset", local_dir=tmp / "test"))
    if Path(src).exists():
        return Path(src)
    if src.startswith("http"):
        from .fetch import fetch
        return fetch(src, tmp)
    return _hf_file(src, tmp)


def _copy_tree(src: Path, dest: Path) -> tuple[int, int]:
    n_img = n_txt = 0
    files = [src] if src.is_file() else [p for p in src.rglob("*") if p.is_file()]
    for p in files:
        if p.name.startswith(".") or "__MACOSX" in p.parts:
            continue
        ext = p.suffix.lower()
        if ext in IMG or ext == ".txt":
            shutil.copy2(p, dest / p.name)
            n_img += ext in IMG
            n_txt += ext == ".txt"
    return n_img, n_txt


def import_sources(sources: list[str], dest: Path) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    total_i = total_t = 0
    with tempfile.TemporaryDirectory(dir=dest.parent) as t:
        tmp = Path(t)
        for i, s in enumerate(sources):
            p = _materialize(s.strip(), tmp)
            if p.suffix.lower() == ".zip":
                x = tmp / f"zip{i}"
                with zipfile.ZipFile(p) as z:
                    z.extractall(x)
                p = x
            n_i, n_t = _copy_tree(p, dest)
            total_i, total_t = total_i + n_i, total_t + n_t
            print(f"[{i + 1}/{len(sources)}] {s}: {n_i} images, {n_t} captions", flush=True)
    imgs = [p for p in dest.iterdir() if p.suffix.lower() in IMG]
    caps = sum((p.with_suffix(".txt")).exists() for p in imgs)
    return f"{dest.name}: {len(imgs)} images, {caps} with captions, {len(imgs) - caps} without"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("sources", nargs="+")
    a = ap.parse_args()
    print(import_sources(a.sources, a.dest), flush=True)


if __name__ == "__main__":
    main()
