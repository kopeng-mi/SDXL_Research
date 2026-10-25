"""Import a dataset into one flat folder. Runs inside the main venv.

  python -m sdxlr.data --dest DIR SRC [SRC ...]

SRC can be: 'test' (5 public photos), a .zip / folder / image path (uploads,
Drive), an https URL, or an HF file 'org/repo/file.zip' (dataset or model repo).
Zips may contain captions (.txt next to each image) or not; nested folders are
flattened. Images are never resized here (the trainers bucket them); phone
photos with an EXIF rotation are saved upright, since the trainers ignore EXIF.
--res/--kind print the checks: images too small for that size, and the bucket spread.
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


def _digest(p: Path) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _free_name(dest: Path, name: str) -> str:
    """name, or name_2, name_3 ... so two sources with the same file name never overwrite each other."""
    stem, ext = Path(name).stem, Path(name).suffix
    k, out = 1, name
    while any((dest / Path(out).with_suffix(e)).exists() for e in (ext, ".txt")):
        k += 1
        out = f"{stem}_{k}{ext}"
    return out


def _copy_tree(src: Path, dest: Path, seen: dict[str, str]) -> tuple[int, int, int]:
    """Copy images (+ their .txt captions) into dest: (images, captions, duplicates skipped).
    seen maps a content hash to the dataset file it is in (filled with what dest already has)."""
    n_img = n_txt = n_dup = 0
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file())
    for p in files:
        heic = p.suffix.lower() in (".heic", ".heif")
        if p.name.startswith(".") or "__MACOSX" in p.parts or not (p.suffix.lower() in IMG or heic):
            continue
        h = _digest(p)
        if h in seen:  # the same photo twice teaches the LoRA that one photo twice as hard
            n_dup += 1
            print(f"skipped {p.name}: same photo as {seen[h]}", flush=True)
            continue
        if heic:  # the trainers read JPEG/PNG only: convert iPhone photos on the way in
            from .engine.edit import HEIC, load_source
            if not HEIC:
                print(f"skipped {p.name}: HEIC needs pillow-heif (export it as JPEG)", flush=True)
                continue
            name = _free_name(dest, p.with_suffix(".jpg").name)
            load_source(p).save(dest / name, quality=95)
        elif _rotated(p):
            from PIL import Image, ImageOps
            name = _free_name(dest, p.name)
            with Image.open(p) as im:
                ImageOps.exif_transpose(im).save(dest / name, quality=95)
        else:
            name = _free_name(dest, p.name)
            shutil.copy2(p, dest / name)
        seen[h] = name
        n_img += 1
        cap = p.with_suffix(".txt")
        if cap.exists():
            shutil.copy2(cap, (dest / name).with_suffix(".txt"))
            n_txt += 1
    return n_img, n_txt, n_dup


def _rotated(p: Path) -> bool:
    """True when EXIF says the photo must be turned (orientation 2..8)."""
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.getexif().get(0x0112, 1) not in (0, 1)
    except Exception:
        return False


def check(dest: Path, kind: str, res: int) -> str:
    """Plain report: count, images too small for `res`, and how they spread over buckets."""
    from PIL import Image

    from . import plan
    sizes, small = [], []
    for p in sorted(dest.iterdir()):
        if p.suffix.lower() in IMG:
            with Image.open(p) as im:
                sizes.append(im.size)
            if plan.too_small(*im.size, res):
                small.append(f"{p.name} {im.size[0]}x{im.size[1]}")
    lines = [f"{len(sizes)} images checked for {kind} at {res}px"]
    if len(sizes) < plan.MIN_PHOTOS:
        lines.append(f"warning: only {len(sizes)} photos (5+ recommended)")
    if small:
        lines.append(f"{len(small)} smaller than {res // 2}px on the short side (they come out soft): " + ", ".join(small))
    b = plan.buckets(sizes, kind, res)
    lines.append("buckets: " + ", ".join(f"{w}x{h}: {n}" for (w, h), n in b.items()))
    if any(n == 1 for n in b.values()) and len(b) > 1:
        lines.append("note: a bucket with 1 image trains that shape with batch 1 (fine, just slower)")
    return "\n".join(lines)


def import_sources(sources: list[str], dest: Path) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    total_i = total_t = total_d = 0
    seen = {_digest(p): p.name for p in dest.iterdir() if p.suffix.lower() in IMG}  # photos already in the dataset
    with tempfile.TemporaryDirectory(dir=dest.parent) as t:
        tmp = Path(t)
        for i, s in enumerate(sources):
            p = _materialize(s.strip(), tmp)
            if p.suffix.lower() == ".zip":
                x = tmp / f"zip{i}"
                with zipfile.ZipFile(p) as z:
                    z.extractall(x)
                p = x
            n_i, n_t, n_d = _copy_tree(p, dest, seen)
            total_i, total_t, total_d = total_i + n_i, total_t + n_t, total_d + n_d
            dup = f", {n_d} duplicates skipped" if n_d else ""
            print(f"[{i + 1}/{len(sources)}] {s}: {n_i} images, {n_t} captions{dup}", flush=True)
    imgs = [p for p in dest.iterdir() if p.suffix.lower() in IMG]
    caps = sum((p.with_suffix(".txt")).exists() for p in imgs)
    dup = f"; {total_d} duplicate photos skipped" if total_d else ""
    return f"{dest.name}: {len(imgs)} images, {caps} with captions, {len(imgs) - caps} without{dup}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("sources", nargs="*")
    ap.add_argument("--res", type=int, default=0, help="print the checks for this training size")
    ap.add_argument("--kind", default="sdxl", choices=["sdxl", "zimage"])
    a = ap.parse_args()
    if a.sources:
        print(import_sources(a.sources, a.dest), flush=True)
    if a.res:
        print(check(a.dest, a.kind, a.res), flush=True)


if __name__ == "__main__":
    main()
