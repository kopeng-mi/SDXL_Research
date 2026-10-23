"""Get a dataset into one flat folder. Runs inside the main venv.

  python -m sdxlr.data --source test|hf|dir --dest DIR [--repo org/name --file x.zip] [--path DIR]
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".txt"}
TEST_REPO = "diffusers/dog-example"  # 5 public DreamBooth photos, for pipeline tests


def _flatten(src: Path, dest: Path) -> int:
    n = 0
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS and not p.name.startswith("."):
            target = dest / p.name
            if p.resolve() != target.resolve():
                shutil.copy2(p, target)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["test", "hf", "dir"], required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--repo", help="HF repo with a dataset zip")
    ap.add_argument("--repo-type", default="dataset")
    ap.add_argument("--file", help="zip filename inside the HF repo")
    ap.add_argument("--path", type=Path, help="local / Drive folder or zip")
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    tmp = a.dest.parent / f".{a.dest.name}_dl"

    from huggingface_hub import hf_hub_download, snapshot_download
    if a.source == "test":
        src = Path(snapshot_download(TEST_REPO, repo_type="dataset", local_dir=tmp))
    elif a.source == "hf":
        z = Path(hf_hub_download(a.repo, a.file, repo_type=a.repo_type, local_dir=tmp))
        src = tmp / "x"
        zipfile.ZipFile(z).extractall(src)
    else:
        src = a.path
        if src.suffix == ".zip":
            zipfile.ZipFile(src).extractall(tmp)
            src = tmp
    n = _flatten(src, a.dest)
    shutil.rmtree(tmp, ignore_errors=True)
    imgs = [p for p in a.dest.iterdir() if p.suffix.lower() != ".txt"]
    caps = list(a.dest.glob("*.txt"))
    print(f"{a.dest}: {len(imgs)} images, {len(caps)} captions ({n} files copied)")


if __name__ == "__main__":
    main()
