"""Download base models and LoRAs into /content/work. Runs inside the main venv.

  python -m sdxlr.fetch model SRC          SRC = org/repo/file, https URL or path
  python -m sdxlr.fetch loras SRC [SRC..]  .safetensors or .zip (flattened)

Downloads stream to disk (never held in RAM); zips are deleted after extraction.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

WORK = Path("/content/work")
MODELS = WORK / "models"
LORAS = WORK / "loras"


def fetch(src: str, dst_dir: Path) -> Path:
    """'org/repo/file', an https URL, or a local path -> local file."""
    if Path(src).exists():
        return Path(src)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.startswith("http"):
        if "civitai.com" in src:
            # file name comes from Content-Disposition; token from Colab Secrets / env
            import os
            hdr = ["-H", f"Authorization: Bearer {os.environ['CIVITAI_TOKEN']}"] if os.environ.get("CIVITAI_TOKEN") else []
            r = subprocess.run(["curl", "-fLJ", "--retry", "3", *hdr, "--output-dir", str(dst_dir), "-O",
                                "-w", "%{filename_effective}", src], check=True, capture_output=True, text=True)
            return Path(r.stdout.strip())
        dst = dst_dir / src.split("?")[0].rsplit("/", 1)[-1]
        if not dst.exists():
            subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(dst), src], check=True)
        return dst
    from huggingface_hub import hf_hub_download
    org, repo, file = src.split("/", 2)
    return Path(hf_hub_download(f"{org}/{repo}", file, local_dir=dst_dir))


def fetch_loras(src: str) -> list[Path]:
    p = fetch(src, LORAS)
    if p.suffix != ".zip":
        return [p]
    out = []
    with zipfile.ZipFile(p) as z:
        for m in z.namelist():
            if m.endswith(".safetensors"):
                t = LORAS / Path(m).name
                if not t.exists():
                    with z.open(m) as s, open(t, "wb") as d:
                        while b := s.read(1 << 20):
                            d.write(b)
                out.append(t)
    if p.parent == LORAS:
        p.unlink()
    return out


def list_models() -> list[Path]:
    """SDXL checkpoints only (Z-Image parts live in models/zimage*/)."""
    return sorted(p for p in MODELS.rglob("*.safetensors")
                  if not any(part.startswith("zimage") for part in p.relative_to(MODELS).parts))


def list_loras() -> list[Path]:
    """Downloaded LoRAs + everything trained in this session."""
    return sorted({*LORAS.glob("*.safetensors"), *(WORK / "runs").glob("*/*.safetensors")}, key=lambda p: p.name)


if __name__ == "__main__":
    kind, *srcs = sys.argv[1:]
    for s in srcs:
        if kind == "model":
            p = fetch(s, MODELS)
            print(f"model: {p} ({p.stat().st_size / 1e9:.2f} GB)")
        else:
            for p in fetch_loras(s):
                print(f"lora: {p.name} ({p.stat().st_size / 1e6:.0f} MB)")
