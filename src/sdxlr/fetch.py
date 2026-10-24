"""Download base models and LoRAs into /content/work. Runs inside the main venv.

  python -m sdxlr.fetch model SRC          SDXL checkpoint
  python -m sdxlr.fetch zimage SRC         Z-Image transformer (.gguf or .safetensors)
  python -m sdxlr.fetch loras SRC [SRC..]  .safetensors or .zip (flattened)
  python -m sdxlr.fetch prefetch-zimage TRANSFORMER TEXT_ENCODER   every file a Z-Image load needs
  python -m sdxlr.fetch prefetch-sdxl                              the fp16-fix VAE

The app runs prefetch as a child process before loading a model, so a download
can be cancelled (the process is killed; partial files resume next time).

SRC = org/repo/file, a huggingface.co blob/resolve link, a Civitai link, any
https URL, or a local path (upload, Drive). Private HF repos use HF_TOKEN.

Downloads stream to disk (never held in RAM); zips are deleted after extraction.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from .paths import LORAS, MODELS, WEIGHTS, WORK, ZIMAGE


def _hf_link(src: str) -> tuple[str, str, str] | None:
    """huggingface.co/org/repo/(blob|resolve)/rev/path -> (repo, path, rev)."""
    m = re.match(r"https?://(?:www\.)?(?:huggingface\.co|hf\.co)/([^/]+/[^/]+)/(?:blob|resolve)/([^/]+)/(.+?)(?:\?.*)?$", src)
    return (m[1], m[3], m[2]) if m else None


def fetch(src: str, dst_dir: Path) -> Path:
    """'org/repo/file', an https URL, or a local path -> local file. Errors come back as plain sentences."""
    if Path(src).exists():
        return Path(src)
    dst_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _fetch(src, dst_dir)
    except subprocess.CalledProcessError as e:
        if e.returncode == 22 and "civitai.com" in src:  # curl -f: HTTP error (401/403 without a token)
            raise RuntimeError("Civitai refused the download: add CIVITAI_TOKEN in Colab Secrets "
                               "(civitai.com > Account > API keys), run Setup again, then retry.") from None
        if e.returncode == 23:
            raise RuntimeError("The download stopped: the disk is full. Free disk space in Library, then retry.") from None
        raise RuntimeError(f"The download failed (curl exit {e.returncode}): {src}. Check the link and retry.") from None
    except Exception as e:  # huggingface_hub errors are OSErrors too: match them by name first
        name = type(e).__name__
        if getattr(e, "errno", None) == 28:  # ENOSPC
            raise RuntimeError("The download stopped: the disk is full. Free disk space in Library, then retry.") from None
        if name in ("GatedRepoError",):
            raise RuntimeError("This Hugging Face model is gated: accept its terms on the model page, add HF_TOKEN in "
                               "Colab Secrets, run Setup again, then retry.") from None
        if name in ("RepositoryNotFoundError", "EntryNotFoundError", "RemoteEntryNotFoundError"):
            raise RuntimeError(f"Not found on Hugging Face: {src}. Check the name (private repos need HF_TOKEN).") from None
        raise


def _curl(args: list[str], dst: Path, src: str) -> None:
    """Download to <dst>.part (resumed if a partial is there) and rename only when complete:
    an interrupted download never leaves a file that looks finished."""
    part = dst.with_name(dst.name + ".part")
    subprocess.run(["curl", "-fL", "--retry", "3", "-C", "-", *args, "-o", str(part), src], check=True)
    part.replace(dst)


def _fetch(src: str, dst_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download  # downloads to .incomplete and resumes on its own
    if hf := _hf_link(src):
        return Path(hf_hub_download(hf[0], hf[1], revision=hf[2], local_dir=dst_dir))
    if src.startswith("http"):
        if "civitai.com" in src:
            # file name comes from Content-Disposition; token from Colab Secrets / env
            import os
            import tempfile
            hdr = ["-H", f"Authorization: Bearer {os.environ['CIVITAI_TOKEN']}"] if os.environ.get("CIVITAI_TOKEN") else []
            with tempfile.TemporaryDirectory(dir=dst_dir, prefix=".dl_") as tmp:  # a failed download leaves nothing
                r = subprocess.run(["curl", "-fLJ", "--retry", "3", *hdr, "--output-dir", tmp, "-O",
                                    "-w", "%{filename_effective}", src], check=True, capture_output=True, text=True)
                got = Path(r.stdout.strip())
                dst = dst_dir / got.name
                shutil.move(str(got), dst)
            return dst
        dst = dst_dir / src.split("?")[0].rsplit("/", 1)[-1]
        if not dst.exists():
            _curl([], dst, src)
        return dst
    org, repo, file = src.split("/", 2)
    return Path(hf_hub_download(f"{org}/{repo}", file, local_dir=dst_dir))


def add_file(path: str, dst_dir: Path) -> Path:
    """Move an uploaded file into the library (keeps its name)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / Path(path).name
    shutil.move(path, dst)
    return dst


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


def prefetch_zimage(transformer: str, text_encoder: str) -> None:
    from huggingface_hub import snapshot_download

    from .paths import Z_REPO
    if ":" in transformer and not Path(transformer).exists():  # 'org/repo:subfolder' (pre-quantized diffusers folder)
        repo, _, sub = transformer.partition(":")
        snapshot_download(repo, allow_patterns=[f"{sub}/*"])
        print(f"model file: {transformer}", flush=True)
    else:
        print(f"model file: {fetch(transformer, MODELS / 'zimage')}", flush=True)
    repo, _, sub = text_encoder.partition(":")
    snapshot_download(repo, allow_patterns=[f"{sub}/*"] if sub else None)
    print(f"text encoder: {text_encoder}", flush=True)
    # tokenizer, VAE, scheduler and configs; the transformer comes from the file above
    snapshot_download(Z_REPO, ignore_patterns=["transformer/*.safetensors", "text_encoder/*.safetensors"])
    print("configs + VAE ready", flush=True)


def prefetch_sdxl() -> None:
    from huggingface_hub import snapshot_download

    from .paths import FP16_VAE
    snapshot_download(FP16_VAE)
    print("fp16 VAE ready", flush=True)


def prefetch_controlnet(name: str) -> None:
    from .paths import CONTROLNETS
    repo, fname = CONTROLNETS[name]
    print(f"controlnet: {fetch(f'{repo}/{fname}', MODELS / 'zimage_cn')}", flush=True)


def prefetch_preview(kind: str) -> None:
    from huggingface_hub import snapshot_download

    from .paths import PREVIEW_VAES
    snapshot_download(PREVIEW_VAES[kind])
    print(f"preview VAE ready ({PREVIEW_VAES[kind]})", flush=True)


def list_models() -> list[Path]:
    """SDXL checkpoints only (Z-Image parts live in models/zimage*/)."""
    return sorted(p for p in MODELS.rglob("*.safetensors")
                  if not any(part.startswith("zimage") for part in p.relative_to(MODELS).parts))


def list_zimage() -> list[Path]:
    """User-added Z-Image transformers (Library: link or upload)."""
    return sorted(p for p in ZIMAGE.glob("*") if p.suffix in WEIGHTS) if ZIMAGE.exists() else []


def list_loras() -> list[Path]:
    """Downloaded LoRAs + everything trained in this session."""
    return sorted({*LORAS.glob("*.safetensors"), *(WORK / "runs").glob("*/*.safetensors")}, key=lambda p: p.name)


if __name__ == "__main__":
    kind, *srcs = sys.argv[1:]
    if kind == "prefetch-zimage":
        prefetch_zimage(*srcs)
        sys.exit()
    if kind == "prefetch-controlnet":
        prefetch_controlnet(*srcs)
        sys.exit()
    if kind == "prefetch-preview":
        prefetch_preview(*srcs)
        sys.exit()
    if kind == "prefetch-sdxl":
        prefetch_sdxl()
        sys.exit()
    for s in srcs:
        if kind in ("model", "zimage"):
            p = fetch(s, MODELS if kind == "model" else ZIMAGE)
            print(f"model: {p} ({p.stat().st_size / 1e9:.2f} GB)")
        else:
            for p in fetch_loras(s):
                print(f"lora: {p.name} ({p.stat().st_size / 1e6:.0f} MB)")
