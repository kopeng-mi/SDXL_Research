"""Train a Z-Image LoRA with kohya musubi-tuner. Runs inside the main venv.

  python -m sdxlr.train_zimage --name myrun --images DIR [--set key=value ...]

T4 plan (15 GB VRAM, 12.7 GB RAM, no bf16/fp8 compute):
  1. cache latents (VAE only) and text-encoder outputs (Qwen3 only), each in
     its own process, so neither sits in VRAM during training
  2. train the DiT in fp16 (musubi has the fp16 overflow fix) with weights
     stored as scaled fp8 (half the memory; compute stays fp16), 512px default
  3. convert every saved LoRA to diffusers format (*_diffusers.safetensors),
     which the app/compare load on top of Z-Image Turbo
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import toml
import yaml

from .fetch import MODELS, WORK, fetch

REPO = Path(__file__).resolve().parents[2]
MUSUBI = REPO / "vendor" / "musubi-tuner"
PRESETS = REPO / "configs" / "train"

FILES = {
    "base": "Comfy-Org/Z-Image/split_files/diffusion_models/z_image_bf16.safetensors",
    "vae": "Comfy-Org/Z-Image/split_files/vae/ae.safetensors",
    "te": "Comfy-Org/Z-Image/split_files/text_encoders/qwen_3_4b.safetensors",
}


def _cast(v: str):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None}.get(v.lower(), v)


def _run(script: str, *args: str) -> None:
    cmd = [sys.executable, "-u", str(MUSUBI / "src" / "musubi_tuner" / script), *args]
    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=MUSUBI, env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
    if rc:
        sys.exit(f"{script} failed with exit code {rc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--images", type=Path, required=True, help="images + .txt captions")
    ap.add_argument("--preset", default="zimage_person")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()

    c = yaml.safe_load((PRESETS / f"{a.preset}.yaml").read_text())
    c |= {k: _cast(v) for k, v in (s.split("=", 1) for s in a.set)}

    out = WORK / "runs" / a.name
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    d = MODELS / "zimage_train"
    dit = fetch(c.get("dit") or FILES["base"], d)
    vae = fetch(FILES["vae"], d)
    te = fetch(FILES["te"], d)

    res = int(c["resolution"])
    dataset = {
        "general": {"resolution": [res, res], "caption_extension": ".txt", "batch_size": int(c["batch_size"]),
                    "enable_bucket": True, "bucket_no_upscale": True},
        "datasets": [{"image_directory": str(a.images), "cache_directory": str(cache),
                      "num_repeats": int(c["repeats"])}],
    }
    ds = out / "dataset.toml"
    ds.write_text(toml.dumps(dataset))

    _run("zimage_cache_latents.py", f"--dataset_config={ds}", f"--vae={vae}")
    _run("zimage_cache_text_encoder_outputs.py", f"--dataset_config={ds}", f"--text_encoder={te}",
         "--batch_size", "4")

    args = [
        f"--dit={dit}", f"--vae={vae}", f"--text_encoder={te}", f"--dataset_config={ds}",
        "--sdpa", "--mixed_precision", "fp16", "--gradient_checkpointing",
        "--timestep_sampling", "shift", "--weighting_scheme", "none",
        "--discrete_flow_shift", str(c["flow_shift"]),
        "--optimizer_type", c["optimizer"], "--learning_rate", str(c["lr"]),
        "--network_module", "networks.lora_zimage",
        "--network_dim", str(c["network_dim"]), "--network_alpha", str(c["network_alpha"]),
        "--max_train_epochs", str(c["epochs"]), "--save_every_n_epochs", str(c["save_every"]),
        "--seed", str(c["seed"]), "--max_data_loader_n_workers", "1", "--persistent_data_loader_workers",
        "--output_dir", str(out), "--output_name", a.name,
    ]
    if c["fp8"]:
        args += ["--fp8_base", "--fp8_scaled"]  # fp8 storage only; T4 computes in fp16
    if c.get("blocks_to_swap"):
        args += ["--blocks_to_swap", str(c["blocks_to_swap"])]  # uses system RAM (only 12.7 GB on Colab)
    _run("zimage_train_network.py", *args)

    for lora in sorted(out.glob(f"{a.name}*.safetensors")):
        if lora.stem.endswith("_diffusers"):
            continue
        dst = lora.with_name(f"{lora.stem}_diffusers.safetensors")
        if not dst.exists():
            _run("networks/convert_lora.py", "--input", str(lora), "--output", str(dst), "--target", "other")
    print(f"LoRAs in {out}: {sorted(p.name for p in out.glob('*_diffusers.safetensors'))}", flush=True)


if __name__ == "__main__":
    main()
