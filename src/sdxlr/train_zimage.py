"""Train a Z-Image LoRA with kohya musubi-tuner. Runs inside the main venv.

  python -m sdxlr.train_zimage --name myrun --images DIR [--set key=value ...]

T4 plan (15 GB VRAM, 12.7 GB RAM, no bf16/fp8 compute):
  1. cache latents (VAE only) and text-encoder outputs (Qwen3 only), each in
     its own process, so neither sits in VRAM during training
  2. train the DiT in fp16 (musubi has the fp16 overflow fix) with weights
     stored as scaled fp8 (half the memory; compute stays fp16), 512px default
  3. convert every saved LoRA to diffusers format (*_diffusers.safetensors),
     which the app/compare load on top of Z-Image Turbo

Auto resume: with --hf-repo and --resume, the dataset and the latest LoRA +
state go to that private repo (hfsync.py keeps only the newest checkpoint);
a later run with the same name restores the dataset if missing and trains the
remaining epochs. musubi restarts its epoch counter on resume, so a resumed
run is named NAME_rDONE and its files are renumbered to the real epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import toml
import yaml

from . import hfsync, loraformat
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
    ap.add_argument("--hf-repo", help="private HF repo for LoRAs + resume state")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--trigger", default="", help="trigger word, saved next to each LoRA")
    ap.add_argument("--class-word", default="", help="class word, saved next to each LoRA")
    a = ap.parse_args()
    if a.resume and not a.hf_repo:
        sys.exit("--resume needs --hf-repo")
    if a.resume and not any(a.images.glob("*")) and not hfsync.pull_dataset(a.hf_repo, a.name, a.images):
        sys.exit(f"no images in {a.images} and none in {a.hf_repo}/{a.name}/dataset")

    c = yaml.safe_load((PRESETS / f"{a.preset}.yaml").read_text())
    c |= {k: _cast(v) for k, v in (s.split("=", 1) for s in a.set)}

    out = WORK / "runs" / a.name
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    done, state, run_name = 0, None, a.name
    if a.resume:
        found = hfsync.pull_latest(a.hf_repo, a.name, out / "resume")
        if found:
            state, done = found
            run_name = f"{a.name}_r{done}"
        print(f"resume from {state} ({done} epochs done)" if state else "no state found, starting fresh", flush=True)
    left = int(c["epochs"]) - done
    if left <= 0:
        sys.exit(f"all {c['epochs']} epochs already done; raise epochs to train more")

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
        "--max_train_epochs", str(left), "--save_every_n_epochs", str(c["save_every"]),
        "--seed", str(c["seed"]), "--max_data_loader_n_workers", "1", "--persistent_data_loader_workers",
        "--output_dir", str(out), "--output_name", run_name,
    ]
    if c["fp8"]:
        args += ["--fp8_base", "--fp8_scaled"]  # fp8 storage only; T4 computes in fp16
    if c.get("blocks_to_swap"):
        args += ["--blocks_to_swap", str(c["blocks_to_swap"])]  # uses system RAM (only 12.7 GB on Colab)
    pusher = None
    if a.resume:
        args += ["--save_state", "--save_last_n_epochs_state", "1"]
        if state:
            args += [f"--resume={state}"]
        hfsync.push_dataset(a.hf_repo, a.name, a.images)
        pusher = hfsync.Pusher(a.hf_repo, out, a.name)
        pusher.start()
    _run("zimage_train_network.py", *args)
    if pusher:
        pusher.stop_ev.set()  # final push after renaming + conversion

    if run_name != a.name:  # renumber a resumed run to the real epoch: NAME_r3-000002 -> NAME-000005
        for f in out.glob(f"{run_name}*.safetensors"):
            m = re.fullmatch(rf"{re.escape(run_name)}-(\d+)", f.stem)
            new = f"{a.name}-{int(m[1]) + done:06d}" if m else a.name
            f.rename(f.with_name(new + f.suffix))

    for lora in sorted(out.glob(f"{a.name}*.safetensors")):
        if lora.stem.endswith("_diffusers"):
            continue
        dst = lora.with_name(f"{lora.stem}_diffusers.safetensors")
        if not dst.exists():
            _run("convert_lora.py", "--input", str(lora), "--output", str(dst), "--target", "other")
    n_imgs = len([p for p in a.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    steps = -(-n_imgs * int(c["repeats"]) * int(c["epochs"]) // max(1, int(c["batch_size"])))
    for lora in out.glob("*.safetensors"):  # the diffusers files are what the app loads
        loraformat.sidecar_path(lora).write_text(json.dumps({k: v for k, v in dict(
            trigger=a.trigger, class_word=a.class_word, dataset=a.images.name, base="Z-Image Base", steps=steps,
            kind="zimage").items() if v}, indent=1), encoding="utf-8")
    print(f"LoRAs in {out}: {sorted(p.name for p in out.glob('*_diffusers.safetensors'))}", flush=True)
    if pusher:
        pusher.finish()


if __name__ == "__main__":
    main()
