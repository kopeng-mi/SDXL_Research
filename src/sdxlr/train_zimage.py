"""Train a Z-Image LoRA with kohya musubi-tuner. Runs inside the main venv.

  python -m sdxlr.train_zimage --name myrun --images DIR [--set key=value ...]

T4 plan (15 GB VRAM, 12.7 GB RAM, no bf16/fp8 compute):
  1. cache latents (VAE only) and text-encoder outputs (Qwen3 only), each in
     its own process, so neither sits in VRAM during training
  2. train the DiT in fp16 (musubi has the fp16 overflow fix) with weights
     stored as scaled fp8 (half the memory; compute stays fp16), 512px default
  3. convert every saved LoRA to diffusers format (*_diffusers.safetensors),
     which the app/compare load on top of Z-Image Turbo

Resume (simple): with --hf-repo, LoRAs + training state are uploaded to that
private repo each epoch. --resume downloads the newest state and trains the
remaining epochs. musubi restarts its epoch counter on resume, so a resumed
run is named NAME_rDONE and its files are renumbered to the real epoch.
"""

from __future__ import annotations

import argparse
import os
import re
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


def _done(state: str, name: str) -> int:
    """Epochs finished by a state dir: NAME-000003-state or NAME_r3-000002-state."""
    m = re.fullmatch(rf"{re.escape(name)}(?:_r(\d+))?-(\d+)-state", state)
    return int(m[1] or 0) + int(m[2]) if m else -1


def latest_state(hf_repo: str, name: str, dst: Path) -> tuple[Path, int] | None:
    from huggingface_hub import HfApi, snapshot_download
    states = {f.split("/")[0] for f in HfApi().list_repo_files(hf_repo) if "-state/" in f}
    best = max(states, key=lambda s: _done(s, name), default=None)
    if best is None or _done(best, name) < 0:
        return None
    snapshot_download(hf_repo, allow_patterns=[f"{best}/*"], local_dir=dst)
    return dst / best, _done(best, name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--images", type=Path, required=True, help="images + .txt captions")
    ap.add_argument("--preset", default="zimage_person")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--hf-repo", help="private HF repo for LoRAs + resume state")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.resume and not a.hf_repo:
        sys.exit("--resume needs --hf-repo")

    c = yaml.safe_load((PRESETS / f"{a.preset}.yaml").read_text())
    c |= {k: _cast(v) for k, v in (s.split("=", 1) for s in a.set)}

    out = WORK / "runs" / a.name
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    done, state, run_name = 0, None, a.name
    if a.resume:
        found = latest_state(a.hf_repo, a.name, out / "resume")
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
    if a.hf_repo:
        args += ["--save_state", "--save_last_n_epochs_state", "1", "--huggingface_repo_id", a.hf_repo,
                 "--huggingface_repo_type", "model", "--huggingface_repo_visibility", "private",
                 "--save_state_to_huggingface", "--async_upload"]
    if state:
        args += [f"--resume={state}"]
    if c["fp8"]:
        args += ["--fp8_base", "--fp8_scaled"]  # fp8 storage only; T4 computes in fp16
    if c.get("blocks_to_swap"):
        args += ["--blocks_to_swap", str(c["blocks_to_swap"])]  # uses system RAM (only 12.7 GB on Colab)
    _run("zimage_train_network.py", *args)

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
    print(f"LoRAs in {out}: {sorted(p.name for p in out.glob('*_diffusers.safetensors'))}", flush=True)


if __name__ == "__main__":
    main()
