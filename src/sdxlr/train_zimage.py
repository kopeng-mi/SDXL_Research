"""Train a Z-Image LoRA with kohya musubi-tuner. Runs inside the main venv.

  python -m sdxlr.train_zimage --name myrun --images DIR [--set key=value ...]

T4 plan (15 GB VRAM, 12.7 GB RAM, no bf16/fp8 compute):
  1. cache latents (VAE only) and text-encoder outputs (Qwen3 only), each in
     its own process, so neither sits in VRAM during training
  2. train the DiT in fp16 (musubi has the fp16 overflow fix) with weights
     stored as scaled fp8 (half the memory; compute stays fp16), 512px default
  3. convert every saved LoRA to diffusers format (*_diffusers.safetensors),
     which the app/compare load on top of Z-Image Turbo

Presets: --preset best (zimage_person.yaml, Base) or fast (zimage_fast.yaml,
Turbo + ostris adapter); repeats 0 = auto (plan.auto). --te picks the text
encoder (paths.Z_ENCODERS keys); test and generate with the same one, since
training caches its outputs.

Auto resume: with --drive ROOT (or --hf-repo) and --resume, every epoch goes
to ROOT/loras/zimage/NAME and the newest state to ROOT/state/NAME; a later run
with the same name trains the remaining epochs. musubi restarts its epoch
counter on resume, so a resumed run is named NAME_rDONE and its files are
renumbered to the real epoch.
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

from . import drive, hfsync, loraformat, plan
from .fetch import MODELS, WORK, fetch

REPO = Path(__file__).resolve().parents[2]
MUSUBI = REPO / "vendor" / "musubi-tuner"
PRESETS = REPO / "configs" / "train"

FILES = {
    "base": "Comfy-Org/Z-Image/split_files/diffusion_models/z_image_bf16.safetensors",
    "vae": "Comfy-Org/Z-Image/split_files/vae/ae.safetensors",
    "te": "Comfy-Org/Z-Image/split_files/text_encoders/qwen_3_4b.safetensors",
    "turbo": "Comfy-Org/Z-Image-Turbo/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
    # ostris's de-distill adapter: training on Turbo with it merged in (musubi --base_weights), use on plain Turbo
    "ostris": "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors",
}
# text encoders as musubi reads them (HF Qwen3ForCausalLM keys; split files load from the first shard)
TE_FILES = {
    "official": [FILES["te"]],
    "abliterated": ["BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1/model.safetensors"],
    "heretic": ["DreamFast/qwen3-4b-heretic/comfyui/qwen3-4b-heretic.safetensors"],
    "engineer": [f"BennyDaBall/Z-Image-Engineer-V6/model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)],
}
PRESET_FILES = {"best": "zimage_person", "fast": "zimage_fast"}


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
    ap.add_argument("--preset", default="best", help="best | fast | a configs/train/*.yaml name")
    ap.add_argument("--te", default="official", help=f"{', '.join(TE_FILES)} or org/repo/file")
    ap.add_argument("--subject", default="person", choices=list(plan.SUBJECTS))
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--drive", type=Path, help="Drive root: epochs + resume state are kept there")
    ap.add_argument("--hf-repo", help="private HF repo for LoRAs + resume state")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--trigger", default="", help="trigger word, saved next to each LoRA")
    ap.add_argument("--class-word", default="", help="class word, saved next to each LoRA")
    a = ap.parse_args()
    if a.resume and not (a.hf_repo or a.drive):
        sys.exit("--resume needs --drive or --hf-repo")
    if a.resume and a.hf_repo and not any(a.images.glob("*")) and not hfsync.pull_dataset(a.hf_repo, a.name, a.images):
        sys.exit(f"no images in {a.images} and none in {a.hf_repo}/{a.name}/dataset")

    c = yaml.safe_load((PRESETS / f"{PRESET_FILES.get(a.preset, a.preset)}.yaml").read_text())
    c |= {k: _cast(v) for k, v in (s.split("=", 1) for s in a.set)}
    n_imgs = len([p for p in a.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    if not n_imgs:
        sys.exit(f"no images in {a.images}")
    p = plan.train("zimage", a.subject, n_imgs, int(c["resolution"]), int(c["batch_size"]), int(c["epochs"]),
                   int(c["repeats"]))
    c["repeats"], c["epochs"] = p.repeats, p.epochs
    print(p.summary(), flush=True)

    out = WORK / "runs" / a.name
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    d_out = a.drive / "loras" / "zimage" / a.name if a.drive else None
    d_state = a.drive / "state" / a.name if a.drive else None
    done, state, run_name = 0, None, a.name
    if a.resume:
        if a.drive:
            state, done = drive.resume(d_state, out / "resume", a.name)
        elif found := hfsync.pull_latest(a.hf_repo, a.name, out / "resume"):
            state, done = found
        if state:
            run_name = f"{a.name}_r{done}"
        print(f"resume from {state} ({done} epochs done)" if state else "no state found, starting fresh", flush=True)
    left = int(c["epochs"]) - done
    if left <= 0:
        sys.exit(f"all {c['epochs']} epochs already done; raise epochs to train more")

    d = MODELS / "zimage_train"
    dit = fetch(FILES.get(c.get("dit") or "base", c.get("dit")), d)  # "base" | "turbo" | org/repo/file
    vae = fetch(FILES["vae"], d)
    te_files = TE_FILES.get(a.te, [a.te])
    te = [fetch(f, d / f"te_{a.te}" if a.te in TE_FILES else d) for f in te_files][0]  # musubi finds the other shards

    res = int(c["resolution"])
    dataset = {
        "general": {"resolution": [res, res], "caption_extension": ".txt", "batch_size": int(c["batch_size"]),
                    "enable_bucket": True, "bucket_no_upscale": True},
        "datasets": [{"image_directory": str(a.images), "cache_directory": str(cache),
                      "num_repeats": int(c["repeats"]),
                      "flip_aug": bool(c.get("flip_aug", plan.SUBJECTS[a.subject]["flip"]))}],
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
        "--save_precision", "fp16",  # musubi saves fp32 by default (twice the size, same LoRA)
    ]
    if c.get("adapter"):  # e.g. "ostris" with dit: turbo
        args += ["--base_weights", str(fetch(FILES.get(c["adapter"], c["adapter"]), d))]
    if c["fp8"]:
        args += ["--fp8_base", "--fp8_scaled"]  # fp8 storage only; T4 computes in fp16
    if c.get("blocks_to_swap"):
        args += ["--blocks_to_swap", str(c["blocks_to_swap"])]  # uses system RAM (only 12.7 GB on Colab)
    pusher = syncer = None
    if a.resume or a.drive:
        args += ["--save_state", "--save_last_n_epochs_state", "1"]
    if state:
        args += [f"--resume={state}"]
    if a.drive:  # raw epochs as saved; the converted files follow after training
        syncer = drive.Syncer(out, a.name, d_out, d_state)  # done() also reads NAME_rN states
        syncer.start()
    elif a.resume:
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
    for lora in out.glob("*.safetensors"):  # the diffusers files are what the app loads
        loraformat.sidecar_path(lora).write_text(json.dumps({k: v for k, v in dict(
            trigger=a.trigger, class_word=a.class_word, dataset=a.images.name, steps=p.steps, te=a.te, subject=a.subject,
            base={"turbo": "Z-Image Turbo"}.get(c.get("dit") or "base", "Z-Image Base")
            + (f" + {c['adapter']} adapter" if c.get("adapter") else ""),
            kind="zimage").items() if v}, indent=1), encoding="utf-8")
    print(f"LoRAs in {out}: {sorted(p.name for p in out.glob('*_diffusers.safetensors'))}", flush=True)
    if pusher:
        pusher.finish()
    if syncer:
        syncer.stop()
        for pat in ("*.safetensors", "*.json"):
            drive.mirror_new(out, d_out, pat, settle=0)


if __name__ == "__main__":
    main()
