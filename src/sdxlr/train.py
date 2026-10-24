"""Train an SDXL LoRA with kohya sd-scripts. Runs inside the main venv.

  python -m sdxlr.train --name myrun --images DIR [--set key=value ...]

Defaults come from configs/train/sdxl_person.yaml (proven T4 settings).
Training runs as a child process; when it exits all VRAM/RAM is released.

Auto resume: with --hf-repo and --resume, the dataset and the latest LoRA +
state go to that private repo (hfsync.py keeps only the newest checkpoint);
a later run with the same name restores the dataset if missing and continues.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import toml
import yaml

from . import hfsync, loraformat
from .fetch import MODELS, fetch

REPO = Path(__file__).resolve().parents[2]
KOHYA = REPO / "vendor" / "sd-scripts"
PRESETS = REPO / "configs" / "train"
WORK = Path("/content/work")


def _cast(v: str):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None}.get(v.lower(), v)


def build_configs(c: dict, model: Path, images: Path, out: Path) -> tuple[Path, Path]:
    module_args = []
    if c["lora_type"] == "LoCon":
        module_args = [f"conv_dim={c['conv_dim']}", f"conv_alpha={c['conv_alpha']}"]
    train = {
        "network_arguments": {
            "network_module": "networks.lora",
            "network_dim": c["network_dim"],
            "network_alpha": c["network_alpha"],
            "network_args": module_args or None,
            "unet_lr": c["unet_lr"],
            "text_encoder_lr": c["text_encoder_lr"],
            "network_train_unet_only": c["text_encoder_lr"] == 0,
            "network_weights": c.get("continue_from") or None,
        },
        "optimizer_arguments": {
            "learning_rate": c["unet_lr"],
            "optimizer_type": c["optimizer"],
            "lr_scheduler": c["lr_scheduler"],
            "lr_scheduler_num_cycles": c["lr_cycles"] if c["lr_scheduler"] == "cosine_with_restarts" else None,
            "lr_warmup_steps": c["lr_warmup_ratio"] if c["lr_scheduler"] not in ("constant", "cosine") else None,
            "max_grad_norm": 1.0,
            "loss_type": "l2",
        },
        "training_arguments": {
            "pretrained_model_name_or_path": str(model),
            "max_train_epochs": c["epochs"],
            "train_batch_size": c["batch_size"],
            "seed": c["seed"],
            "max_token_length": 225,
            "sdpa": True,
            "mixed_precision": "fp16",  # T4 has no fast bf16
            "full_fp16": c["full_fp16"],
            "no_half_vae": True,  # SDXL VAE NaNs in fp16
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "max_data_loader_n_workers": 1,
            "persistent_data_loader_workers": True,
            "lowram": False,
            "min_snr_gamma": c["min_snr_gamma"] or None,
            "ip_noise_gamma": c["ip_noise_gamma"] or None,
            "multires_noise_iterations": 6 if c["multires_noise"] else None,
            "multires_noise_discount": 0.3 if c["multires_noise"] else None,
        },
        "saving_arguments": {
            "output_dir": str(out),
            "output_name": c["name"],
            "save_model_as": "safetensors",
            "save_precision": "fp16",
            "save_every_n_epochs": c["save_every"],
            "save_last_n_epochs": c["keep_last"],
            "save_state": bool(c.get("hf_repo")),
            "save_last_n_epochs_state": 1,
            "logging_dir": str(out / "logs"),
        },
    }
    for sec in train.values():
        for k in [k for k, v in sec.items() if v is None]:
            del sec[k]

    dataset = {
        "general": {
            "resolution": c["resolution"],
            "caption_extension": ".txt",
            "shuffle_caption": c["shuffle_caption"],
            "keep_tokens": 1,  # keep the trigger first when shuffling
            "enable_bucket": True,
            "bucket_no_upscale": True,
            "min_bucket_reso": 256,
            "max_bucket_reso": 2048,
            "flip_aug": False,
        },
        "datasets": [{"subsets": [{"image_dir": str(images), "num_repeats": c["repeats"]}]}],
    }
    cfg, ds = out / "train.toml", out / "dataset.toml"
    cfg.write_text(toml.dumps(train))
    ds.write_text(toml.dumps(dataset))
    return cfg, ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--images", type=Path, required=True, help="images + .txt captions")
    ap.add_argument("--preset", default="sdxl_person")
    ap.add_argument("--model", help="override preset model: org/repo/file, URL or path")
    ap.add_argument("--hf-repo", help="private HF repo for LoRAs + resume state")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--trigger", default="", help="trigger word, saved next to each LoRA")
    ap.add_argument("--class-word", default="", help="class word, saved next to each LoRA")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()

    c = yaml.safe_load((PRESETS / f"{a.preset}.yaml").read_text())
    c |= {k: _cast(v) for k, v in (s.split("=", 1) for s in a.set)}
    c["name"], c["hf_repo"] = a.name, a.hf_repo or c.get("hf_repo")
    if a.model:
        c["model"] = a.model

    if a.resume and not c["hf_repo"]:
        sys.exit("--resume needs --hf-repo")
    if a.resume and not any(a.images.glob("*")) and not hfsync.pull_dataset(c["hf_repo"], a.name, a.images):
        sys.exit(f"no images in {a.images} and none in {c['hf_repo']}/{a.name}/dataset")
    caps = list(a.images.glob("*.txt"))
    imgs = [p for p in a.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    if not imgs:
        sys.exit(f"no images in {a.images}")
    print(f"{len(imgs)} images, {len(caps)} captions", flush=True)

    out = WORK / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    model = fetch(c["model"], MODELS)
    cfg, ds = build_configs(c, model, a.images, out)

    cmd = [sys.executable, "-u", str(KOHYA / "sdxl_train_network.py"),
           f"--config_file={cfg}", f"--dataset_config={ds}"]
    pusher = None
    if a.resume:
        found = hfsync.pull_latest(c["hf_repo"], a.name, out / "resume")
        print(f"resume from {found[0]} ({found[1]} epochs done)" if found else "no saved state, starting fresh", flush=True)
        if found:
            cmd.append(f"--resume={found[0]}")
        hfsync.push_dataset(c["hf_repo"], a.name, a.images)
        pusher = hfsync.Pusher(c["hf_repo"], out, a.name)
        pusher.start()

    shutil.copy(cfg, out / "train.used.toml")
    print(" ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=KOHYA, env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
    if pusher:
        pusher.finish()
    steps = -(-len(imgs) * int(c["repeats"]) * int(c["epochs"]) // max(1, int(c["batch_size"])))
    loraformat.write_sidecars(out, trigger=a.trigger, class_word=a.class_word, dataset=a.images.name, base=str(c["model"]),
                              steps=steps, kind="sdxl")  # the UI reads these first
    print(f"LoRAs in {out}: {sorted(p.name for p in out.glob('*.safetensors'))}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
