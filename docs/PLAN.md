# SDXL_Research: plan, decisions, findings

Living notes. Update when a decision or finding changes.

## Goal

One notebook for the **Colab free tier (T4)**, run from the **Colab web UI**:
caption a dataset, train identity LoRAs (SDXL and Z-Image), compare them, and
generate with a simple, clean UI that mixes LoRAs (Fooocus-like ease of use).

## Hard limits (T4 free tier)

| Resource | Limit | Consequence |
|---|---|---|
| VRAM | 15 GB | one base model at a time; Z-Image needs GGUF + 4-bit text encoder |
| System RAM | ~12.7 GB | the real bottleneck when loading; stream weights, avoid CPU offload |
| Precision | no fast bf16, no fp8 compute | fp16 everywhere; fp8 only as weight storage |
| GPU time | ~5 h per account | test on the CPU VM first, keep T4 sessions to one goal |

## Decisions

| # | Decision |
|---|---|
| UI | NiceGUI web app over Tailscale (step3+); Gradio removed in step4 (the pin stays: musubi-tuner depends on it) |
| Engine | its own worker process (step4): the web process never imports torch; RAM guard + oom_score_adj keep the session alive |
| Editing | crop-and-stitch inpaint with feathered, pixel-exact paste-back for both models; Z-Image ControlNet lite as "better blend" |
| Notebook | 2 cells only: **Setup** (installs everything) and **Launch** (opens the app) |
| Env | uv-managed Python 3.12 venv at `/content/venv`; never the Colab system Python |
| Processes | the kernel only orchestrates; every GPU job is a child process in the venv |
| Persistence | private HF repo, synced from the Library tab; token from Colab Secrets (`HF_TOKEN`), repo names set in the UI |
| Presets | JSON files (prompt, model, mode, LoRAs + weights, size, seed) |
| Captioner | Qwen3-VL-4B default, Qwen3-VL-2B, JoyCaption Beta One (nf4); Florence-2 dropped |
| Caption style | identity preset: clothing, pose, angle, framing, background, light; never face/body/skin; `ohwx_person` trigger + class word |
| SDXL trainer | kohya sd-scripts v0.12.2 (submodule) |
| Z-Image trainer | kohya musubi-tuner v0.3.6 (submodule); train on Z-Image **base**, use on **Turbo** |
| Git | conventional commits, small and scoped; `main` = tested; one branch per step |

## Pinned stack

torch 2.14.1 (cu130), diffusers 0.40.0, transformers 5.17.0, accelerate 1.15.0,
peft 0.21.2, gradio 6.30.0 (musubi's), bitsandbytes 0.50.2, nicegui 3.18.0, pillow-heif 1.8.1. kohya and musubi pin the same
diffusers/transformers/accelerate, so everything shares one venv.

## Findings (measured on T4)

| Item | Result |
|---|---|
| venv install | 107 s fresh, 9 s re-run |
| Qwen3-VL-4B captions | ~3.4 s/image, ~145 s first download + load |
| SDXL LoRA training (kohya) | batch 3, 1024 px, full fp16: ~5.1 s/step |
| SDXL generation | 896x1152, 25 steps: ~20-26 s, peak ~9 GB |
| Compare grid | 2 images in 52 s |
| Z-Image Turbo, plain fp16 | **black images** (NaN; Tongyi-MAI/Z-Image#14) |
| Z-Image Turbo, fp32 | works, 166 s/image |
| Z-Image Turbo, fp16 + overflow fix | works, **~42 s/image**, peak 11.9 GB (Q8 GGUF + nf4 encoder) |
| Indian faces | Z-Image much better: right faces, skin tones, clothes, cultural details. Pony V17: generic faces, westernized details, ignores clothes, NSFW drift ("saree" became lingerie) |
| Z-Image seed variety | low (turbo trait): two seeds look almost the same |
| Z-Image LoRA training (musubi) | base DiT, fp16 + fp8 weight storage, 512 px, batch 1: **~5.7 s/step**, peak 13.9 GB VRAM, 9.8 GB RAM; 1024 px expected to OOM |
| Z-Image Turbo in the engine | 768x1024, 9 steps: ~30 s/image; first load ~9 min (downloads) |
| UI pass, SDXL base 896x1152 | Speed (20 steps) 18 s, Quality (30) 25 s, **Fast (Lightning 8) 10 s**, inpaint 16 s |
| UI pass, Z-Image Q8 896x1152 | Fast (6) 28 s, Speed (9) 41 s, Quality (12) 54 s, inpaint 34 s; VRAM shows 14.5 GB (mostly CUDA cache) |
| WD EVA02 v3 tagger | works on T4; tags like "dog, tongue out, orange background" |

## Findings step4 (T4, 2026-10-24, `tools/t4_check.py`)

| Item | Result |
|---|---|
| Z-Image Q8 Speed 896x1152 | **38.6 s cold, 35.1 s warm** (denoise 27-30 s, decode 2 s); live preview from step 2 |
| Q6_K / Q4_K_M | 42-46 s: **slower than Q8** (K-quant dequant cost); they only save disk/VRAM. Q8 stays the default |
| Q8 load | 193 s incl. downloads; worker RAM peak 9.6 GB (3.4 GB left free) before the DiT went to the GPU early |
| unsloth bnb-4bit text encoder | its load next to a RAM-resident Q8 tripped the RAM guard; rerun after the early-GPU fix (see below) |
| Painted edit (Z, ~1 MP crop) | plain 37-39 s; ControlNet lite "better blend" 47-51 s (peak VRAM 13.8 GB) |
| Extend 25 % per side / whole image | 45 s / 26 s |
| Upscale 2x (Z) | 136-181 s for ~4.7 MP (6-8 tiles of 1024); tile ControlNet 347 s. Tiles now balanced (fewest of <= 1280 px) |
| 24 MP photo, painted edit | 38 s, worker RAM 2.8 GB (work copy capped at ~1.2 MP) |
| SDXL base | load 61 s; gen 20 s; painted 19 s; extend 27 s; whole 15 s; upscale 2x 57 s |
| RAM guard | a synthetic hog (0.5 GB/s) in the worker: **session survived**, next Generate worked. At 700 MB the kernel OOM killer won the race, so the guard now acts at 1 GB |
| ControlNet left loaded | 2 GB VRAM next to Q8 pushed the next upscale into OOM retries: it is now freed when an op doesn't use it |

## Bugs found and fixed

- kohya text-encoder LoRAs failed to load in diffusers: transformers 5 dropped
  `text_model.` from CLIPTextModel (te1) but not from te2. Fixed per encoder.
- Failed LoRA load left its adapter name taken. Now cleaned up.
- Inpaint doubled VRAM: `from_pipe` without `torch_dtype` cast shared models to fp32.
- Old compare notebook's clip-skip hack removed 2 extra encoder layers; clip skip 2
  (A1111 numbering) is diffusers' default for SDXL.
- diffusers truncates prompts at 77 tokens and ignores `(word:1.3)`; own encoder added.
- Old caption prompt ("a woman or a man") made the model invent a photographer.

## Architecture

```
notebook: Setup cell -> bootstrap venv; Launch cell -> python -m sdxlr.webui (NiceGUI, no torch)
web process: pages read cached file indexes; RAM guard; job queue
 ├─ engine worker process (torch): SDXL or Z-Image, LoRA cache by VRAM budget, prompt caches, edits,
 │   previews; restartable, killed for GPU jobs and when RAM runs low
 ├─ job processes: caption / train / compare / import / download, one GPU job at a time
 └─ /content/work: models, loras, runs, outputs, sessions, logs  <-> private HF repo
```

## UI plan (5 tabs)

- **Generate**: model picker (only T4-fit options), Quality/Speed/Fast modes,
  aspect presets, prompt + styles, LoRA slots with sliders, gallery -> inpaint/vary,
  Advanced folded (steps, CFG, sampler, clip skip, encoder, VAE), JSON presets.
- **Dataset**: drop a zip with or without captions, or an HF/Drive link; grid
  with editable captions; captioner choice; trigger + class word.
- **Train**: SDXL (kohya) or Z-Image (musubi); simple fields + Advanced;
  ETA before start from measured s/step, refined live; loss + samples.
- **Compare**: LoRAs or mixes x strengths, same seed, grid.
- **Library**: paste a link (HF, Civitai, URL) for models/LoRAs; sizes; delete;
  HF sync; repo settings.
- Header: VRAM/RAM, queue status with ETA and cancel.

## Status

Done (branch `step1-scaffold`): bootstrap, notebook builder, captioner, SDXL
trainer, dataset fetch, prompt encoder, engine (SDXL + Z-Image), compare,
first Gradio app, job queue, presets, settings, Z-Image trainer (untested).

Z-Image end to end works on T4: train (512 px) -> convert to diffusers ->
load on Turbo (attaches to 210 transformer modules) -> generate; Z-Image inpaint
works (~30 s). The 30-step test LoRA was too weak to change the image (lora_B
norm ~0.09); a real run is needed to judge identity learning.

ETA rule of thumb for the Train tab: steps = images x repeats x epochs / batch;
SDXL 1024 px batch 3 ~5.1 s/step; Z-Image 512 px batch 1 ~5.7 s/step; plus
caching (~2-4 min) and first-run downloads.

New UI (5 tabs + Jobs) checked on a Colab CPU VM through a browser: renders,
dataset import as a job with auto-refresh, caption editor, Library download,
JSON presets save/load. GPU paths in the UI (Load, Generate, Train, Compare,
Caption) still need a T4 pass.

Branch `step2-polish` (T4 UI pass done: every GPU path works from the app's
handlers; header bars, load preview and session restore checked in a browser):
- Library: HF `blob/`/`resolve/` links, private repos (HF_TOKEN), uploads,
  custom Z-Image checkpoints (shown in the model pickers)
- simpler tabs (extras under Advanced); Generate loads the picked model itself
- WD tagger captioner (timm)
- header bars: VRAM/RAM/disk split into named blocks; hatched preview of a
  picked model (red if it won't fit); "Clear download cache" in Library
- 4-digit session code: UI state saved on the VM every 5 s and after each
  generation; reload restores it, the code restores it in another browser
- auto resume (both trainers): HF repo keeps only `NAME/ckpt/` (latest LoRA +
  state, replaced each push) and `NAME/dataset/`; same name later = continue

Next:
1. User test from Colab web: Z-Image identity run with auto resume (needs HF_TOKEN; untested here)
2. 768 px Z-Image training
3. Merge `step2-polish` into `main`

Branch `step4-overhaul` (overhaul plan in `docs/OVERHAUL_PLAN.md`, state in `docs/HANDOVER.md`):
engine worker process + RAM guard, brush-mask editing (painted part / whole / extend / upscale) with
crop-and-stitch and ControlNet lite, live previews, Z-Image prompt cache, LoRA family checks and trigger chips,
HEIC uploads, dataset dedupe, NiceGUI rework (toolbar, sticky image, connection line). Measured on T4 above.

## Testing notes (for the agent)

- `colab exec` drops long cells (10 s read timeout): run long jobs detached on the
  VM (`nohup ... > /content/logs/x.log`) and poll with short commands.
- Git Bash rewrites `/content/...` paths: use `MSYS_NO_PATHCONV=1` for `colab upload`.
- The Herdr side pane runs nushell; check for the VM prompt before sending bash.
- The home router's DNS often refuses new `*.colab.dev` hosts: run the colab CLI through a wrapper that resolves
  them via 1.1.1.1 (patch `socket.getaddrinfo` before importing `colab_cli.cli`), or switch networks.
- Long checks run detached on the VM (`tools/t4_check.py`, `tools/train_check.py`) and write
  `WORK/t4_check/report.md` / `WORK/train_check/report.md`; poll their logs with short `colab exec` calls.

