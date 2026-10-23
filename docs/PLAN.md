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
| UI | Gradio 6 with a custom theme + CSS (not ComfyUI, not Fooocus) |
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
peft 0.21.2, gradio 6.30.0, bitsandbytes 0.50.2. kohya and musubi pin the same
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
notebook: Setup cell -> bootstrap venv; Launch cell -> python -m sdxlr.app
app (one process, owns the in-process image engine)
 ├─ JobManager: GPU jobs one at a time (caption/train/compare), CPU jobs alongside
 │   (downloads, zips); unloads the engine before a GPU job, reloads after
 ├─ engine: SDXL or Z-Image, LoRA adapter cache, prompt cache, OOM retry, inpaint
 └─ /content/work: models, loras, runs, presets, outputs, settings.json  <-> private HF repo
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

Next:
1. Confirm Z-Image LoRA effect on Turbo; try 768 px training
2. New UI (tabs above) on top of JobManager
3. Fast mode for SDXL (Lightning/Hyper LoRA), measure speeds
4. More captioners (e.g. WD tagger for tag-style captions)
5. Test the whole flow from Colab web, then merge to `main`

## Testing notes (for the agent)

- `colab exec` drops long cells (10 s read timeout): run long jobs detached on the
  VM (`nohup ... > /content/logs/x.log`) and poll with short commands.
- Git Bash rewrites `/content/...` paths: use `MSYS_NO_PATHCONV=1` for `colab upload`.
- The Herdr side pane runs nushell; check for the VM prompt before sending bash.
