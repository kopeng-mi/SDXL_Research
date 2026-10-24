# Overhaul plan (step4)

For the agents who implement this. Read `docs/HANDOVER.md` first (state, testing workflow, user preferences), then this. Terse on purpose. Research date: 2026-10-24.

## For the implementer (GLM 5.3 in Z Code)
- The plan was researched and written by another agent; the user's rules live outside this repo, so here they are:
  - Plan with the user before anything non-trivial that this file doesn't already decide. Ask questions in chat as numbered options, recommended one marked (`Q1. ...  1. A (recommended)`); the user answers tersely, e.g. `1:2 2:1`.
  - **Ask first** before installing any package or tool, deleting large things (check size first), and any git history or remote change other than pushing this work branch. Commit only small conventional commits, no co-author lines.
  - Replies: answer first, short, plain words, bold key terms, no filler. Give shell commands for the user in **nushell**.
  - Windows 11 host; shells: PowerShell 7 and Git Bash. Prefer `rg`, `fd`, `jq`, `sd`, `uv`.
  - Public repo: no personal data. Tokens (HF_TOKEN, CIVITAI_TOKEN, TS_AUTHKEY) come from Colab Secrets only, never files or commits. `dataset_test/` photos never leave this PC except to a test VM.
  - Free Colab T4: stop every VM when done.
  - Verify before saying done: run tests, the fake-engine UI, or the T4 run, and report failures as they are.
- Tools named in HANDOVER (`colab` CLI, `wsf` web search) are installed on this PC; check `--help` before use. For browser checks use Z Code's built-in browser. Where this plan says "Haiku user tests", read it as "subagent UI tests, ask the user first".
- Library facts below were checked against the pinned versions; recheck anything not listed against the pinned source (`diffusers==0.40.0`, `nicegui==3.18.0`, `transformers==5.17.0`) before relying on memory.

## Decisions (user, final)
| # | Decision |
|---|---|
| UI | Keep NiceGUI. Restructure; interaction-heavy parts (brush mask, before/after) become browser-side JS components |
| Engine | Runs in its own worker process. The web process never imports torch |
| Inpaint | Crop-and-stitch + feather + paste-back by default for both models; Z-Image ControlNet lite as an optional "better blend" |
| HTTPS | Skip Tailscale Serve/HTTPS (cert limit 5 per 7 days per name, names land in public CT logs) |
| Text encoder | Default to the unsloth pre-quantized 4-bit Qwen3 after a quality check on T4 |
| Branch | New branch `step4-overhaul` from `step3-nicegui`; phases P0 to P5 in order; merge to `main` only after full testing (no squash), then BRANCH in `cells/00_setup.py` back to `main` |

Still to ask the user when reached (ask first, never assume): `pytest` as a dev tool; `pillow-heif` for iPhone HEIC uploads; GGUF CUDA kernels (`kernels` package + `DIFFUSERS_TRUST_REMOTE_KERNELS`); downloading the ostris training adapter; Haiku subagent UI tests.

## Test data
`dataset_test/` (git-ignored, personal photos: never commit, never push to HF or anywhere public, delete from the VM after tests). 11 photos: 6 JPG (720x960 to 1080x1440, 70 to 230 KB), 5 PNG (941x1672 to 1478x1064, about 2 MB each). Two PNGs share 941x1672: check for duplicates. No captions yet.

## Research results (verified, with what it means for us)
- **diffusers stays at 0.40.0** (newest is 0.42.0, 2026-10-23; needs Py >= 3.11). 0.40 already ships `ZImageInpaintPipeline`, `ZImageImg2ImgPipeline`, `ZImageControlNetPipeline`, `ZImageControlNetInpaintPipeline`, `ZImageControlNetModel`, and community `examples/community/pipeline_z_image_differential_img2img.py` (soft masks). kohya/musubi pins stay valid.
- **`ZImageInpaintPipeline` (0.40)**: has `strength`, blends latents outside the mask each step (`latents = (1 - mask) * init_latents_proper + mask * latents`), but **binarizes the mask** (`do_binarize=True`) and has **no `padding_mask_crop`** and no pixel overlay. So feathering, crop-and-stitch and paste-back are ours. SDXL inpaint has `padding_mask_crop`, but use our own crop-stitch for both, for one code path.
- **Z-Image ControlNet Union 2.1** (`alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`): full files 6.71 GB (too big next to Q8 on a T4); **lite files 2.01 GB**: `...Union-2.1-lite-2601-8steps.safetensors` (canny, depth, pose, mlsd, hed, scribble, inpaint), `...Union-2.1-lite-2602-8steps` (+gray), `...Tile-2.1-lite-2601-8steps` (upscale). Use 8 steps, `controlnet_conditioning_scale` 0.65 to 1.0 (higher for inpaint continuity). Lite allows higher scale with natural results.
  - **Lite is not auto-detected by diffusers 0.40**: detection keys on `control_layers.14...`; lite has `control_layers.0-2` and `control_noise_refiner.0-1`, `control_all_x_embedder.2-1.weight` [3840,132] (so `control_in_dim` 33), bf16. Load with `ZImageControlNetModel.from_single_file(path, config=<dir>, torch_dtype=torch.float16)` where `<dir>/config.json` = `hlky/Z-Image-Turbo-Fun-Controlnet-Union-2.1` config with `control_layers_places: [0, 10, 20]` (from VideoX-Fun `config/z_image/z_image_control_2.1_lite.yaml`), `control_refiner_layers_places: [0, 1]`, `add_control_noise_refiner: "control_noise_refiner"`, `control_in_dim: 33`. Ship that config in the repo. Read the Tile lite header before assuming the same config.
  - **ControlNet inpaint pipeline (0.40)**: no `strength`, starts from noise, **requires `control_image`** (VAE-encodes it), concatenates [control latents, mask, masked init latents]. For plain inpaint (no pose/depth) we need zero control latents: check how VideoX-Fun does inpaint-only, then write a minimal subclass. No latent blending outside the mask, so paste-back is mandatory.
  - **fp16 fix**: `zimage_fp16.patch` matches `isinstance(block, transformer_z_image.ZImageTransformerBlock)`. The ControlNet module has its own copied `ZImageTransformerBlock` class and `ZImageControlTransformerBlock` (same `feed_forward.w3/w2`, `attention`, `attention_norm2`, `ffn_norm2`, plus `before_proj`/`after_proj`). Patch by structure (has those attributes) and clamp `after_proj` output. Test for black images.
- **Nunchaku int4 Z-Image**: supports Turing since 2026-01-11 (PR #833), about 3x faster, but newest wheels (v1.3.0dev20260306) stop at torch 2.12 (we pin 2.14.1) and diffusers-path Z-Image LoRA (PR #866) is still open. **Rejected** (identity work needs LoRAs). Recheck later.
- **unsloth `Z-Image-Turbo-unsloth-bnb-4bit`**: text encoder `Qwen3ForCausalLM` nf4 double-quant, 3.32 GB (official bf16 TE is 7.9 GB download and gets quantized on every load); DiT bnb-4bit 3.28 GB. Both store `bnb_4bit_compute_dtype: bfloat16` and keep some modules unquantized in bf16: on T4 force fp16 (set `compute_dtype` on every `Linear4bit` after load, load with `dtype=torch.float16`). Our code loads the TE with `AutoModel` (base `Qwen3Model`); check the `model.` prefix maps correctly. The DiT variant is a candidate "small" model: benchmark vs GGUF.
- **TAEF1** (`madebyollin/taef1`, `AutoencoderTiny`, 16 latent channels, scaling 1.0, shift 0) fits Z-Image's VAE (Flux VAE: scaling 0.3611, shift 0.1159). Preview decode: `taef1.decode(latents / 0.3611 + 0.1159)`. SDXL: `madebyollin/taesdxl`.
- **SageAttention**: no T4 support (bf16 PTX errors on sm75). Skip. SDPA stays.
- **GGUF CUDA kernels**: about 10% faster, needs `kernels` package and trusting remote code. Low priority, ask first.
- **Layerwise casting** (fp8 storage, fp16 compute) exists but needs non-GGUF weights; not worth it with 12.7 GB RAM.
- **NiceGUI 3.18.0 is the newest.** Binding to a plain dict = "active links" polled every 0.1 s on the event loop (`binding_refresh_interval`). Our whole form state is a dict: move to `@binding.bindable_dataclass` (since 2.11). Custom components: `class X(ui.element, component='x.js')`, plain Vue object, no build step; `self.run_method(...)` returns values; `$emit` sends events.
- **NiceGUI socket.io uses the default `max_http_buffer_size` (1 MB)**: a bigger client message disconnects the client. Send masks and other blobs by HTTP POST to an app route, never over the websocket.
- **Colab**: on RAM exhaustion Colab kills the whole session ("Your session crashed after using all available RAM"). Keep total RAM under about 11.5 GB of 12.7 GB; our own guard must act first.
- **Linux OOM score**: any process may raise its own `oom_score_adj` (to 1000) without privileges; lowering needs root (Colab runs as root).
- **Tailscale**: newest 1.104.2 (we pin it). Userspace mode forwards tailnet connections to localhost, so the app sees 127.0.0.1 and can't map a request to a peer. `tailscale status --json` peers expose `CurAddr` (direct ip:port, empty when relayed) and `Relay` (DERP region): use to show "direct" vs "relayed" (relayed adds lag). **Taildrop**: needs "Send Files" on in the admin console; Linux receives with `tailscale file get [--loop] [--conflict=rename] DIR`; works only between the same user's devices; unverified in userspace mode: test before building on it.
- **Crop-and-stitch practice** (ComfyUI-Inpaint-CropAndStitch, 2025-04 rewrite): hipass the mask (ignore values under ~0.1), optional fill holes, grow mask N px, context = mask bbox extended by a factor (default about 1.5 to 2), crop before resize, keep the crop inside the image when possible (extend edges, never mirror), resize crop to target size, pad to a multiple of 16 (Z) / 8 (SDXL), stitch with a grown and blurred blend mask, pixel-exact paste-back.
- **LoRA fuse (SDXL)**: `set_adapters(...)` then `fuse_lora(adapter_names=..., lora_scale=1.0)`; `unfuse_lora()` to change. fp16 fuse/unfuse may drift slightly over many switches. GGUF and bnb weights can't be fused (Z-Image LoRAs stay unfused).
- **Z-Image LoRA training**: musubi recommends training on the **Base** model since January (our preset already uses `Comfy-Org/Z-Image` `z_image_bf16.safetensors`, shift 2.0, adamw8bit, lr 1e-4). The alternative is Turbo with a de-distill training adapter (`--base_weights`, ostris adapter v2, or `ostris/Z-Image-De-Turbo` as the DiT). Base-trained LoRAs may need strength above 1 on Turbo. A/B test both on `dataset_test`.
- **Z-Image Edit** (instruction editing) is not released; Tongyi-MAI has only Turbo and Base.

## Target architecture
```
Colab kernel (stdlib) -> nb.serve -> web process (NiceGUI, no torch)
                                       |- engine worker (torch; one at a time; restartable)
                                       |- job processes (caption/train/compare/download), as now
```
```
src/sdxlr/
  paths.py        dirs (WORK, MODELS, LORAS, ...) + model catalog constants (Z_REPO, Z_MODELS, Z_ENCODERS,
                  FP16_VAE, CONTROLNETS, PREVIEW_VAES). No heavy imports: fetch, UI and jobs import this
  loraformat.py   header-only LoRA inspection (family, rank, has text encoder, trigger words)
  engine/
    worker.py     process main: request loop, cancel event, heartbeat with memory parts, exit when parent dies
    client.py     (web side) start/restart/kill worker, send requests, turn events into activity Tasks
    models.py     Engine: load_sdxl, load_zimage with part reuse, unload, fp16 patch by structure
    loras.py      adapter cache by VRAM budget, set_loras, SDXL fuse policy, family check
    prompts.py    SDXL weighted encode (today's prompt.py) + Z-Image embedding cache
    run.py        txt2img, img2img, inpaint, controlnet inpaint, outpaint calls
    edit.py       pure PIL/numpy: size caps, mask ops, context box, stitch, outpaint canvas (no torch, unit-tested)
    upscale.py    tile plan (pure) + tile run (img2img or Tile ControlNet)
    preview.py    TAEF1/TAESDXL step previews
    fake.py       fake engine for local UI tests (SDXLR_FAKE_ENGINE=1), replaces colab_tests/dev_ui.py monkeypatching
    zimage_fp16.py
  service/
    state.py      bindable dataclass per session, autosave on change (debounced)
    files.py      cached indexes (history, models, LoRAs, datasets), refreshed in a thread; UI never scans disk
    guard.py      RAM/disk watchdog, oom_score_adj, health log, crash log export, pidfiles
    gen.py        state -> engine request (today's generate_blocking minus the engine)
    jobs.py       JobManager + submit_* (from ui/core.py)
    library.py    shred, sizes, downloads
  ui/ theme.py, frame.py, pages/*, components/mask_painter.{py,js}, components/before_after.{py,js}
  net/tailnet.py
  train.py, train_zimage.py, caption.py, compare.py, data.py, fetch.py, hfsync.py   (stay; compare.py imports engine.models)
```
Rule for all UI code: **no disk access and no blocking calls in timers or handlers**; read cached indexes, push work to threads or the worker.

## Worker protocol
- Start with `multiprocessing.get_context("spawn").Process(target=engine.worker.main, args=(conn, cancel_event))`; the web process imports only `engine.client`. Images travel as file paths, never pickled pixels.
- Requests: `{"id", "op": "load"|"unload"|"generate"|"edit"|"upscale"|"status", "args"}`. Events: `{"id", "type": "stage"|"step"|"preview"|"result"|"error"|"heartbeat", ...}`. One op at a time; the client keeps a queue of one and rejects more with a plain message.
- Cancel: set the event; the worker checks in step callbacks and between stages. During a non-interruptible stage (`from_single_file`, `from_pretrained`) Cancel kills the worker (instant, nothing to lose) and the next request restarts it.
- Heartbeat every 2 s: VRAM parts (today's `resources.model_parts`), adapters, allocated/reserved, worker RSS. The torch parts of `resources.py` move into the worker.
- Death: `exitcode` -9 or the pipe closes. Then the task fails with "The image engine ran out of memory and was stopped. Your images are safe; press Generate to start it again." Log the last health line.
- Before a GPU job: kill the worker (guaranteed free VRAM), restart lazily after.
- Worker sets its own `oom_score_adj` to 1000; job processes 800; web process -500 (root on Colab, ignore failure locally). The worker exits if its parent pid changes (orphan check every 2 s).
- Pidfiles in `WORK/run/`: on web start, kill a stale worker; for stale training jobs, either re-adopt (tail their log) or kill with a notice (start with kill + notice).

## Phases

### P0 Stability (first)
1. `paths.py`; `fetch.py` prefetch imports only `paths` (today it imports `engine` and so torch, +1 to 2 GB RAM).
2. Engine worker + client + `fake.py`; Activity tasks fed from worker events; Cancel semantics above.
3. `guard.py`: thread at 1 s reading `/proc/meminfo` `MemAvailable`; under the kill threshold (start 700 MB, tune on T4) kill the worker, then a running GPU job if still low; toast and log. Refuse load or generate when the estimated need (file sizes from `paths` catalog; record measured peaks in P2) exceeds available RAM, with a message naming what holds RAM. Health line in `app.log` every 30 s (RAM, VRAM, worker RSS, disk free). On a worker death or job failure, copy `logs/` to Drive (`/content/drive/MyDrive/sdxlr_logs/`) if mounted and to the HF sync repo if set.
4. Input caps: keep uploaded originals untouched; work copies capped (edit/whole about 1.2 MP, longest side 2048); upload size limit 50 MB; refuse under 256 px.
5. `service/files.py` cached indexes; remove every disk scan from timers (today `generate.py` tick at 0.5 s runs `history.items()` twice, `z_names()`/`model_names()` rglob and reads a JSON).
6. State to `bindable_dataclass` (no more active links); migrate saved session JSON (unknown keys dropped, missing keys default).
7. Stage timing (encode, denoise, decode, save) in the result info and the log.
Accept: local fake run works end to end; `rg "import torch" src/sdxlr/ui src/sdxlr/service` is empty; killing the worker by hand shows the message and the next Generate works; no timer touches the disk.

### P1 Image editing
UI (on the shown image or an upload): **Edit** opens tools over the image: Brush, Eraser, Box, Invert, Clear, brush size, Feather. Mode: **Change painted part** (inpaint), **Change whole image** (img2img), **Extend** (outpaint, per side or to an aspect), **Upscale** (1.5x, 2x). Strength slider with plain hints. Variations: count up to 4 (run one after another, not batched). Result: before/after slider (client side), Keep, Try again (new seed), Undo (via the `source` link in the JSON). Save the mask PNG next to the result, so "edit again" can reuse it.
Mask painter component: canvas over the image, pointer events (mouse, touch, pen), mask kept at natural image size, uploaded by `fetch` POST to `/api/mask/<session>` as PNG (8-bit L), then an emitted event with the stored id.
Inpaint pipeline (`edit.py` + `run.py`), both models:
1. Source: `exif_transpose`, RGB (alpha composited on white; first frame of animated files).
2. Mask: resize to source size; hipass 0.1; optional fill holes; grow by `max(8, 1% of short side)` px.
3. Context box = mask bbox x 1.75 (min 512 px side), kept inside the image; if disjoint regions make the box over 2x the regions' area, process regions one by one.
4. Crop, then resize so the crop is about 1 MP (small regions get upscaled: the main quality win), multiple of 16 (Z) / 8 (SDXL).
5. Run: Z-Image `ZImageInpaintPipeline` (strength, default 0.85) or ControlNet-lite inpaint (opt-in, scale 0.9, 8 steps); SDXL `StableDiffusionXLInpaintPipeline` (strength 0.75).
6. Resize back; blend mask = grown mask blurred by Feather (default 12 px); composite on the crop; paste into the original. Pixels outside the blend mask stay byte-identical (test this).
Whole image: work copy at about 1 MP (keep aspect, /16), run, save at work size (info says the size); option "Keep original size" resizes back with Lanczos.
Outpaint: new canvas; fill new area with edge-extended pixels + blur + light noise; mask = new area + 32 px band into the original; inpaint at strength 1.0 through the same crop-stitch.
Upscale: today's tiles (1024, overlap 128, feathered) with the prompt encoded once; Z-Image option "Tile ControlNet" (lite, 8 steps, scale about 0.8) per tile. Output cap 8 MP.
Stretch: soft-mask editing with the community differential img2img (vendor the file, Apache-2.0, record source and version).
Accept: on T4 with `dataset_test`: change clothes on a photo (both models) with no visible seam at feather 12; unmasked pixels identical; outpaint 25% each side; upscale 2x a 1 MP image under 2 minutes with no tile seams; a 24 MP upload works without RAM spikes.

### P2 Speed + benchmark (one T4 session, about 45 min)
1. Z-Image prompt embedding cache (key: prompt, TE id, TE-touching LoRAs). Upscale tiles reuse it.
2. Live previews (TAEF1 / TAESDXL), every 2nd step, JPEG q70 to `WORK/previews/<task>.jpg`, shown under the progress bar.
3. Display copies: WebP q90 saved by the worker beside each PNG plus thumbs; UI shows WebP, Download gives PNG.
4. unsloth TE: compare against today's nf4 (load time, peak RAM, embedding cosine similarity, same-seed images). Default it if images match closely.
5. Part reuse: changing only the DiT or only the TE keeps the other parts.
6. Benchmarks at 896x1152, Speed mode, fixed seed: GGUF Q8, Q6, Q4 and unsloth bnb-4bit DiT: s/step, total, peak VRAM, peak RAM during load, quality grid. SDXL: 0, 1, 3 LoRAs unfused vs fused; fuse only if the gain is over 10%. Record numbers in `docs/PLAN.md`.
7. VAE tiling only for decodes over 1.5 MP.
Accept: numbers recorded; Speed mode per-image time at or under today's 38 s, with preview visible from step 2.

### P3 LoRAs and models
1. `loraformat.inspect(path)` from the safetensors header only (cache by mtime): family by key names. SDXL: `lora_unet_(input|output|middle)_blocks|down_blocks|up_blocks|mid_block`, `lora_te1_/lora_te2_`, `unet.`. Z-Image: `layers.N.attention|feed_forward|noise_refiner|context_refiner` with `diffusion_model.`/`transformer.`/`lora_unet_layers_` prefixes. Flux (`double_blocks`, `single_blocks`) and SD1.5 (no te2, 768 dims): refuse. Also rank, text-encoder keys, metadata (`ss_tag_frequency` top tags, `ss_output_name`, `modelspec.*`).
2. Training writes a sidecar `<lora>.json` (trigger, class word, dataset, base, steps) for every saved LoRA; the UI reads it first.
3. UI: LoRA picker filtered by the picked model's family (others shown disabled with the reason); trigger chips insert into the prompt; warn when the trigger is missing. Strength range SDXL -1..2, Z-Image 0..2.5.
4. Adapter cache by VRAM budget (1.5 GB) instead of count 12.
5. Checkpoint check before load: an SDXL slot refuses SD1.5/Flux files with a plain message.
Accept: an SDXL LoRA can't be picked for Z-Image; trained LoRAs show their trigger.

### P4 UI rework
- Generate: prompt first; compact model chip (opens a model sheet: model, quality, prompt reader, VAE); mode and size; LoRAs with triggers; Generate. Image stage with live preview, action bar (Edit, Upscale, Variations, Reuse, Download, Delete), recent strip. Edit tools open over the image.
- Phone layout (single column, touch painting); a quick check on a phone over Tailscale.
- Sidebar: connection "direct" or "relayed (region)" from `tailscale status --json`.
- Taildrop inbox (only if the userspace test passes and the user turned on Send Files): `tailscale file get --loop --conflict=rename WORK/inbox`; Dataset page shows the inbox, with "add to dataset".
- Leftovers: stage text during downloads; dataset tile heights; menu items as real buttons; import count (0/6 vs 5); Activity "Done" row keeping download text; Compare grid zoom and per-cell download.
Accept: two Haiku user tests (ask first) find no blocker; the Generate page stays responsive with 1000 History items (fake engine).

### P5 Cleanup and merge
Remove the Gradio `app.py` and the gradio pin; update `README`, `docs/PLAN.md`, `docs/HANDOVER.md`; rebuild `notebook.ipynb`; full T4 run (below); merge to `main` without squash; BRANCH back to `main`.

## Edge cases (each needs a test or a deliberate decision)
**Inputs**: 24 MP / 20 MB photo; tiny under 256 px; 1:4 aspect; CMYK JPEG; 16-bit PNG; PNG with alpha; animated GIF/WebP; HEIC (no decoder: plain message unless pillow-heif is approved); corrupt or renamed non-image; EXIF rotation; duplicate photos in a dataset (hash check on import).
**Masks**: empty (ask to paint), full (switch to whole image), a few pixels (grow), touching the border, disjoint far-apart regions, mask drawn on a scaled display (send natural size), mask from an earlier image of another size (refuse).
**Prompts**: empty (allowed for upscale and inpaint, with a hint), longer than 512 Qwen tokens (warn about truncation), SDXL weights syntax errors (plain message).
**Memory**: low RAM before load (refuse, name the RAM holder); CUDA OOM (retry with VAE tiling, then a message suggesting a smaller size); RAM exhaustion (guard kills the worker, session survives); ControlNet + Q8 + TE over budget (encode first, then park or free the TE; measure); many runs (fragmentation, `empty_cache`).
**Models and LoRAs**: wrong family; deleted file still in a preset or history (skip with a warning); huge-rank LoRA (VRAM budget evicts); corrupt safetensors; gated or private HF repo without a token (plain message); Civitai without a token; disk full mid-download (message, remove the partial file); interrupted download (resume).
**Concurrency**: Generate while training (refuse with the job's ETA); two tabs on one session (last write wins, both see progress); Cancel in every stage (download, load, encode, denoise, decode, save); double click; web restart while the worker or a job runs (pidfiles); same-second file names (unique suffix).
**Network**: relayed Tailscale path (show it); websocket drop and reconnect (state survives, task continues); upload interrupted (remove the partial file); messages over 1 MB (HTTP only); stale browser cache (`?v=mtime`).
**Colab**: VM recycled (only HF sync or Drive survive: remind in Library); TS_AUTHKEY missing or expired (plain message in the Launch cell, local address fallback); ephemeral node name reuse (`sdxlr`, else `sdxlr-N`); idle disconnect when only the app is used and the Colab tab sits idle: measure, then document "keep the Colab tab open" or add a keep-alive if needed.

## Test plan
- **Local, no GPU**: unit tests for `edit.py` (crop box, stitch exactness, grow/feather, outpaint canvas), `upscale.py` tile plan, `loraformat.py` (fixture headers written in the test), state migration, guard thresholds with fake meminfo. UI with `SDXLR_FAKE_ENGINE=1` plus browser walkthroughs and screenshots (see HANDOVER testing workflow, port 7870).
- **T4 session A (end of P2, about 45 min)**: health log and guard (a synthetic RAM hog must kill the worker, not the session); 24 MP upload; loads Q8/Q6/Q4/unsloth with peaks; stage timings; TE comparison; LoRA fuse benchmark.
- **T4 session B (end of P4)**: editing on `dataset_test` (inpaint both models, ControlNet lite, outpaint, whole, upscale both ways); phone check.
- **T4 session C (training, 2 to 3 h, stop the VM right after)**: import `dataset_test` (dedupe check) -> captions (WD and Qwen) -> SDXL LoRA (11 x 6 repeats x 15 epochs / batch 3 = 330 steps, about 30 min) -> Z-Image LoRA on Base at 512 (11 x 4 x 10 = 440 steps, about 42 min) and, if approved, Turbo + training adapter -> compare grids -> generate and face-inpaint with the LoRAs -> face similarity with `colab_tests/faces_common.py`. Then delete the photos, captions and LoRAs from the VM (Library > Delete everything) unless the user says otherwise.
- Budget GPU time: one VM per session, stop it when done; never leave a VM idle.

## Commit plan
Small conventional commits per numbered item (for example `refactor(paths): constants without torch`, `feat(engine): worker process with cancel and heartbeat`, `feat(guard): RAM watchdog and health log`). Push `step4-overhaul` after each phase. No co-author lines.
