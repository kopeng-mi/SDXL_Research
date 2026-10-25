# Step 5: notebooks plan

Branch `step5-notebooks` (from `step4-overhaul`, not merged). Supersedes the web app's dataset/caption/train/compare
pages. Read with `HANDOVER.md` (code map, T4 numbers in `PLAN.md`). Terse on purpose.

## Decisions (user, 2026-10-25)
| # | Decision |
|---|---|
| D1 | 5 notebooks: `captioning`, `train-test-sdxl`, `train-test-zimage`, `sdxl` (Fooocus), `zimage` (our app, trimmed) |
| D2 | Notebooks are thin: Colab forms (`#@param`) only, no widgets. Logic lives in `src/sdxlr`; each notebook clones the repo |
| D3 | Fooocus: official 2.5.5 in its own Python 3.11 venv, pinned (the old `fooocus_v2` recipe) |
| D4 | Captioning + training read from and write to Google Drive. HF stays a download source only |
| D5 | Abliterated options wherever a model can refuse (captioners, Z-Image text encoder, prompt writer) |
| D6 | Tailscale for `sdxl` (host `sdxl`, :7865) and `zimage` (host `zimage`, :7860) |
| D7 | Each notebook offers **Fast** and **Best** presets. Best = proven T4 settings; Fast = measured in P6 |

## Research facts (2026-10-25)
- Fooocus: no code since 2025-01 (v2.5.5). Colab breaks on py3.12, numpy, and cupy. Its `launch.py` pins torch 2.1.0 cu121. `config.txt` `path_checkpoints`/`path_loras` take lists (so Drive folders can be added). No v-pred support. Its prompt expansion (GPT-2) never refuses, so nothing to abliterate.
- Z-Image public weights: Turbo, Base (2026-01-27). Edit and Omni-Base: not public.
- Z-Image TE = `Qwen3ForCausalLM` (Qwen3-4B), `tie_word_embeddings: true`, so the lm_head is `embed_tokens` and the loaded TE can generate text.
- The official prompt-enhancer template is `spaces/Tongyi-MAI/Z-Image-Turbo/pe.py` (Apache-2.0; `{prompt}` slot, Chinese system text, outputs only the prompt).
- Trainer pins are the latest: kohya v0.12.2, musubi v0.3.6.
  - musubi changed its default `--save_precision` to fp32 (2026-06), so we must pass `--save_precision fp16`.
  - `--block_swap_h2d_only` is useful only with block swap (not used on T4).
  - LoHa/LoKr are available.
- transformers 5.17 has `qwen3_5` (Qwen3.5 0.8/2/4/9B; vision is native; hybrid GDN attention, so T4 speed is unknown).
- Florence-2 PromptGen needs transformers 4.51, so it is dropped (conflicts with the shared venv).

## Model catalogs (go in `paths.py`)
**Captioners** (`CAPTIONERS`: key -> repo, load mode)
| key | repo | file GB | load |
|---|---|---|---|
| qwen3vl-2b | Qwen/Qwen3-VL-2B-Instruct | 4.3 | fp16 |
| qwen3vl-4b (default) | Qwen/Qwen3-VL-4B-Instruct | 8.9 | fp16 |
| qwen3vl-4b-abl | huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated | 8.9 | fp16 |
| qwen3vl-8b-abl | huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated | 17.5 | nf4 |
| qwen35-4b-abl (exp) | huihui-ai/Huihui-Qwen3.5-4B-abliterated | 9.1 | fp16 |
| qwen35-9b-abl (exp) | huihui-ai/Huihui-Qwen3.5-9B-abliterated | 19.3 | nf4 |
| joy | fancyfeast/llama-joycaption-beta-one-hf-llava (uncensored by design) | 16 | nf4 LLM, fp16 vision |
| wd | SmilingWolf/wd-eva02-large-tagger-v3 | 1.2 | fp16 |

**Z-Image text encoders** (`Z_ENCODERS`; all HF format, root weights, Z_REPO tokenizer)
| key | repo | note |
|---|---|---|
| official | Tongyi-MAI/Z-Image-Turbo:text_encoder | default |
| abliterated | BennyDaBall/Qwen3-4b-Z-Image-Turbo-AbliteratedV1 | heretic on the Z TE, KL 0.0004, refusals 4/100 |
| heretic | DreamFast/qwen3-4b-heretic | heretic v1.2 on Qwen3-4B; ships `comfyui/*.safetensors` for musubi |
| engineer | BennyDaBall/Z-Image-Engineer-V6 | DoRA-tuned prompt writer + TE (changes the look per seed; not abliterated) |

Rule: train and test with the **same TE** (musubi caches TE outputs). The notebook warns on a mismatch (sidecar `te`).

**SDXL training bases** (`SDXL_BASES`): sdxl-base, Pony V6, Illustrious 0.1, NoobAI eps 1.1, NoobAI v-pred 1.0, plus custom (HF/Civitai/Drive link).
- v-pred is detected from the header (`v_pred` key), which sets `v_parameterization` + `zero_terminal_snr`.
- In Fooocus a v-pred checkpoint is refused with a plain message.

## Layout and wiring
```
cells/_setup.py                     shared Setup cell (template; GROUPS + NAME injected by the builder)
cells/<nb>/NN_*.py                  notebook cells  ->  notebooks/<nb>.ipynb  (tools/build_notebook.py, --check)
src/sdxlr/drive.py           NEW    Drive layout + mirror/push/pull (stdlib)
src/sdxlr/caption.py         CHG    catalog, styles, batching, quant, --mirror
src/sdxlr/train.py           CHG    presets fast/best, auto repeats, v-pred, --drive-out, resume from Drive
src/sdxlr/train_zimage.py    CHG    same + te choice, --save_precision fp16, custom DiT
src/sdxlr/compare.py         CHG    --epochs last:K | list, --drive-out, writes grid.json
src/sdxlr/plan.py            NEW    pure math: repeats, steps, ETA, VRAM fit, caption budget (tested)
src/sdxlr/fooocus.py         NEW    venv, pins, config.txt, launch (stdlib; runs on system python)
src/sdxlr/engine/enhance.py  NEW    prompt writer on the loaded TE
src/sdxlr/bootstrap.py       CHG    --groups (core always; caption|train-sdxl|train-z|app)
src/sdxlr/tailnet.py         CHG    hostname param
src/sdxlr/nb.py              CHG    run(py=...) for other venvs; show_images(); drive_mount()
REMOVE: hfsync.py, presets push/pull, ui/pages/{train,dataset,compare}.py, app SDXL kind, cells/00_setup.py,
        cells/10_launch.py, notebook.ipynb, tools/train_check.py (-> tools/t4_nb_check.py), webui --sync
```

**Drive layout** (`drive.ROOT` = `/content/drive/MyDrive/SDXLR`, form field)
```
datasets/<ds>/          images + .txt (captioning writes here)
loras/sdxl/<run>/       every saved epoch + sidecar json; <run>.safetensors = picked epoch
loras/zimage/<run>/     *_diffusers.safetensors (what the app loads) + raw musubi files
state/<run>/            newest resume state only
compare/<run>/          grid.png + grid.json
models/sdxl/, models/zimage/   optional user checkpoints (Fooocus and the app read them)
outputs/fooocus/, outputs/zimage/   optional (form toggle)
```
**Sync rules** (`drive.py`):
- `pull(src_dir, local)`: copy a file when `(size, mtime_ns)` differs. Training reads local copies (Drive FUSE is slow for many reads).
- `mirror(local_file, drive_dir)`: copy after each write (caption .txt, each epoch LoRA), so a disconnect loses at most one item.
- `push_state(run)`: a thread every 120 s copies the newest `*-state` dir as `state/<run>/<name>.tmp`, then renames it and deletes older ones. Rename makes it atomic: a half-copied state is never picked.
- `resume(run)`: the newest state dir; epochs done from the name (`hfsync.done` regex moves here).
- The last cell calls `drive.flush_and_unmount()` (Colab writes back lazily).

**Setup cell** (shared): clone/fetch BRANCH, then `python -m sdxlr.bootstrap --groups <G>`, then `nb.load_secrets()`, then `drive.mount('/content/drive')` when USE_DRIVE.
- The venv stamp = hash(pyproject, kohya reqs, musubi pyproject, sorted groups).
- Groups only add packages, so a VM that runs two notebooks keeps one venv with the same pins.

| notebook | groups | extra venv |
|---|---|---|
| captioning | core, caption | - |
| train-test-sdxl | core, caption, train-sdxl | - |
| train-test-zimage | core, caption, train-z | - |
| sdxl | none (stdlib + huggingface_hub in the Fooocus venv) | /content/fooocus-venv py3.11 |
| zimage | core, app | - |

### captioning
1. **Settings**: `SRC` (Drive folder \| zip link \| HF file), `DS_NAME`, `MODEL` (catalog), `STYLE` (`id-sdxl` \| `id-zimage` \| `tags` \| `detailed` \| `custom`), `TRIGGER`, `CLASS_WORD`, `MAX_WORDS` (0 = from style), `CUSTOM_PROMPT`, `OVERWRITE`, `BATCH` (0 = auto), `QUANT` (auto \| fp16 \| nf4).
2. **Import**: `-m sdxlr.data --dest WORK/datasets/DS SRC` (dedupe, HEIC), then mirror the images to `datasets/DS` on Drive.
3. **Preview**: `-m sdxlr.caption ... --limit 4`, then `nb.show_images(paths, captions)` (IPython.display in the kernel, no torch).
4. **Caption all**: `--mirror DRIVE/datasets/DS` (skip existing unless OVERWRITE).
5. **Review**: word/token stats, flags captions over budget or missing the trigger, optional `FIND`/`REPLACE` pass, zip download.

STYLES (prompt templates; `{cls}`, `{n}`):
- `id-sdxl`: the current IDENTITY_PROMPT, 30 to 50 words.
- `id-zimage`: the same rules, 80 to 150 words, full sentences (Z-Image likes long prompts).
- `tags`: WD only.
- `detailed`: general captions, no identity rules (style/concept LoRAs).

### train-test-sdxl / train-test-zimage (same shape)
1. **Dataset**: `DS` from Drive, then `drive.pull`. If captions are missing, caption here (style id-sdxl / id-zimage).
2. **Settings**: `BASE` (catalog \| link), `SUBJECT`, `PRESET` (Fast \| Best), `RUN`, `TRIGGER`, `CLASS_WORD`, `EPOCHS`/`REPEATS` (0 = auto), `DIM`, `LR` (blank = preset), `RESUME`; zimage also `TE`. Prints the plan: steps, ETA, VRAM fit (from `plan.py`).
3. **Train**: `-m sdxlr.train[_zimage] --name RUN --images LOCAL --preset sdxl_fast|sdxl_best --model BASE --drive-out DRIVE/loras/<kind>/RUN --resume`.
4. **Test**: `-m sdxlr.compare --epochs last:3 --strengths 0.6 0.8 1.0 --prompts P1 P2 --seed 42 --drive-out ...`. Shows the grid inline. Z tests run on Turbo (unsloth 4-bit) with the training TE.
5. **Pick**: `PICK_EPOCH` copies that epoch to `loras/<kind>/RUN/RUN.safetensors` (+ sidecar). The app and Fooocus list it.

### sdxl (Fooocus)
1. **Setup Fooocus** (`fooocus.py`, system python, stdlib):
   - uv venv py3.11 at /content/fooocus-venv;
   - clone `lllyasviel/Fooocus` at a pinned commit (`FOOOCUS_REF`);
   - `pip torch==2.1.0 torchvision==0.16.0 --index-url cu121`, then `-r requirements_versions.txt`, then `pygit2==1.15.1 numpy==1.26.4 huggingface_hub`, then uninstall cupy*;
   - idempotent stamp.
2. **Models**:
   - `CHECKPOINTS` (catalog multi-pick + links) and `LORAS` (links/zips) go through `PYTHONPATH=src fooocus-venv/python -m sdxlr.fetch` into `Fooocus/models/*`;
   - v-pred files are refused;
   - writes `config.txt`: `path_checkpoints=[local, DRIVE/models/sdxl]`, `path_loras=[local, DRIVE/loras/sdxl]`, `path_outputs` = local or Drive, `default_model` = the first checkpoint.
3. **Launch**: `PRESET` (default \| realistic \| anime); `nb.serve(["launch.py","--listen","0.0.0.0","--port","7865","--always-high-vram","--disable-analytics", "--preset", P], py=FOOOCUS_PY, cwd=Fooocus)`, then `tailnet.up(7865, hostname="sdxl")`. With no TS_AUTHKEY it falls back to `--share`.
4. **Wipe outputs** (optional, from the old notebook).

### zimage (our app, trimmed)
1. **Models** (optional prefetch):
   - `MODEL` (Z_MODELS + custom link);
   - `TE` (Z_ENCODERS);
   - `LORAS` links;
   - Drive LoRA and model folders are added to the app's search dirs.
2. **Launch**: `nb.serve(["-m","sdxlr.webui","--port",7860,"--host-name","zimage","--drive",ROOT])`.

App changes:
- Z-Image only;
- pages: Generate, History, Library, Activity;
- Library lists Drive LoRAs read-only (the engine loads them from the Drive path);
- **custom checkpoint** path (below);
- **Write a detailed prompt** button (below).

## Algorithms and math

**Training plan** (`plan.py`, pure, tested). N = images, R = repeats, B = batch, E = epochs, res = side px.
- Steps: `S = ceil(N·R / B) · E`.
- Auto repeats keep the proven number of image views V = N·R·E:
  - SDXL Best `V* = 990` (11·6·15), so `R = clamp(round(V*/(N·E)), 1, 20)`;
  - Z Best `V* = 440` (11·4·10).
  - Fewer photos get more repeats, more photos fewer.
  - Big sets: when `R = 1` still gives `N·E > 1.5·V*`, epochs shrink too: `E = clamp(round(V*/N), 4, E)`.
    Example (SDXL, E 15): N 11 gives R 6; N 40 gives R 2; N 200 gives R 1, E 5 (≈ 1000 views, 335 steps).
  - Below 5 photos the form warns: identity is weak and the LoRA overfits faster.
- **Subject type** (form field `SUBJECT`, sets caption style, V* and aug):
  | SUBJECT | captions | V* SDXL / Z | flip_aug | note |
  |---|---|---|---|---|
  | person (measured) | id-* (no face/body words) | 990 / 440 | off | faces are asymmetric |
  | character/object | id-* with "the {cls}" | 990 / 440 | off | est, same as person |
  | style | detailed (describe content, not style) | 1500 / 660 | on | est; TE lr 0 for SDXL |
  | concept/pose | detailed | 1200 / 530 | on | est |
  Non-person rows are starting points marked "est" until a real run confirms them.
- **Image checks at import** (`data.py`, printed and then kept or skipped):
  - min side < `0.5·res`: flagged (it is upscaled into a bucket and comes out soft);
  - exact duplicates: dropped (sha1, already done);
  - EXIF rotation: applied;
  - mixed aspect ratios: fine (buckets, area = res², side step 64 for SDXL and 16 for Z).
  - The plan cell prints the bucket spread so one odd bucket with a single image is visible.
- Step time: `t = t0 · (res/res0)² · (B/B0)`.
  - t0 is measured on T4: SDXL (1024, B3) 5.1 s; Z (512, B1) 5.7 s.
  - P6 replaces this with a table of measured points; anything else is marked "est".
- ETA: `S·t + overhead` (SDXL 120 s, Z 300 s; first run adds the download, `GB / 0.1 GB/s` est).
- Best totals at N = 11 (other N: steps stay near these, because V* is fixed):
  - SDXL: S = 22·15 = 330, so 330·5.1 + 120 ≈ **30 min**;
  - Z: S = 44·10 = 440, so 440·5.7 + 300 ≈ **47 min**.
- LoRA scale = `(alpha/dim) · strength` (SDXL 16/24 = 0.667). Changing dim keeps alpha/dim, so the LR needs no retune.
- Fast candidates (P6 measures; until then they are guesses):
  - SDXL: res 768 (pixels ×0.5625), TE frozen (`text_encoder_lr 0`, `cache_text_encoder_outputs`, requires `shuffle_caption false`), E 10, R auto.
  - Z: Turbo + ostris adapter (smoke epoch 505 s vs Base 570 s), E 6, lr 1.5e-4.
- Keep last K epochs; Drive space `≈ K · lora_MB` (SDXL dim 24 ≈ 170 MB fp16; Z dim 16 fp16 measured in P6).

**Captioning** (`plan.py`):
- VRAM fit: `GB ≈ P_B · bytes · 1.15 + vision`. bytes = 2 (fp16) or 0.56 (nf4 incl. scales). Use fp16 when it is ≤ 12.5 GB, else nf4.
  - 4B: 9.2, so fp16. 8B: 18.4, so nf4 ≈ 5.2. 9B: nf4 ≈ 5.8.
- Vision tokens (Qwen3-VL: patch 16, merge 2, so 32 px per token): `ceil(W/32)·ceil(H/32)`. 1024² = 1024 tokens; MAX_SIDE 768 gives 576 (faster prefill, enough for captions).
- Batch: auto `B = 4` for ≤4B fp16, 2 for nf4. Left-padded batched `generate`. P6 measures images/min.
- Caption budgets (≈1.3 tokens/word):
  - SDXL CLIP 75 tokens/chunk, so ≤ 55 words keeps one chunk (kohya `max_token_length 225` allows 3);
  - Z TE max 512 tokens, so ≤ 350 words (style target 80 to 150).
  - Review flags anything over budget.
- Trigger first plus `keep_tokens 1`, so shuffling (tags only) never moves it.

**Prompt writer** (`engine/enhance.py`, worker op `enhance {text, mode, max_new}`):
- Builds a CausalLM view over the loaded TE with no new weights: `lm = Qwen3ForCausalLM(cfg)`; `lm.model = te`; `lm.lm_head.weight = te.embed_tokens.weight`. That tied weight is fp16; bnb quantizes Linear layers only, so the embedding stays fp16 (0.78 GB, already in VRAM).
- Chat: system = `pe.py` template (mode `official`; vendored with attribution) or none (mode `engineer`, as its card says). `enable_thinking=False`; sampling temp 0.7, top_p 0.8, top_k 20 (Qwen3 non-thinking defaults); `max_new 400`. Then strip `<think>`/quotes.
- The result goes into the prompt box (undo kept). The embed cache key changes with the text, so nothing stale is reused.
- The abliterated TE gives an uncensored writer for free.
- Speed is est 15 to 25 tok/s nf4 on T4 (P6 measures). If the DiT is not loaded the op still works: the TE loads alone.

**Custom Z-Image checkpoint** (safetensors bf16/fp16/fp8 from Civitai/HF/Drive):
- Load with `from_single_file(..., config=Z_REPO, subfolder="transformer", quantization_config=BnB nf4, compute fp16)`, then `zimage_fp16.patch`.
- Size: 6.15e9 params × 0.5 B + absmax (1 fp32 per 64 → 0.38 GB, double-quant less) ≈ **3.4 GB VRAM** (matches unsloth's 3.3).
- Budget: DiT 3.4 + TE nf4 2.5 + VAE 0.2 + activations (1024², ~3) ≈ 9.1 of 15 GB.
- Cache: `save_pretrained` of the quantized DiT to `models/zimage/<stem>-nf4/`, so the next load is like unsloth (no re-quantize).
- RAM risk: a 12.3 GB bf16 file vs 12.7 GB RAM. Rely on mmap; the guard watches. P6 checks the peak, and if it is too high, convert per tensor to GGUF Q8 on CPU (`gguf` writer, streamed) instead.
- GGUF and `repo:subfolder` already work.

**Compare grid**: cells = |epochs| · |strengths| · |prompts|; time ≈ cells · t_img + load.
- Z Turbo 4-bit Speed t_img ≈ 36 s, so the 3·3·2 default = 18 cells ≈ 11 min. SDXL 20 steps ≈ 20 s gives ≈ 6 min.
- The form shows the estimate before running.

**Fooocus speed**: unmeasured on T4 (P6).

## Phases (each ends with tests + one commit series; T4 only in P6)
| P | Work | Done when |
|---|---|---|
| P0 | builder (multi-notebook, shared setup), bootstrap groups, nb.py, drive.py, plan.py + tests, tailnet hostname | `--check` passes; plan/drive unit tests pass |
| P1 | captioning: catalog, styles, batching, quant, mirror, review | unit tests for styles, budgets, quant pick; real run in P6 |
| P2 | train-test-sdxl: presets, auto R, v-pred, Drive out/resume, compare --epochs | dry-run prints configs; resume picks newest state |
| P3 | train-test-zimage: TE choice, save_precision fp16, custom DiT, Drive | same |
| P4 | sdxl: fooocus.py, config.txt, fetch in its venv, tailnet | config.txt unit test |
| P5 | zimage app trim, Drive dirs, custom checkpoint, enhance op + button | fake-engine browser pass |
| P6 | T4: setup times, captioner images/min, Fast vs Best training (1 run each), Fooocus gen time, enhance tok/s, custom ckpt RAM peak | numbers written to PLAN.md; presets updated |

Each phase removes the code it replaces. `README.md` + `HANDOVER.md` are updated at P6.
