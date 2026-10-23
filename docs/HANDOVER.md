# Handover

Snapshot for picking the work up later. Read with `docs/PLAN.md` (decisions,
measured T4 numbers, bugs fixed, UI design).

## Where things are

| Item | Location |
|---|---|
| Repo | `kopeng-mi/SDXL_Research` (public; code only, no personal data) |
| Work branch | `step1-scaffold` (all work, pushed). `main` = first commit only; merge after the T4 UI pass |
| Local clone | `C:\Users\raman\Documents\MyGit\temps\SDXL_Research` |
| Old reference notebooks | same folder, git-ignored (personal data), kept locally |
| Test scripts (not in repo) | `C:\Users\raman\Documents\MyGit\temps\colab_tests\` |
| Reference clones/sources | `C:\Users\raman\Documents\MyGit\temps\refs\` (old `lora_backend_sdxl`, diffusers/musubi source snippets) |
| Submodules | `vendor/sd-scripts` (kohya v0.12.2), `vendor/musubi-tuner` (v0.3.6) |

## Code map (`src/sdxlr/`)

| Module | Role |
|---|---|
| `bootstrap.py` | stdlib-only; uv-managed Python 3.12 venv at `/content/venv`; installs kohya reqs, musubi, this package with `uv pip install`; idempotent stamp |
| `nb.py` | kernel-side helpers: stream child output into cells, export `HF_TOKEN`/`CIVITAI_TOKEN` from Colab Secrets |
| `app.py` | Gradio 6 UI, 5 tabs + Jobs; GPU lock; auto-refresh on job finish |
| `jobs.py` | job queue: one GPU job at a time, CPU jobs alongside; progress/ETA/cancel; unload engine before GPU jobs |
| `engine.py` | SDXL + Z-Image generation; LoRA adapter cache; prompt cache; OOM retry; inpaint; T4 fixes |
| `prompt.py` | A1111 weights, >77-token prompts, clip skip (A1111 numbering) for SDXL |
| `zimage_fp16.py` | hooks that stop Z-Image fp16 overflow (black images) on T4 |
| `caption.py` | Qwen3-VL 4B/2B, JoyCaption (nf4); identity prompt; trigger + class word |
| `train.py` / `train_zimage.py` | kohya SDXL / musubi Z-Image LoRA training (+ diffusers conversion) |
| `compare.py` | LoRA x strength grid, mixes, resume-safe |
| `data.py` / `fetch.py` | dataset import (zip/images/URL/HF/Drive, captions optional); model/LoRA downloads incl. Civitai |
| `presets.py` / `settings.py` / `modes.py` | JSON presets + HF sync; UI-set repo names; modes, aspects, styles, training ETA |

Notebook = `cells/00_setup.py` + `cells/10_launch.py`, built into `notebook.ipynb`
by `python tools/build_notebook.py` (edit cells, never the .ipynb).

## Status

Working and tested on T4: venv, captions (Qwen3-VL 4B), SDXL training,
SDXL generation + LoRAs + inpaint, compare grid, Z-Image Turbo generation
(fp16 fix), Z-Image LoRA training at 512 px, LoRA conversion + loading on Turbo,
Z-Image inpaint.

Tested on CPU VM through a browser: new UI renders; dataset import job with
auto-refresh; caption editor; Library download; preset save/load.

## Open items (in order)

1. **Pending question to the user (Q17):** run a T4 pass of the UI's GPU paths
   now (Load both model types, generate in Quality/Speed/Fast, inpaint, short
   train + caption + compare from the UI, measure SDXL Fast mode), or let the
   user test from Colab web.
2. SDXL Fast mode uses the SDXL-Lightning 8-step LoRA: speed/quality unmeasured;
   may look worse on Pony/fine-tunes.
3. Z-Image identity learning unproven: the 30-step test LoRA was too weak to
   change output. Needs a real run on the user's dataset (512 px, ~5 epochs x
   2 repeats fits one session; ETA shown in the Train tab).
4. Z-Image training has no resume across sessions yet (SDXL has HF resume).
5. 768 px Z-Image training untested (estimate only).
6. More captioners (e.g. WD tagger) were requested ("many captioner support").
7. Optional: Setup "prefetch" of usual models to save time each session.
8. Merge `step1-scaffold` into `main` (no squash) after the T4 pass; setup cell
   default branch is `main`.

## Testing workflow (agent notes)

- Colab CLI: `colab new -s <name> [--gpu T4]`, `colab upload/download`,
  `colab console -s <name>`, `colab stop`. Free account: T4 ~5 h, use the CPU VM
  for anything that does not need a GPU, stop sessions when done.
- `colab exec` drops long cells (10 s read timeout): start long jobs detached on
  the VM (`{ nohup bash -c "..." > /content/logs/x.log 2>&1 & }`) and poll.
- Drive the VM through `colab console` in the Herdr side pane (`wK:pE`). That
  pane's local shell is **nushell**: check for the VM prompt (`/content#`) before
  sending bash. `colab_tests/vm.sh` has a guarded `vm 'cmd' MARK` helper.
- Don't scrape the console screen for results: write them to a file on the VM
  and `colab download` it (screen redraws and echoed commands cause false matches).
- Git Bash rewrites `/content/...`: prefix `MSYS_NO_PATHCONV=1` for colab upload/download.
- UI checks: run the app with `--share` on the VM, then `agent-browser` (named
  session) to open the gradio.live link, click, and screenshot.

## User preferences (from their global instructions)

- Plan first for non-trivial work; ask numbered questions in chat (not the
  question tool), mark the recommended option; terse replies like `1:1 2:2`.
- Replies: answer first, short, plain words, bold key terms, no filler.
- Ask before installing packages, deleting big things, or git history/remote
  changes beyond what was approved (pushing the work branch is approved).
- Conventional commits, small and scoped; no AI co-author line.
- Supporting commands/scripts for the user go in chat in **nushell**.
