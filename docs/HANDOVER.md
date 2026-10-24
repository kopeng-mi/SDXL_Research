# Handover

Snapshot for picking the work up later. Read with `docs/PLAN.md` (decisions,
measured T4 numbers, bugs fixed, UI design).

## Where things are

| Item | Location |
|---|---|
| Repo | `kopeng-mi/SDXL_Research` (public; code only, no personal data) |
| Work branch | `step2-polish` (on top of `step1-scaffold`, pushed). `main` = first commit only; merge after the user's Colab web test |
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
| `resources.py` | header bars (VRAM/RAM/disk by named block), load estimate, clear download cache |
| `session.py` | 4-digit session codes, UI state in `/content/work/sessions/` |
| `hfsync.py` | auto resume: dataset + latest LoRA/state in the private HF repo, older checkpoint removed |

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

1. User tests from Colab web: Z-Image identity run on their photos with auto
   resume (HF_TOKEN in Colab Secrets). Auto resume and HF dataset sync are
   untested (no token on the agent's VM).
2. 768 px Z-Image training untested (estimate only).
3. Optional: Setup "prefetch" of usual models.
4. Merge `step2-polish` into `main` (no squash) after that test; setup cell
   default branch is `main`.

## Testing workflow (agent notes)

- Colab CLI: `colab new -s <name> [--gpu T4]`, `colab upload/download`,
  `colab console -s <name>`, `colab stop`. Free account: T4 ~5 h, use the CPU VM
  for anything that does not need a GPU, stop sessions when done.
- Your router's DNS sometimes refuses new VM hostnames ("getaddrinfo failed"):
  stop the session and create a new one. Run `colab exec` with
  `PYTHONIOENCODING=utf-8` or it crashes printing progress bars.
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
