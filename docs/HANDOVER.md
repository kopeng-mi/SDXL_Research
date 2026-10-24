# Handover

Read with `docs/PLAN.md` (decisions, T4 numbers) and `docs/OVERHAUL_PLAN.md` (the step4 plan). Terse on purpose.

## State
- Repo `kopeng-mi/SDXL_Research` (public, no personal data). Local: `C:\Users\raman\Documents\MyGit\temps\SDXL_Research`.
- Branch **`step4-overhaul`** (pushed). P0-P4 implemented and checked: unit tests, fake-engine browser passes (tester agent),
  two T4 runs (`tools/t4_check.py`) and a one-epoch training smoke run (`tools/train_check.py`). Numbers in `docs/PLAN.md`.
- `main` = init only. Merge after the final T4 run (no squash), then set `cells/00_setup.py` BRANCH to `main` and rebuild
  the notebook (`python tools/build_notebook.py`).
- Notebook = `cells/00_setup.py` + `cells/10_launch.py`. Launch runs `nb.serve(["-m","sdxlr.webui",...])` (restarts on crash).

## Code map (step4)
| File | Role |
|---|---|
| `paths.py` | dirs + model catalog (Z_MODELS incl. unsloth 4-bit DiT, Z_ENCODERS, CONTROLNETS, PREVIEW_VAES, z_key); stdlib only |
| `engine/client.py` | web-side proxy: spawns the worker; reader thread (beats any time, stale request events dropped); one op at a time; cancel watch per request; `kill(reason)`; idle death reported once |
| `engine/worker.py` | process main: ops load/unload/generate/upscale (+`hog` with `SDXLR_DEBUG_OPS=1`); edit tools area/extend/whole via crop-and-stitch; beats; previews; `UserError` = plain sentence |
| `engine/models.py` | torch Engine: SDXL (LoRAs fused by default) and Z-Image (GGUF or unsloth bnb-4bit DiT, official TE nf4/int8, prompt-embed cache, part reuse); inpaint_crop (plain / ControlNet lite), upscale (balanced tiles, tile model) |
| `engine/zcontrol.py` | ControlNet lite (union = better blend, tile) with zeroed context parts; config in `engine/zcn_lite/` |
| `engine/edit.py`, `keys.py`, `preview.py`, `fake.py` | pure mask/stitch ops; pure decisions (embed key, reuse plan, tiling, tile plan); TAEF1/TAESDXL previews; stub Engine for `SDXLR_FAKE_ENGINE=1` |
| `service/gen.py` | form -> engine requests; RAM check (stops another model, waits up to 15 s for freed RAM); `safe_update` for presets/"Make it again" |
| `service/guard.py` | RAM watchdog (acts at 1 GB free), health line, crash-log export, stale jobs |
| `service/jobs.py`, `files.py`, `state.py`, `masks.py` | job queue (process trees), cached indexes (history up to 5000, incremental), bindable session state, mask upload route |
| `ui/pages/*`, `ui/components/*` | pages; mask painter and before/after (plain Vue) |
| `loraformat.py`, `data.py`, `fetch.py` | header-only LoRA/checkpoint checks + sidecars; import with dedupe + HEIC; downloads to `.part` with plain errors |
| `tools/t4_check.py`, `tools/train_check.py` | the T4 test kits (see Testing) |

## Decisions taken during step4 (beyond OVERHAUL_PLAN)
- Desktop only (user): no phone layout work.
- Q8 stays the default Z-Image model (Q6/Q4 are slower on T4). unsloth bnb-4bit DiT: same speed and look, 6 GB VRAM; candidate default once LoRAs on it are checked.
- unsloth pre-quantized text encoder dropped (does not load through AutoModel).
- SDXL LoRA fusing on by default (`SDXLR_FUSE=0` = off).
- gradio pin kept (musubi-tuner depends on gradio); the Gradio app is gone.
- User picks for the open questions: no pytest (plain-script tests), pillow-heif yes, GGUF kernels no, ostris adapter yes (`dit: turbo, adapter: ostris`), browser tests by the tester agent.
- No face-similarity score (would need insightface); identity is judged from compare grids.

## Access
Tailscale only: `http://sdxlr:7860` (non-ephemeral key -> `sdxlr-N`). Test keys pasted in chat must be deleted after use.

## Open
1. Full-length training (15 / 10 epochs) not run yet; the smoke run covers every trainer for one epoch.
2. Taildrop inbox not built (needs the userspace test and "Send Files" on).
3. Model chip / model sheet redesign (P4) not done; the current model card was fine in the usability pass.
4. Stretch: soft-mask editing with the community differential img2img.
5. Colab idle disconnect is documented in the Launch cell, not measured.

## Testing workflow
- Unit tests (plain scripts): `uv run --no-project -q --with psutil --with pillow --with numpy [--with fastapi --with httpx --with nicegui==3.18.0] python -I tests/<name>.py` for test_edit, test_keys, test_loraformat, test_masks, test_data, test_edit_ops, test_client. `uvx pyflakes src/sdxlr tools tests` for undefined names.
- Fake UI: `SDXLR_WORK=<dir> SDXLR_FAKE_ENGINE=1 PYTHONPATH=src uv run --no-project --with nicegui==3.18.0 --with pyyaml --with psutil --with huggingface_hub --with pillow --with toml --with numpy python -m sdxlr.webui --port 7870 --access none`. Switches: `SDXLR_FAKE_CRASH=1`, `SDXLR_FAKE_SLOWLOAD=1`, `SDXLR_FAKE_SCALE=0.1`, `SDXLR_FAKE_FAST=1`. Browser checks: the tester agent (omp, built-in browser) or the `agent-browser` skill.
- T4: `colab new -s <name> --gpu T4`; clone + bootstrap + `tools/t4_check.py` / `tools/train_check.py` detached (`nohup`, logs in /content/logs), poll with short `colab exec` calls; `MSYS_NO_PATHCONV=1` for uploads; if the router's DNS refuses the VM host, run the colab CLI through a wrapper that resolves `*.colab.dev` via 1.1.1.1. Stop the VM when done; delete `dataset_test` photos from it.
- Gotchas: Git Bash heredocs turn `\n` into a newline in inline Python (use the Write tool or raw strings); never call the scoop `7z` shim (it recursed into 3000 processes once); port 8080 = Colab Jupyter.

## User preferences
Plan first; numbered questions in chat (`Q1. ... 1. (recommended)`), terse answers like `1:2`. Short plain replies, bold key terms, no filler. Ask before new packages, big deletes, git remote/history (pushing work branches is fine). Small conventional commits, no co-author. Nushell for commands. Free Colab: stop VMs when idle. Subagents only when asked.
