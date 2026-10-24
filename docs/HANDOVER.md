# Handover

Read with `docs/PLAN.md` (decisions, T4 numbers). Terse on purpose.

## State
- Repo `kopeng-mi/SDXL_Research` (public, no personal data). Local: `C:\Users\raman\Documents\MyGit\temps\SDXL_Research`.
- Branch **`step3-nicegui`** (pushed, last `121a242`). Built on `step2-polish` → `step1-scaffold`. `main` = init only. **Merge pending** (user: after testing; no squash; then set `cells/00_setup.py` BRANCH back to `main`).
- Notebook = `cells/00_setup.py` + `cells/10_launch.py` → `python tools/build_notebook.py`. Launch runs `nb.serve(["-m","sdxlr.webui",...])` (auto-restart on crash).
- Gradio `app.py` is dead code (still in repo, gradio still pinned): remove when NiceGUI is final.

## Code map (new parts)
| File | Role |
|---|---|
| `webui.py` | shim → `ui/frame.main` |
| `ui/core.py` | state (`DEFAULTS`, `STATES` per 4-digit session, autosave), engine/jobs singletons, checks (`check_*`, raise ValueError with user text), `generate_blocking` (tools: None/area/whole/upscale), `submit_*` jobs, `save_upload`, `shred`, `job_error`, usage + disk-rate sampler threads |
| `ui/theme.py` | CSS (darkroom palette, IBM Plex Sans, no motion), `labeled`, `seg`, `main_button`, `button`, `confirm`, `JobLine` |
| `ui/frame.py` | sidebar (nav, model, VRAM/RAM/disk bars, current task + Cancel, session menu), failure/done toasts, startup, Tailscale |
| `ui/pages/*.py` | generate, dataset, train, compare, library, history, activity |
| `engine.py` | + status/step callbacks, img2img (Z + SDXL), `upscale()` tiled 1024/128 overlap, `empty_cache` after each image |
| `activity.py` | in-process Task (stage, steps, cancel, child proc), disk-rate download watcher, logging to `WORK/logs/app.log` + stdout |
| `jobs.py` | + per-job logs `logs/jobs/<id>.log`, throttled log lines |
| `fetch.py` | + `prefetch-zimage/-sdxl` (child process so Cancel kills downloads), HF blob links, uploads, `zimage_custom` |
| `hfsync.py` | auto resume: HF `NAME/ckpt/` (latest LoRA+state only) + `NAME/dataset/` |
| `tailnet.py` | tailscaled userspace, `TS_AUTHKEY`, hostname `sdxlr`, logout on shutdown |
| `history.py` / `session.py` / `resources.py` | image JSON sidecars; session codes (`latest()` auto-continue); usage blocks (NVML) |

## Access
Tailscale only (Cloudflare removed: kept breaking). Open `http://sdxlr:7860`. Non-ephemeral key → each new VM is `sdxlr-N` (Colab kill skips logout). User will make an **ephemeral** key. **The test key in chat history must be deleted** (user said they will).

## Measured on T4 (new UI)
First Z-Image Q8 load+gen 5m08s (cancel mid-download works, resumes). Speed 38 s, Quality 49 s, area change 37 s. Qwen captions 5m39s first run (mostly 8.8 GB download). VRAM Q8 run: DiT 6.7, TE nf4 2.4, VAE 0.2, cache 5.1 GB (now freed after each image).

## Next: `docs/OVERHAUL_PLAN.md`
Decided plan for step4 (engine worker process, brush-mask editing with crop-and-stitch, speed, LoRA checks, UI rework), with research, edge cases and the T4 test plan. Test photos in `dataset_test/` (git-ignored, personal). The list below is the background it came from.

## Open (user wants to rethink these next)
1. **OOM / VM froze** during first img2img test on T4 (log lost). Suspects: uploads not downscaled (cap ~1.2 MP for area/whole), `fetch prefetch` imports `engine` → torch (+1-2 GB RAM; move constants to a light module), no free-RAM guard. Add RAM/VRAM log every 30 s and copy logs off-VM on failure.
2. **Rethink img2img UX**: prompt + masking (brush, not only boxes), whole/area/upscale in one clear flow. Untested on real GPU: whole-image, upscale, SDXL img2img.
3. **Rethink UI** generally (user found it clunky; two Haiku user tests done, most findings fixed).
4. Optimization: optional TE unload after encode (2.4 GB), VAE tiling for peak, smaller GGUF.
5. Small: stage text during download, image swaps late over slow link, dataset tile heights, Haiku leftovers (menu a11y, import count 0/6 vs 5, Activity "Done" row keeps download text).
6. Still unproven: Z-Image identity LoRA quality (needs user's real run), auto resume (needs HF token), 768 px Z training.

## Testing workflow
- Local UI without GPU: `colab_tests\dev_ui.py` (fakes engine, load, jobs; work dir `colab_tests\dev_work`, port 7870). Run: `uv run --no-project --with nicegui==3.18.0 --with pyyaml --with psutil --with huggingface_hub --with pillow --with toml --with numpy python dev_ui.py`. Real jobs are faked (they once downloaded 0.9 GB to `C:\content`).
- Colab: `colab new -s t4 --gpu T4`; upload key to `/content/.tskey`; `colab exec -f colab_tests\_ts_start.py` (clones branch, bootstrap, webui); poll `_ts_chk.py`; `PYTHONIOENCODING=utf-8` for exec. Browser checks: the agent's built-in browser on `http://sdxlr[-N]:7860` (this PC is on the tailnet).
- Gotchas: Git Bash heredocs turn `\\n` into a real newline (write edit scripts with the Write tool); `pkill -f name` matches its own shell (use `name[x]`); port 8080 = Colab Jupyter, never expose; router DNS sometimes refuses new VM hosts (new session fixes).
- Leftover: empty `C:\content` folder (tool blocks deleting it; user: `rm C:\content`).

## User preferences
Plan first; numbered questions in chat (`Q1. ... 1. (recommended)`), terse answers like `36:2`. Short plain replies, bold key terms, no filler. Ask before new packages, big deletes, git remote/history (pushing work branches is fine). Small conventional commits, no co-author. Nushell for commands. Free Colab: stop VMs when idle. Subagents only when asked (Haiku).
