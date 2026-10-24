# Handover

Read with `docs/PLAN.md` (decisions, T4 numbers). Terse on purpose.

## State
- Repo `kopeng-mi/SDXL_Research` (public, no personal data). Local: `C:\Users\raman\Documents\MyGit\temps\SDXL_Research`.
- Branch **`step4-overhaul`**. **P0 (stability) done** — 8 commits from `refactor(paths)` to the thumb prewarm. Verified locally with the fake engine end to end (load, txt2img, area/whole/upscale tools with a 24 MP source capped to ~1.2 MP, cancel, worker crash + recovery, upload caps, GPU-job handoff) and a browser UI pass; `rg "import torch" src/sdxlr/ui src/sdxlr/service` is empty. **The real worker process has never run on a GPU** — first T4 session must cover: spawn, load, generate, cancel-during-load (kill path), guard trip, beat-driven sidebar. P1 (image editing) is next per `docs/OVERHAUL_PLAN.md`.
- `main` = init only; merge after full testing (no squash), then set `cells/00_setup.py` BRANCH back to `main`.
- Notebook = `cells/00_setup.py` + `cells/10_launch.py` → `python tools/build_notebook.py`. Launch runs `nb.serve(["-m","sdxlr.webui",...])` (auto-restart on crash).
- Gradio `app.py` is dead code (still in repo, gradio still pinned, imports updated to engine/models): remove in P5.

## Code map (step4 layout)
| File | Role |
|---|---|
| `paths.py` | dirs + model catalog (Z_REPO, Z_MODELS, Z_ENCODERS, SAMPLERS, z_key), stdlib only. Local runs (no /content) use ./work; `SDXLR_WORK` overrides |
| `engine/client.py` | web-side proxy: spawn worker (torch imports only in the child), one op at a time, stage/step events -> task, beats -> VRAM mirror, cancel (kill after 8 s), death -> plain sentence + restart on next call |
| `engine/worker.py` | process main: ops (load/unload/status/generate/upscale), heartbeats every 2 s (VRAM parts, rss), orphan exit, pidfile, oom_score 1000, work_copy caps (~1.2 MP), saves PNG + JSON sidecar |
| `engine/models.py` | the torch Engine (load/set_loras/generate/upscale) + stage timing; `engine/fake.py` = fake client (`SDXLR_FAKE_ENGINE=1`, `SDXLR_FAKE_CRASH=1` kills once) |
| `service/gen.py` | state -> engine requests (load_blocking, generate_blocking, reuse); RAM refusal via guard |
| `service/jobs.py` | JobManager (moved), gpu_lock, pidfiles + oom_score 800 per job, on_fail hook |
| `service/guard.py` | RAM watchdog (700 MB kill line: worker then GPU job), health line every 30 s, load refusal, crash-log export to Drive/sync repo, stale-job cleanup |
| `service/files.py` | cached listings (history+meta, prompts, LoRAs/models/datasets, caption texts, thumbs 160/320 prewarmed) refreshed every 2.5 s or on poke(); url/thumb live here |
| `service/state.py` | bindable-dataclass session state (dict-style access), JSON migration on load |
| `ui/core.py` | wiring + re-exports; `ui/frame.py` starts guard, files refresh, autosave; pages read only caches |
| rest (`webui, ui/*, train*, caption, compare, data, fetch, hfsync, tailnet, resources, activity, history, session, settings`) | unchanged roles; resources is torch-free (beats + NVML) |

## Access
Tailscale only. Open `http://sdxlr:7860`. Non-ephemeral key → each new VM is `sdxlr-N`. **The test key in chat history must be deleted** (user said they will).

## Measured on T4 (step3 UI, before step4)
First Z-Image Q8 load+gen 5m08s. Speed 38 s, Quality 49 s, area change 37 s. Qwen captions 5m39s first run. VRAM Q8 run: DiT 6.7, TE nf4 2.4, VAE 0.2 GB. Step4 numbers to re-measure on the first T4 session.

## Open (from step3, still relevant)
1. ~~OOM / VM freeze~~ addressed by P0 (worker process, caps, guard) — needs the T4 proof run.
2. Rethink img2img UX (P1: brush mask + crop-and-stitch, one flow).
3. Rethink UI generally (P4).
4. Optimization list = P2 (TE unload, VAE tiling, previews, unsloth TE, benchmarks).
5. Still unproven: Z-Image identity LoRA quality, auto resume (needs HF token), 768 px Z training.

## Testing workflow
- Local fake UI: `SDXLR_WORK=work SDXLR_FAKE_ENGINE=1 PYTHONPATH=src uv run --no-project --with nicegui==3.18.0 --with pyyaml --with psutil --with huggingface_hub --with pillow --with toml --with numpy python -m sdxlr.webui --port 7870 --access none` (fake engine never downloads; needs_prefetch=False gates prefetch + the RAM refusal). Browser checks: Z Code's built-in browser on 127.0.0.1:7870.
- Colab: `colab new -s t4 --gpu T4`; upload key to `/content/.tskey`; `colab exec -f colab_tests\_ts_start.py` (clones branch, bootstrap, webui); poll `_ts_chk.py`; `PYTHONIOENCODING=utf-8` for exec. `colab_tests\` is **not in the repo** — it lived on the old VM; rebuild the two small scripts (ts_start: clone+bootstrap+launch; ts_chk: curl the port) or run the cells.
- Gotchas: Git Bash heredocs turn `\n` into a real newline (write edit scripts with the Write tool); `pkill -f name` matches its own shell (use `name[x]`); port 8080 = Colab Jupyter, never expose; router DNS sometimes refuses new VM hosts.
- Leftover: empty `C:\content` folder (user: `rm C:\content`).

## User preferences
Plan first; numbered questions in chat (`Q1. ... 1. (recommended)`), terse answers like `36:2`. Short plain replies, bold key terms, no filler. Ask before new packages, big deletes, git remote/history (pushing work branches is fine). Small conventional commits, no co-author. Nushell for commands. Free Colab: stop VMs when idle. Subagents only when asked (Haiku).

