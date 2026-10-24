# Handover

Read with `docs/PLAN.md` (decisions, T4 numbers). Terse on purpose.

## State
- Repo `kopeng-mi/SDXL_Research` (public, no personal data). Local: `C:\Users\raman\Documents\MyGit\temps\SDXL_Research`.
- Branch **`step4-overhaul`**. **P0 (stability) done** and reviewed; batches A+B of review fixes are in (client reader thread, guard reasons, process-tree stop, upload/thumb fixes, serving limits, fake = stub engine in the real worker). `rg "import torch" src/sdxlr/ui src/sdxlr/service` is empty; `tests/*.py` pass. **The real worker process has never run on a GPU** — first T4 session must cover: spawn, load, generate, cancel-during-load (kill path), guard trip, beat-driven sidebar, and the Linux-only items under Open. **P1 next: wire 712bc4a's edit ops (`engine/edit.py`, worker `area`/`extend`/`whole`) into the UI** (brush mask component, modes), per `docs/OVERHAUL_PLAN.md`.
- `main` = init only; merge after full testing (no squash), then set `cells/00_setup.py` BRANCH back to `main`.
- Notebook = `cells/00_setup.py` + `cells/10_launch.py` → `python tools/build_notebook.py`. Launch runs `nb.serve(["-m","sdxlr.webui",...])` (auto-restart on crash).
- Gradio `app.py` is dead code (still in repo, gradio still pinned, imports updated to engine/models): remove in P5.

## Code map (step4 layout)
| File | Role |
|---|---|
| `paths.py` | dirs + model catalog (Z_REPO, Z_MODELS, Z_ENCODERS, SAMPLERS, z_key), stdlib only. Local runs (no /content) use ./work; `SDXLR_WORK` overrides |
| `engine/client.py` | web-side proxy: spawns the worker (torch imports only in the child); a **reader thread** drains the pipe (beats update state any time, idle too; events of other request ids are dropped); one op at a time; stage/step -> task; cancel watch kills only the request it was armed for; `kill(reason)` (guard) vs user cancel; a worker that dies idle is reported by the next call, once; stale worker found by pidfile `<pid> <create time>` |
| `engine/worker.py` | process main: ops (load/unload/status/generate/upscale; generate also does the `area`/`extend`/`whole` edit tools), beats every `SDXLR_BEAT_SECS` (2 s; state stamped under the send lock, errors logged once, loop never dies), every result carries `state`, orphan exit, pidfile, oom_score 1000, work_copy caps (~1.2 MP), saves PNG + JSON sidecar (+ mask for edits) |
| `engine/models.py` | the torch Engine (load/set_loras/generate/inpaint_crop/upscale) + stage timing |
| `engine/fake.py` | **stub Engine** with the same surface; `SDXLR_FAKE_ENGINE=1` makes the real worker run it, so the fake UI exercises the real pipe, reader, cancel kill and death message. `client.make()` always returns the real `EngineClient` |
| `engine/edit.py` | pure PIL/numpy mask, crop-and-stitch, outpaint canvas (unit-tested, no torch) |
| `service/gen.py` | state -> engine requests (load_blocking, generate_blocking, reuse); RAM refusal via guard (stops another loaded model only if the check fails) |
| `service/jobs.py` | JobManager, gpu_lock, jobs in their own session, `stop_tree` (whole tree), pidfiles + oom_score 800 per job, on_fail hook |
| `service/guard.py` | RAM watchdog (700 MB kill line: `kill(reason="low RAM")` the worker, then the GPU job tree), health line every 30 s, load refusal, incremental crash-log export to Drive/sync repo, stale-job cleanup, toasts as (seq, msg) per tab |
| `service/files.py` | cached listings (history+meta, prompts, LoRAs/models/datasets, captions, sizes, compare grids); the refresh thread rescans when directory mtimes change (or on poke(), or every 20 s) and makes all thumbs; `thumb()` never decodes on the event loop; `SERVED` = the only dirs the browser can fetch (`/work/outputs|thumbs|compare`) |
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
6. **Linux-only paths, check on the first T4 session**: jobs start with `start_new_session` and `stop_tree` also sweeps the process group; `oom_score_adj` (worker 1000, jobs 800, web -500 as root); `_kill_stale` by create time (restart the web app while a worker runs); guard kill on a RAM hog (synthetic allocation must stop the worker, not the session).
7. P0 leftovers, move to **P4**: RAM check only on load (not generate/upscale); the "GPU is busy" message has no job ETA; Library note "a recycled VM keeps nothing but HF sync / Drive"; measure and document Colab idle disconnect (keep the tab open).

## Testing workflow
- Unit tests (plain scripts, no pytest): `uv run --no-project -q --with numpy --with pillow python -I tests/test_edit.py`; `uv run --no-project -q --with psutil --with pillow --with numpy python -I tests/test_client.py` (real client + real worker + stub engine, about 1 min).
- Local fake UI: `SDXLR_WORK=work SDXLR_FAKE_ENGINE=1 PYTHONPATH=src uv run --no-project --with nicegui==3.18.0 --with pyyaml --with psutil --with huggingface_hub --with pillow --with toml --with numpy python -m sdxlr.webui --port 7870 --access none` (`needs_prefetch` is False under the stub: no downloads, no RAM refusal). Switches: `SDXLR_FAKE_CRASH=1` (first generate per work dir exits with code 9: death message, next Generate works), `SDXLR_FAKE_SLOWLOAD=1` (first load sleeps 20 s ignoring cancel: tests the 8 s cancel kill), `SDXLR_FAKE_SCALE=0.1` / `SDXLR_FAKE_FAST=1` (shorter / no sleeps). Browser checks: the `agent-browser` skill (the agent drives the fake UI on 127.0.0.1:7870, or the Colab app over the tailnet). Put the work dir outside the repo for throwaway runs (`./work` is git-ignored too).
- Colab: `colab new -s t4 --gpu T4`; upload key to `/content/.tskey`; `colab exec -f colab_tests\_ts_start.py` (clones branch, bootstrap, webui); poll `_ts_chk.py`; `PYTHONIOENCODING=utf-8` for exec. `colab_tests\` is **not in the repo** — it lived on the old VM; rebuild the two small scripts (ts_start: clone+bootstrap+launch; ts_chk: curl the port) or run the cells.
- Gotchas: Git Bash heredocs turn `\n` into a real newline (write edit scripts with the Write tool); `pkill -f name` matches its own shell (use `name[x]`); port 8080 = Colab Jupyter, never expose; router DNS sometimes refuses new VM hosts.
- Leftover: empty `C:\content` folder (user: `rm C:\content`).

## User preferences
Plan first; numbered questions in chat (`Q1. ... 1. (recommended)`), terse answers like `36:2`. Short plain replies, bold key terms, no filler. Ask before new packages, big deletes, git remote/history (pushing work branches is fine). Small conventional commits, no co-author. Nushell for commands. Free Colab: stop VMs when idle. Subagents only when asked (Haiku).

