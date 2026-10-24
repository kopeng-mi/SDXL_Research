"""RAM/disk watchdog and crash-log export: keep the Colab VM alive.

Free-tier Colab kills the whole session when RAM runs out, so this guard acts
first: under KILL_MB of available RAM it stops the engine worker (rebuilt for
free on the next Generate); if RAM is still short, it stops a running GPU job
too (auto resume keeps training progress). Loads are refused up front when
the model cannot fit in free RAM, with a message naming what holds it.

Also: a health line in the log every 30 s, oom_score_adj so Linux prefers
killing the worker over the web app, job pidfiles cleaned up at start, and
logs copied to Drive / the sync repo when something dies.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

from .. import resources, settings
from ..activity import log
from ..paths import LOGS, RUN, WORK
from ..engine.worker import oom_score

KILL_MB = 700  # MemAvailable under this: act before Colab kills the session (tune on T4)
HEALTH_EVERY = 30.0
RAM_OVERHEAD_GB = 1.6  # python + torch runtime + load-time working memory
GB = 2**30

EVENTS: list[str] = []  # notes for the UI to toast; drained by the frame's tick


def note(msg: str) -> None:
    EVENTS.append(msg)
    log.info(f"guard: {msg}")


def mem_available_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024
    except OSError:
        pass
    import psutil
    return psutil.virtual_memory().available / 2**20


# ---- load refusal ----
def ram_need_gb(kind: str, sdxl_path: str | None, z_transformer: str | None, te_quant: str) -> float:
    from ..paths import TE_GB, Z_MODELS

    if kind == "Z-Image":
        p = None
        if z_transformer:
            name = Path(z_transformer).name
            for d in (WORK / "models" / "zimage", WORK / "models" / "zimage_custom"):
                if (d / name).exists():
                    p = d / name
                    break
        if p is not None:
            size = p.stat().st_size / GB
        else:  # not downloaded yet: the known sizes of the built-in GGUFs
            size = 7.2 if "Q8" in z_transformer else 6.6 if "Q6" in z_transformer else \
                5.6 if "Q5" in z_transformer else 5.0 if "Q4" in z_transformer else 7.2
        return size + TE_GB.get(te_quant, 2.5) + RAM_OVERHEAD_GB
    size = Path(sdxl_path).stat().st_size / GB if sdxl_path and Path(sdxl_path).exists() else resources.SDXL_GB
    return size + RAM_OVERHEAD_GB


def check_ram(kind: str, sdxl_path: str | None, z_transformer: str | None, te_quant: str, beat: dict, jm) -> None:
    """Raise ValueError when the picked model cannot load into free RAM."""
    need = ram_need_gb(kind, sdxl_path, z_transformer, te_quant)
    avail = mem_available_mb() / 1024
    if need <= avail:
        return
    holders = ", ".join(f"{n} ({g:.1f} GB)" for n, g in sorted(resources.usage(beat, jm).get("RAM", (0, []))[1],
                                                              key=lambda b: -b[1])[:2] if g > 0.3)
    what = "Z-Image" if kind == "Z-Image" else Path(sdxl_path or "").name
    raise ValueError(f"Loading {what} needs about {need:.0f} GB of memory, but only {avail:.0f} GB is free"
                     + (f" (held by {holders})" if holders else "")
                     + ". Stop the running job first, pick a smaller model, or wait a moment and try again.")


# ---- watchdog ----
def _kill_jobs_if_low(jm, mb: float) -> None:
    for j in list(jm.jobs.values()):
        if j.status == "running" and j.gpu and j.proc and j.proc.poll() is None:
            note(f"still only {mb:.0f} MB free: stopping the job \"{j.title}\" (auto resume keeps its progress)")
            j.proc.terminate()
            return


def _watch(engc, jm) -> None:
    tripped = 0.0
    while True:
        time.sleep(1.0)
        try:
            mb = mem_available_mb()
        except Exception:
            continue
        if mb < KILL_MB and time.time() - tripped > 30:
            tripped = time.time()
            note(f"low RAM ({mb:.0f} MB free): stopping the image engine so the session survives")
            try:
                engc.kill()
            except Exception as e:
                log.info(f"guard: killing the engine failed: {e}")
            if mem_available_mb() < KILL_MB:
                _kill_jobs_if_low(jm, mem_available_mb())


def _health(engc) -> None:
    last = 0.0
    while True:
        time.sleep(5.0)
        if time.time() - last < HEALTH_EVERY:
            continue
        last = time.time()
        try:
            beat = engc.beat or {}
            d = shutil.disk_usage(str(WORK.resolve().anchor))
            log.info(f"health: RAM free {mem_available_mb() / 1024:.1f} GB | engine worker rss {beat.get('rss', 0):.1f} GB, "
                     f"VRAM alloc {beat.get('alloc', 0):.1f} GB | disk free {d.free / GB:.0f} GB")
        except Exception as e:
            log.info(f"health line failed: {e}")


# ---- pidfiles for leftover jobs ----
def _kill_stale_jobs() -> None:
    import os
    import psutil
    d = RUN / "jobs"
    if not d.exists():
        return
    for f in list(d.glob("*.pid")):
        try:
            pid = int(f.read_text().strip())
            p = psutil.Process(pid)
            alive = "python" in p.name().lower() and any("sdxlr" in str(c) for c in p.cmdline())
        except (OSError, ValueError):
            f.unlink(missing_ok=True)
            continue
        except psutil.Error:
            alive = False
        if alive:
            try:
                p.kill()
                note(f"stopped a leftover job from an earlier app run (pid {pid})")
            except psutil.Error:
                pass
        f.unlink(missing_ok=True)


# ---- crash-log export ----
def export_logs(reason: str) -> None:
    """Copy the logs off the VM when something dies: Drive if mounted, sync repo if set."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    drive = Path("/content/drive/MyDrive/sdxlr_logs")
    if drive.parent.exists():
        try:
            dest = drive / stamp
            shutil.copytree(LOGS, dest, dirs_exist_ok=True)
            log.info(f"guard: logs copied to Drive ({reason}): {dest}")
        except Exception as e:
            log.info(f"guard: Drive log copy failed: {e}")
    repo = settings.get()["sync_repo"]
    if repo:
        def push():
            try:
                import zipfile
                from huggingface_hub import HfApi
                out = WORK / f"sdxlr_logs_{stamp}.zip"
                with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in LOGS.rglob("*.log"):
                        z.write(f, f.relative_to(LOGS))
                HfApi().upload_file(path_or_fileobj=str(out), path_in_repo=f"logs/{stamp}.zip", repo_id=repo)
                out.unlink(missing_ok=True)
                log.info(f"guard: logs uploaded to {repo} ({reason})")
            except Exception as e:
                log.info(f"guard: log upload to {repo} failed: {e}")

        threading.Thread(target=push, daemon=True).start()


def start(engc, jm) -> None:
    """Wire the guard into the running app (called once from the frame at startup)."""
    RUN.mkdir(parents=True, exist_ok=True)
    oom_score(-500)  # the web app should be the last thing Linux kills (root on Colab)
    _kill_stale_jobs()
    engc.on_death = lambda: export_logs("engine worker died")
    jm.on_fail = lambda job: export_logs(f"job {job.id} failed")
    threading.Thread(target=_watch, args=(engc, jm), daemon=True).start()
    threading.Thread(target=_health, args=(engc,), daemon=True).start()
