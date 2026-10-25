"""Google Drive layout and sync for the notebooks. Stdlib only (kernel and venv both import it).

Drive FUSE is slow for many small reads, so jobs work on local copies:
`pull` copies what changed, `mirror` copies each output right after it is written
(a disconnect loses at most one item), and `StatePusher` keeps only the newest
training state on Drive, written as `.tmp` then renamed so a half copy is never used.
"""

from __future__ import annotations

import re
import shutil
import threading
import time
from pathlib import Path

MOUNT = "/content/drive"
ROOT = Path(f"{MOUNT}/MyDrive/SDXLR")
PUSH_EVERY = 120


def layout(root: str | Path = ROOT) -> dict[str, Path]:
    r = Path(root)
    return {"datasets": r / "datasets", "loras_sdxl": r / "loras" / "sdxl", "loras_zimage": r / "loras" / "zimage",
            "state": r / "state", "compare": r / "compare", "models_sdxl": r / "models" / "sdxl",
            "models_zimage": r / "models" / "zimage", "outputs": r / "outputs"}


def mount(root: str | Path = ROOT) -> Path:
    """Mount Drive in the Colab kernel and create the folders."""
    from google.colab import drive  # type: ignore
    if not Path(MOUNT, "MyDrive").exists():
        drive.mount(MOUNT)
    for p in layout(root).values():
        p.mkdir(parents=True, exist_ok=True)
    return Path(root)


def flush_and_unmount() -> None:
    """Colab writes Drive back lazily: call this before the VM goes away."""
    try:
        from google.colab import drive  # type: ignore
        drive.flush_and_unmount()
        print("drive: flushed")
    except Exception as e:
        print(f"drive: flush failed ({type(e).__name__}: {e})")


def _same(a: Path, b: Path) -> bool:
    try:
        sa, sb = a.stat(), b.stat()
    except OSError:
        return False
    return sa.st_size == sb.st_size and sa.st_mtime_ns == sb.st_mtime_ns


def pull(src: str | Path, dest: str | Path) -> int:
    """Copy files from src to dest when (size, mtime_ns) differ; returns how many were copied."""
    src, dest, n = Path(src), Path(dest), 0
    for f in src.rglob("*"):
        if f.is_file():
            t = dest / f.relative_to(src)
            if not _same(f, t):
                t.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, t)
                n += 1
    return n


def mirror(local: str | Path, drive_dir: str | Path) -> Path | None:
    """Copy one finished file to Drive (skips unchanged); never raises, Drive hiccups only print."""
    local, t = Path(local), Path(drive_dir) / Path(local).name
    try:
        if not _same(local, t):
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, t)
        return t
    except OSError as e:
        print(f"drive: could not copy {local.name} ({e})", flush=True)
        return None


def done(state: str, name: str) -> int:
    """Epochs finished according to a state dir name, -1 if it isn't this run's."""
    m = re.fullmatch(rf"{re.escape(name)}(?:_r(\d+))?-(\d+)-state", state)
    return int(m[1] or 0) + int(m[2]) if m else -1


def newest_state(folder: str | Path, name: str) -> Path | None:
    """The state dir of run `name` with the most finished epochs (ignores .tmp copies)."""
    f = Path(folder)
    dirs = [d for d in f.iterdir() if d.is_dir() and done(d.name, name) >= 0] if f.exists() else []
    return max(dirs, key=lambda d: done(d.name, name), default=None)


def push_state(local_out: str | Path, drive_state: str | Path, name: str) -> Path | None:
    """Copy the newest local state to Drive atomically and delete the older ones there."""
    src = newest_state(local_out, name)
    if src is None:
        return None
    dst_dir = Path(drive_state)
    final = dst_dir / src.name
    if final.exists():
        return final
    tmp = dst_dir / (src.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    tmp.rename(final)
    for d in dst_dir.iterdir():
        if d != final and d.is_dir() and (done(d.name, name) >= 0 or d.name.endswith(".tmp")):
            shutil.rmtree(d, ignore_errors=True)
    return final


def resume(drive_state: str | Path, local_out: str | Path, name: str) -> tuple[Path | None, int]:
    """(local copy of the newest Drive state, epochs done); (None, 0) when there is none."""
    src = newest_state(drive_state, name)
    if src is None:
        return None, 0
    dst = Path(local_out) / src.name
    pull(src, dst)
    return dst, done(src.name, name)


def mirror_new(local_out: str | Path, drive_dir: str | Path, pattern: str = "*.safetensors", settle: float = 10) -> int:
    """Copy finished files (unchanged for `settle` s) that Drive lacks; returns how many."""
    n, now = 0, time.time()
    for f in sorted(Path(local_out).glob(pattern)):
        if now - f.stat().st_mtime >= settle and not _same(f, Path(drive_dir) / f.name):
            n += mirror(f, drive_dir) is not None
    return n


class Syncer(threading.Thread):
    """Background while training: every PUSH_EVERY s copy new epoch files to `drive_out` and the
    newest state to `drive_state`; `stop()` does a final pass."""

    def __init__(self, local_out: str | Path, name: str, drive_out: str | Path | None = None,
                 drive_state: str | Path | None = None, pattern: str = "*.safetensors", every: float = PUSH_EVERY):
        super().__init__(daemon=True)
        self.local, self.name, self.out, self.state, self.pattern = Path(local_out), name, drive_out, drive_state, pattern
        self.every, self._stop_ev = every, threading.Event()

    def _pass(self, settle: float) -> None:
        try:
            if self.out and (n := mirror_new(self.local, self.out, self.pattern, settle)):
                print(f"drive: {n} LoRA file(s) saved", flush=True)
            if self.state:
                Path(self.state).mkdir(parents=True, exist_ok=True)
                if (p := push_state(self.local, self.state, self.name)) is not None:
                    print(f"drive: state {p.name} saved", flush=True)
        except OSError as e:
            print(f"drive: sync failed ({e})", flush=True)

    def run(self) -> None:
        while not self._stop_ev.wait(self.every):
            self._pass(10)

    def stop(self) -> None:
        self._stop_ev.set()
        self._pass(0)
