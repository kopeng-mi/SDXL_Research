"""What the app is doing right now, for the UI and the log.

In-process work (model load, generation) is a Task with a stage, an optional
step count and a detail line. While a task runs, a watcher measures how fast
the disk fills, so downloads show bytes and speed without hooking each library.
Child-process jobs (caption/train/compare/downloads) stay in jobs.py; the UI
shows both.

Log lines go to /content/work/logs/app.log and stdout (the Launch cell).
Step updates are logged at most every LOG_EVERY seconds.
"""

from __future__ import annotations

import logging
import shutil
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from .paths import LOGS

LOG_EVERY = 5.0
log = logging.getLogger("sdxlr")


def setup_logging() -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.FileHandler(LOGS / "app.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.propagate = False
    return LOGS / "app.log"


def gb(n: float) -> str:
    return f"{n / 2**30:.1f} GB" if n >= 2**30 else f"{n / 2**20:.0f} MB"


class Cancelled(Exception):
    pass


@dataclass
class Task:
    kind: str  # load | generate
    title: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    stage: str = "Starting"
    done: int = 0
    total: int = 0
    detail: str = ""
    status: str = "running"  # running | done | failed
    error: str = ""
    started: float = field(default_factory=time.time)
    ended: float | None = None
    _logged: float = 0.0
    cancelled: bool = False
    preview: str = ""  # file name of the latest live preview (served under /work/previews/)
    preview_n: int = 0  # bumps with every preview, to bust the browser cache
    proc: object = None  # a child process doing part of the work (download); killed on cancel
    on_cancel: object = None  # set while an engine op runs; stops the worker when the task is cancelled

    def cancel(self) -> None:
        self.cancelled = True
        log.info(f"[{self.kind}] {self.title}: cancel requested")
        if self.on_cancel is not None:
            try:
                self.on_cancel()
            except Exception as e:
                log.info(f"cancel handler failed: {e}")
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()

    def check(self) -> None:
        if self.cancelled:
            raise Cancelled("cancelled")

    def set_stage(self, stage: str) -> None:
        self.check()
        self.stage, self.done, self.total, self.detail = stage, 0, 0, ""
        log.info(f"[{self.kind}] {self.title}: {stage}")

    def step(self, done: int, total: int) -> None:
        self.check()  # raising inside the diffusers step callback stops the generation
        self.done, self.total = done, total
        if time.time() - self._logged > LOG_EVERY or done == total:
            self._logged = time.time()
            log.info(f"[{self.kind}] {self.stage} {done}/{total}")

    def finish(self, note: str = "") -> None:
        self.status, self.ended = "done", time.time()
        self.stage, self.detail = "Done", note
        log.info(f"[{self.kind}] {self.title}: done in {self.ended - self.started:.0f}s {note}".rstrip())

    def fail(self, err: str) -> None:
        if self.cancelled:
            self.status, self.ended, self.error = "cancelled", time.time(), ""
            log.info(f"[{self.kind}] {self.title}: cancelled")
            return
        self.status, self.ended, self.error = "failed", time.time(), err
        log.info(f"[{self.kind}] {self.title}: failed: {err}")

    @property
    def elapsed(self) -> float:
        return (self.ended or time.time()) - self.started


class Activity:
    def __init__(self) -> None:
        self.tasks: deque[Task] = deque(maxlen=50)

    def start(self, kind: str, title: str) -> Task:
        t = Task(kind, title)
        self.tasks.appendleft(t)
        log.info(f"[{kind}] {title}: started")
        threading.Thread(target=self._watch_disk, args=(t,), daemon=True).start()
        return t

    def running(self) -> list[Task]:
        return [t for t in self.tasks if t.status == "running"]

    @staticmethod
    def _watch_disk(t: Task) -> None:
        """Bytes written to disk while the task runs = downloads (loading only reads)."""
        root = "/content" if Path("/content").exists() else "."
        start = last = shutil.disk_usage(root).used
        last_t, logged, rate = time.time(), 0.0, 0.0
        while t.status == "running":
            time.sleep(1.0)
            if t.status != "running":  # finished while asleep: never write a download note onto a done task
                break
            used, now = shutil.disk_usage(root).used, time.time()
            rate = 0.7 * rate + 0.3 * max(used - last, 0) / (now - last_t)
            last, last_t = used, now
            fetching = t.proc is not None or t.stage.startswith("Getting")  # saves and previews also write to disk
            if fetching and rate > 1 << 20:  # > 1 MB/s while fetching: a download
                t.detail = f"Downloading: {gb(used - start)} so far, {rate / 2**20:.0f} MB/s"
                if now - logged > LOG_EVERY:
                    logged = now
                    log.info(f"[{t.kind}] {t.detail}")
            elif t.detail.startswith("Downloading"):
                t.detail = ""


act = Activity()
