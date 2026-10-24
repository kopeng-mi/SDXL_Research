"""Job queue for the app: one GPU job at a time, CPU jobs alongside.

Every long task (caption, train, compare, download) runs as a child process
in the venv, so stopping it returns all its memory. Before a GPU job starts,
the app's in-process image model is unloaded (on_gpu_job_start) and it can be
reloaded afterwards (on_gpu_job_end).

Progress is parsed from tqdm-style "12/300" output; ETA = elapsed / done * left.
Every job line goes to the app log (logs/app.log + the Launch cell) and to
logs/jobs/<id>.log; progress redraws are logged at most every 5 s.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..activity import LOG_EVERY, LOGS, log

REPO = Path(__file__).resolve().parents[3]
_PROGRESS = re.compile(r"(\d+)/(\d+)\s*\[")  # tqdm: " 12/300 [00:10<..."
_STEP = re.compile(r"\[(\d+)/(\d+)\]")  # our scripts: "[3/5] file ..."

gpu_lock = threading.Lock()  # keeps generation and GPU jobs apart (held across a whole generate)


@dataclass
class JobInfo:
    id: str
    kind: str  # caption | train | train_zimage | compare | download | ...
    title: str
    cmd: list[str]
    gpu: bool
    status: str = "queued"  # queued | running | done | failed | cancelled
    created: float = field(default_factory=time.time)
    started: float | None = None
    ended: float | None = None
    done: int = 0
    logged: float = 0.0
    total: int = 0
    log: deque = field(default_factory=lambda: deque(maxlen=400))
    returncode: int | None = None
    proc: subprocess.Popen | None = None

    @property
    def eta(self) -> float | None:
        if self.status != "running" or not self.started or not self.done or not self.total:
            return None
        return (time.time() - self.started) / self.done * (self.total - self.done)

    def summary(self) -> str:
        pct = f" {self.done}/{self.total}" if self.total else ""
        eta = f" · ETA {fmt_secs(self.eta)}" if self.eta else ""
        took = f" · {fmt_secs((self.ended or time.time()) - self.started)}" if self.started else ""
        return f"{self.status:9s} {self.title}{pct}{eta}{took}"


def fmt_secs(s: float | None) -> str:
    if s is None:
        return "?"
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


class JobManager:
    def __init__(self, on_gpu_job_start=None, on_gpu_job_end=None) -> None:
        self.jobs: dict[str, JobInfo] = {}
        self._gpu_queue: deque[str] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self.on_gpu_job_start = on_gpu_job_start
        self.on_gpu_job_end = on_gpu_job_end
        threading.Thread(target=self._gpu_worker, daemon=True).start()

    # ---- public ----
    def submit(self, kind: str, title: str, module_args: list[str], gpu: bool = True) -> JobInfo:
        """module_args: e.g. ['-m', 'sdxlr.caption', '--images', ...] run with the venv python."""
        job = JobInfo(id=uuid.uuid4().hex[:8], kind=kind, title=title,
                      cmd=[sys.executable, "-u", *map(str, module_args)], gpu=gpu)
        with self._lock:
            self.jobs[job.id] = job
        if gpu:
            self._gpu_queue.append(job.id)
            self._wake.set()
        else:
            threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def cancel(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        if job.status == "queued":
            job.status = "cancelled"
        elif job.status == "running" and job.proc:
            job.status = "cancelled"
            job.proc.terminate()

    def gpu_busy(self) -> bool:
        return any(j.gpu and j.status == "running" for j in self.jobs.values())

    def recent(self, n: int = 8) -> list[JobInfo]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)[:n]

    # ---- workers ----
    def _gpu_worker(self) -> None:
        while True:
            self._wake.wait()
            while self._gpu_queue:
                job = self.jobs[self._gpu_queue.popleft()]
                if job.status == "cancelled":
                    continue
                if self.on_gpu_job_start:
                    self.on_gpu_job_start(job)  # frees the app's VRAM
                try:
                    self._run(job)
                finally:
                    if self.on_gpu_job_end:
                        self.on_gpu_job_end(job)
            self._wake.clear()

    def _run(self, job: JobInfo) -> None:
        job.status, job.started = "running", time.time()
        log.info(f"[job {job.id}] {job.title}: started")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
               "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": str(REPO / "src")}
        try:
            job.proc = subprocess.Popen(job.cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT)
            buf = b""
            while chunk := job.proc.stdout.read1(4096):
                buf += chunk
                *lines, buf = re.split(rb"[\r\n]", buf)
                for raw in lines:
                    self._line(job, raw.decode("utf-8", "replace").rstrip())
            job.returncode = job.proc.wait()
        except Exception as e:  # e.g. bad command
            job.log.append(f"{type(e).__name__}: {e}")
            job.returncode = -1
        job.ended = time.time()
        if job.status != "cancelled":
            job.status = "done" if job.returncode == 0 else "failed"
        log.info(f"[job {job.id}] {job.title}: {job.status} (exit {job.returncode}, {fmt_secs(job.ended - job.started)})")

    @staticmethod
    def _line(job: JobInfo, text: str) -> None:
        if not text:
            return
        (LOGS / "jobs").mkdir(parents=True, exist_ok=True)
        with open(LOGS / "jobs" / f"{job.id}.log", "a", encoding="utf-8") as f:
            f.write(text + "\n")
        m = _PROGRESS.search(text) or _STEP.search(text)
        if m:
            job.done, job.total = int(m.group(1)), int(m.group(2))
            if time.time() - job.logged > LOG_EVERY:
                job.logged = time.time()
                log.info(f"[job {job.id}] {text}")
            if job.log and (_PROGRESS.search(job.log[-1]) or "it/s" in job.log[-1] or "s/it" in job.log[-1]):
                job.log[-1] = text  # replace the previous progress line instead of piling up
                return
        else:
            log.info(f"[job {job.id}] {text}")
        job.log.append(text)


def _before_gpu_job(job) -> None:
    from . import gen  # late import: gen needs gpu_lock and jm from here
    gen.before_gpu_job(job)


jm = JobManager(on_gpu_job_start=_before_gpu_job, on_gpu_job_end=lambda job: gpu_lock.release())
