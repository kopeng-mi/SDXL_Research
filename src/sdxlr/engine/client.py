"""The engine as the web process sees it: a worker process it owns and restarts.

No torch here. One op at a time (callers keep the user to one Generate); a
second concurrent call waits on _call_lock. Cancel: first ask — the worker
stops at the next step or stage boundary; if it cannot stop within
CANCEL_KILL_SECS (a model load is not interruptible), kill it, and the next
request starts a fresh worker. A dead worker is reported with a plain
sentence; nothing the user made is lost.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import psutil

from ..activity import Cancelled, log

CANCEL_KILL_SECS = 8.0
DEATH_MSG = ("The image engine stopped (most likely out of memory). Your images are safe; "
             "press Generate and it starts again.")


class WorkerDied(RuntimeError):
    pass


def _child_entry(conn, cancel_event, parent_pid: int) -> None:
    """Spawn target: worker (and so torch) is imported only in the child."""
    src = str(Path(__file__).resolve().parents[2])
    if src not in sys.path:
        sys.path.insert(0, src)
    from . import worker
    worker.main(conn, cancel_event, parent_pid)


class EngineClient:
    """Owns the worker process; mirrors its loaded state from heartbeats."""

    needs_prefetch = True  # downloads run before a load; the fake engine has nothing to fetch

    def __init__(self) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self.proc = None
        self.conn = None
        self.cancel_event = None
        self.beat: dict = {}
        self.kind: str | None = None
        self.model_path: str | None = None
        self.vae_mode: str | None = None
        self.adapters: dict[str, str] = {}
        self._call_lock = threading.Lock()
        self._busy = False
        self._killed = False  # set by kill(): the reader reports Cancelled, not death
        self._stale_checked = False

    # ---- state (from heartbeats) ----
    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    @property
    def loaded(self) -> bool:
        return self.model_path is not None

    def _reset(self) -> None:
        self.kind = self.model_path = self.vae_mode = None
        self.adapters = {}
        self.beat = {}

    def _apply_beat(self, ev: dict) -> None:
        self.beat = ev
        self.kind = ev.get("kind")
        self.model_path = ev.get("model_path")
        self.vae_mode = ev.get("vae_mode")
        self.adapters = ev.get("adapters") or {}

    # ---- process ----
    def _kill_stale(self) -> None:
        """A worker left by an earlier web process (crash, restart) holds the GPU."""
        from ..paths import RUN
        f = RUN / "worker.pid"
        try:
            pid = int(f.read_text().strip())
        except (OSError, ValueError):
            f.unlink(missing_ok=True)
            return
        try:
            p = psutil.Process(pid)
            if "python" in p.name().lower() and any("sdxlr" in str(c) for c in p.cmdline()):
                p.kill()
                p.wait(5)
                log.info(f"killed a stale engine worker (pid {pid}) left by an earlier app run")
        except psutil.Error:
            pass
        f.unlink(missing_ok=True)

    def _ensure(self) -> None:
        if self.alive:
            return
        if not self._stale_checked:
            self._kill_stale()
            self._stale_checked = True
        parent, child = self._ctx.Pipe()
        self._reset()
        self._killed = False
        self.cancel_event = self._ctx.Event()
        self.proc = self._ctx.Process(target=_child_entry, args=(child, self.cancel_event, os.getpid()), daemon=True)
        self.proc.start()
        child.close()
        self.conn = parent
        log.info(f"engine worker started (pid {self.proc.pid})")

    def kill(self) -> None:
        """Stop the worker now — frees all its VRAM; the next call restarts it."""
        p, self.proc = self.proc, None
        conn, self.conn = self.conn, None
        self._killed = True
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        self._reset()
        if p is not None:
            if p.is_alive():
                p.kill()
            p.join(3)
            log.info(f"engine worker killed (pid {p.pid})")

    def unload(self) -> None:
        """Drop the loaded model (only when idle; GPU jobs kill the worker instead)."""
        if self._busy:
            raise RuntimeError("the engine is busy")
        if self.alive:
            self.call("unload")

    def shutdown(self) -> None:
        if self.alive and not self._busy:
            try:
                self.call("shutdown")
            except Exception:
                pass
        self.kill()

    # ---- ops ----
    def call(self, op: str, args: dict | None = None, task=None) -> dict:
        """Run one op in the worker and wait for its result. Stage and step
        events feed the task; cancelling the task stops the op."""
        with self._call_lock:
            self._ensure()
            conn = self.conn
            rid = uuid.uuid4().hex[:8]
            self._busy = True
            if task is not None:
                task.on_cancel = self._cancel_current
            try:
                conn.send({"id": rid, "op": op, "args": args or {}})
                while True:
                    try:
                        if not conn.poll(0.25):
                            if not self.proc.is_alive():
                                raise EOFError
                            continue
                        ev = conn.recv()
                    except (EOFError, OSError):
                        try:
                            conn.close()
                        except OSError:
                            pass
                        self.proc = None
                        died = not self._killed
                        self._reset()
                        if died:
                            log.info("engine worker died; last beat: " + json.dumps(self.beat)[:400])
                        raise (WorkerDied(DEATH_MSG) if died else Cancelled("cancelled")) from None
                    t = ev.get("type")
                    if t == "beat":
                        self._apply_beat(ev)
                    elif t == "stage" and task is not None:
                        task.set_stage(ev["stage"])  # raises Cancelled if the task was cancelled
                    elif t == "step" and task is not None:
                        task.step(ev["done"], ev["total"])
                    elif t == "result" and ev.get("id") == rid:
                        return ev.get("result") or {}
                    elif t == "cancelled":
                        raise Cancelled("cancelled")
                    elif t == "error":
                        log.info(f"engine op failed: {ev.get('error')}\n{ev.get('tb', '')[-1500:]}")
                        raise RuntimeError(ev.get("error") or "engine error")
            finally:
                self._busy = False
                if task is not None and task.on_cancel == self._cancel_current:
                    task.on_cancel = None

    def _cancel_current(self) -> None:
        """Stop the running op: ask first, kill if it cannot stop in time."""
        if not self._busy or self.cancel_event is None:
            return
        self.cancel_event.set()

        def watch():
            deadline = time.time() + CANCEL_KILL_SECS
            while self._busy and time.time() < deadline:
                time.sleep(0.1)
            if self._busy and self.proc is not None and self.proc.is_alive():
                log.info("engine worker would not cancel in time (a load is not interruptible); killing it")
                self.kill()

        threading.Thread(target=watch, daemon=True).start()


def make() -> EngineClient:
    """The engine for the web process; the fake one for local UI tests
    (SDXLR_FAKE_ENGINE=1) so a UI run needs no torch and no GPU."""
    if os.environ.get("SDXLR_FAKE_ENGINE"):
        from .fake import FakeEngineClient
        return FakeEngineClient()  # type: ignore[return-value]
    return EngineClient()
