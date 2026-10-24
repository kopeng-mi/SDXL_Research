"""The engine as the web process sees it: a worker process it owns and restarts.

No torch here. One op at a time (callers keep the user to one Generate); a
second concurrent call waits on _call_lock. A reader thread per worker always
drains the pipe: heartbeats update the mirrored state at any time, other
events go to the running call (events of an abandoned request are dropped).
Cancel: first ask — the worker stops at the next step or stage boundary; if
it cannot stop within CANCEL_KILL_SECS (a model load is not interruptible),
kill it, and the next request starts a fresh worker. A dead worker is
reported with a plain sentence, also when it died while idle; nothing the
user made is lost.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import queue
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
LOW_RAM_MSG = ("The image engine was stopped because the VM was low on memory. Your images are safe; "
               "press Generate to start it again.")
TERMINAL = ("result", "error", "cancelled")  # the last event of a request


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

    def __init__(self) -> None:
        self.needs_prefetch = not os.environ.get("SDXLR_FAKE_ENGINE")  # downloads run before a load; the stub engine has nothing to fetch
        self._ctx = multiprocessing.get_context("spawn")
        self.proc = None
        self.conn = None
        self.cancel_event = None
        self.beat: dict = {}
        self.kind: str | None = None
        self.model_path: str | None = None
        self.vae_mode: str | None = None
        self.adapters: dict[str, str] = {}
        self._call_lock = threading.Lock()  # one op at a time
        self._lock = threading.RLock()  # guards the worker handles and the request bookkeeping
        self._gen = 0  # bumped whenever a worker is replaced or killed; its old reader then stops
        self._cur: str | None = None  # id of the running call
        self._q: queue.Queue | None = None  # events for the running call
        self._pending: set[str] = set()  # sent, no final event yet (an abandoned op may still run)
        self.died_idle = False  # the worker stopped with no call running: the next call reports it once
        self.death_msg = DEATH_MSG
        self._stale_checked = False
        self.on_death = None  # guard hook: export logs when the worker dies

    # ---- state (from heartbeats) ----
    @property
    def alive(self) -> bool:
        p = self.proc
        return p is not None and p.is_alive()

    @property
    def loaded(self) -> bool:
        return self.model_path is not None and self.alive

    @property
    def busy(self) -> bool:
        return self._cur is not None

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

    def _apply_loaded(self, args: dict, result: dict) -> None:
        """State right after a load, before the next heartbeat says so."""
        if args.get("kind") == "zimage":
            from ..paths import z_key
            self.kind, self.vae_mode = "zimage", None
            self.model_path = z_key(args["transformer"], args["text_encoder"], args["te_quant"])
        else:
            self.kind, self.model_path, self.vae_mode = "sdxl", args.get("path"), args.get("vae", "fp16fix")

    # ---- process ----
    def _kill_stale(self) -> None:
        """A worker left by an earlier web process (crash, restart) holds the GPU."""
        from ..paths import RUN
        f = RUN / "worker.pid"
        try:
            pid_s, ctime_s = f.read_text().split()  # "<pid> <create time>"
            pid, ctime = int(pid_s), float(ctime_s)
        except (OSError, ValueError):
            f.unlink(missing_ok=True)
            return
        try:
            p = psutil.Process(pid)
            if abs(p.create_time() - ctime) < 1.0:  # the same process, not a reused pid
                p.kill()
                p.wait(5)
                log.info(f"killed a stale engine worker (pid {pid}) left by an earlier app run")
        except psutil.Error:
            pass
        f.unlink(missing_ok=True)

    def _ensure(self) -> None:
        """Start a worker if there is none (caller holds _lock)."""
        if self.alive:
            return
        if not self._stale_checked:
            self._kill_stale()
            self._stale_checked = True
        parent, child = self._ctx.Pipe()
        self._reset()
        self._gen += 1
        self.cancel_event = self._ctx.Event()
        proc = self._ctx.Process(target=_child_entry, args=(child, self.cancel_event, os.getpid()), daemon=True)
        proc.start()
        child.close()
        self.proc, self.conn = proc, parent
        threading.Thread(target=self._read, args=(self._gen, proc, parent), daemon=True).start()
        log.info(f"engine worker started (pid {proc.pid})")

    def _read(self, gen: int, proc, conn) -> None:
        """Drain the pipe for as long as this worker lives."""
        while self._gen == gen:
            try:
                if conn.poll(0.5):
                    ev = conn.recv()
                elif proc.is_alive() or conn.poll(0):
                    continue
                else:
                    break
            except (EOFError, OSError):
                break
            if not isinstance(ev, dict):
                continue
            t, rid = ev.get("type"), ev.get("id")
            with self._lock:
                if self._gen != gen:
                    return
                if t == "beat":
                    self._apply_beat(ev)
                    continue
                if t in TERMINAL:
                    self._pending.discard(rid)
                if rid == self._cur and self._q is not None:
                    self._q.put(ev)  # events of any other request are stale: dropped
        with self._lock:
            if self._gen == gen:
                self._died_locked(proc)

    def _died_locked(self, proc) -> None:
        """The worker is gone without being asked to stop (caller holds _lock)."""
        conn, self.conn, self.proc = self.conn, None, None
        self._gen += 1
        self._pending.clear()
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        proc.join(2)
        code = proc.exitcode
        log.info(f"engine worker died (exit code {code}{', killed: likely out of memory' if code == -9 else ''}); "
                 "last beat: " + json.dumps(self.beat)[:400])
        self._reset()
        self._stopped(DEATH_MSG)

    def _stopped(self, msg: str) -> None:
        """Tell the running call (or, when idle, the next one) that the worker stopped."""
        if self._cur is not None and self._q is not None:
            self._q.put({"type": "_died", "msg": msg})
        else:
            self.died_idle, self.death_msg = True, msg
        if self.on_death is not None:
            threading.Thread(target=self.on_death, daemon=True).start()

    def kill(self, reason: str | None = None) -> None:
        """Stop the worker now — frees all its VRAM; the next call restarts it.
        reason=None is a user cancel (the running call raises Cancelled); a reason
        (the RAM guard) makes the running call raise WorkerDied naming it."""
        with self._lock:
            proc, conn, self.proc, self.conn = self.proc, self.conn, None, None
            if proc is None:
                return
            self._gen += 1
            self._pending.clear()
            self._reset()
            if reason is None:
                if self._cur is not None and self._q is not None:
                    self._q.put({"type": "_killed"})
            else:
                self._stopped(LOW_RAM_MSG if reason == "low RAM" else
                              f"The image engine was stopped ({reason}). Your images are safe; "
                              "press Generate to start it again.")
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if proc.is_alive():
            proc.kill()
        proc.join(3)
        log.info(f"engine worker killed (pid {proc.pid}{', ' + reason if reason else ''})")

    def unload(self) -> None:
        """Drop the loaded model (only when idle; GPU jobs kill the worker instead)."""
        if self.busy:
            raise RuntimeError("the engine is busy")
        if self.alive:
            self.call("unload")

    def shutdown(self) -> None:
        if self.alive and not self.busy:
            try:
                self.call("shutdown")
            except Exception:
                pass
        self.kill()

    # ---- ops ----
    def _wait_abandoned(self, task) -> None:
        """A cancelled op can still be winding down in the worker: wait for it
        (the cancel watch kills the worker at its deadline)."""
        deadline = time.time() + CANCEL_KILL_SECS + 5
        while True:
            with self._lock:
                if not self._pending:
                    return
            if task is not None:
                task.check()
            if time.time() > deadline:
                log.info("engine worker kept running a cancelled op; killing it")
                self.kill()
                return
            time.sleep(0.1)

    def call(self, op: str, args: dict | None = None, task=None) -> dict:
        """Run one op in the worker and wait for its result. Stage and step
        events feed the task; cancelling the task stops the op."""
        args = args or {}
        with self._call_lock:
            with self._lock:
                if self.proc is not None and not self.proc.is_alive():  # died unseen: report it first
                    self._died_locked(self.proc)
                if self.died_idle:
                    self.died_idle = False
                    raise WorkerDied(self.death_msg)
            self._wait_abandoned(task)
            rid = uuid.uuid4().hex[:8]
            with self._lock:
                self._ensure()
                q = self._q = queue.Queue()
                self._cur = rid
                self._pending.add(rid)
                conn = self.conn
            if task is not None:
                task.on_cancel = self._cancel_current
            try:
                try:
                    conn.send({"id": rid, "op": op, "args": args})
                except (OSError, ValueError):
                    with self._lock:
                        if self.conn is conn and self.proc is not None and not self.proc.is_alive():
                            self._died_locked(self.proc)
                    self.died_idle = False
                    raise WorkerDied(self.death_msg) from None
                while True:
                    ev = q.get()
                    t = ev.get("type")
                    if t == "_died":
                        self.died_idle = False
                        raise WorkerDied(ev["msg"])
                    if t == "_killed":
                        raise Cancelled("cancelled")
                    if t == "stage" and task is not None:
                        task.set_stage(ev["stage"])  # raises Cancelled if the task was cancelled
                    elif t == "step" and task is not None:
                        task.step(ev["done"], ev["total"])
                    elif t == "result":
                        res = ev.get("result") or {}
                        st = res.pop("state", None)
                        with self._lock:
                            if st:  # the worker's own view after the op, ahead of the next heartbeat
                                self._apply_beat({**self.beat, **st})
                            elif op == "load":
                                self._apply_loaded(args, res)
                            elif op == "unload":
                                self._reset()
                        return res
                    elif t == "cancelled":
                        raise Cancelled("cancelled")
                    elif t == "error":
                        log.info(f"engine op failed: {ev.get('error')}\n{ev.get('tb', '')[-1500:]}")
                        if ev.get("plain"):  # a sentence for the user (a bad request, not an engine failure)
                            raise ValueError(ev["error"])
                        raise RuntimeError(ev.get("error") or "engine error")
            finally:
                with self._lock:
                    if self._cur == rid:
                        self._cur = self._q = None
                if task is not None and task.on_cancel == self._cancel_current:
                    task.on_cancel = None

    def _cancel_current(self) -> None:
        """Stop the running op: ask first, kill if it cannot stop in time. The
        watch is armed for this request only; a later op is never killed by it."""
        with self._lock:
            rid, ev, gen = self._cur, self.cancel_event, self._gen
        if rid is None or ev is None:
            return
        ev.set()

        def watch():
            deadline = time.time() + CANCEL_KILL_SECS
            while time.time() < deadline:
                with self._lock:
                    if rid not in self._pending or self._gen != gen:
                        return
                time.sleep(0.1)
            with self._lock:
                still = rid in self._pending and self._gen == gen
            if still:
                log.info("engine worker would not cancel in time (a load is not interruptible); killing it")
                self.kill()

        threading.Thread(target=watch, daemon=True).start()


def make() -> EngineClient:
    """The engine for the web process. SDXLR_FAKE_ENGINE=1 makes the worker run the stub engine."""
    return EngineClient()
