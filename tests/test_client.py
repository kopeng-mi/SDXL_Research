"""Tests for engine/client.py against the real worker loop with the stub engine from engine/fake.py (no torch).

Run: python tests/test_client.py  (needs psutil and pillow)
Each test gets its own work dir and environment switches (SDXLR_FAKE_*, SDXLR_BEAT_SECS).
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
os.environ.setdefault("SDXLR_WORK", tempfile.mkdtemp(prefix="sdxlr_client_test_"))
sys.path.insert(0, SRC)

JOB = {"prompt": "x", "steps": 20, "cfg": 5, "width": 64, "height": 64}
LOAD = {"kind": "sdxl", "path": "/m/a.safetensors"}
SWITCHES = ("SDXLR_FAKE_CRASH", "SDXLR_FAKE_SLOWLOAD", "SDXLR_FAKE_FAST", "SDXLR_FAKE_SCALE", "SDXLR_BEAT_SECS")


@contextlib.contextmanager
def client(**env):
    """A client whose worker (spawned with this environment) runs the stub engine in a fresh work dir."""
    for k in SWITCHES:
        os.environ.pop(k, None)
    os.environ.update({"SDXLR_FAKE_ENGINE": "1", "SDXLR_WORK": tempfile.mkdtemp(prefix="sdxlr_client_test_"), **env})
    from sdxlr.engine import client as mod
    c = mod.EngineClient()
    try:
        yield c
    finally:
        c.kill()


def gen(c, name="g"):
    from sdxlr.activity import Cancelled, Task
    try:
        c.call("generate", {"job": JOB, "meta": {}}, Task("generate", name))
        return "ok"
    except Cancelled:
        return "Cancelled"
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def test_state_after_load() -> None:
    with client(SDXLR_FAKE_FAST="1") as c:
        c.call("load", LOAD)
        assert c.kind == "sdxl" and c.model_path == "/m/a.safetensors" and c.loaded, (c.kind, c.model_path)
        c.call("unload")
        assert not c.loaded and c.kind is None


def test_cancel_then_generate_twice() -> None:
    from sdxlr.activity import Cancelled, Task
    with client(SDXLR_FAKE_SCALE="0.15") as c:
        c.call("load", LOAD)
        t, res = Task("generate", "g1"), {}

        def run():
            try:
                c.call("generate", {"job": JOB, "meta": {}}, t)
                res["r"] = "ok"
            except Cancelled:
                res["r"] = "Cancelled"

        th = threading.Thread(target=run, daemon=True)
        th.start()
        time.sleep(0.8)
        t.cancel()
        th.join()
        assert res["r"] == "Cancelled", res
        t0 = time.time()
        assert gen(c, "g2") == "ok"  # waits for the cancelled op to stop, no stale Cancelled
        assert gen(c, "g3") == "ok"  # the old cancel watch must not kill this one
        assert time.time() - t0 < 12


def test_worker_killed_while_idle() -> None:
    from sdxlr.engine.client import WorkerDied
    with client(SDXLR_FAKE_FAST="1") as c:
        c.call("load", LOAD)
        c.proc.kill()
        deadline = time.time() + 5
        while c.alive and time.time() < deadline:
            time.sleep(0.1)
        time.sleep(1.0)  # the reader notices
        assert not c.loaded and c.died_idle, (c.loaded, c.died_idle)
        try:
            c.call("generate", {"job": JOB, "meta": {}})
            raise AssertionError("expected WorkerDied")
        except WorkerDied as e:
            assert "stopped" in str(e)
        c.call("load", LOAD)  # the next call works
        assert c.loaded


def test_kill_with_reason() -> None:
    from sdxlr.engine.client import WorkerDied
    with client(SDXLR_FAKE_SCALE="0.15") as c:
        c.call("load", LOAD)
        out = {}

        def run():
            try:
                c.call("generate", {"job": JOB, "meta": {}})
            except Exception as e:
                out["e"] = e

        th = threading.Thread(target=run, daemon=True)
        th.start()
        time.sleep(0.8)
        c.kill(reason="low RAM")
        th.join(5)
        assert isinstance(out.get("e"), WorkerDied) and "low on memory" in str(out["e"]), out


def test_heartbeat_while_idle() -> None:
    with client(SDXLR_FAKE_FAST="1", SDXLR_BEAT_SECS="0.3") as c:
        c.call("load", LOAD)
        c.beat = {}
        time.sleep(1.5)
        assert c.beat.get("type") == "beat" and c.beat.get("kind") == "sdxl", c.beat


def test_beat_never_unloads_a_fresh_load() -> None:
    with client(SDXLR_FAKE_FAST="1", SDXLR_BEAT_SECS="0.005") as c:
        for i in range(20):
            path = f"/m/model{i}.safetensors"
            c.call("load", {"kind": "sdxl", "path": path})
            assert c.loaded and c.model_path == path, (i, c.kind, c.model_path)
            time.sleep(0.02)  # beats keep coming; they must agree with the load
            assert c.loaded and c.model_path == path, (i, c.kind, c.model_path)


def test_cancel_during_uninterruptible_load() -> None:
    from sdxlr.activity import Cancelled, Task
    with client(SDXLR_FAKE_SLOWLOAD="1", SDXLR_FAKE_SCALE="0.1") as c:
        t, res = Task("load", "slow"), {}

        def run():
            try:
                c.call("load", LOAD, t)
                res["r"] = "ok"
            except Cancelled:
                res["r"] = "Cancelled"

        th = threading.Thread(target=run, daemon=True)
        th.start()
        time.sleep(1.0)
        t.cancel()
        t0 = time.time()
        th.join(20)
        assert res.get("r") == "Cancelled" and 6 < time.time() - t0 < 11, (res, time.time() - t0)  # the 8 s kill
        c.call("load", LOAD)  # a fresh worker; the slow load only happens once
        assert gen(c) == "ok" and time.time() - t0 < 14


def test_worker_crash_then_restart() -> None:
    from sdxlr.engine.client import WorkerDied
    with client(SDXLR_FAKE_CRASH="1", SDXLR_FAKE_SCALE="0.1") as c:
        c.call("load", LOAD)
        pid = c.proc.pid
        try:
            c.call("generate", {"job": JOB, "meta": {}})
            raise AssertionError("expected WorkerDied")
        except WorkerDied as e:
            assert "stopped" in str(e)
        assert not c.loaded
        c.call("load", LOAD)
        assert c.proc.pid != pid
        assert gen(c) == "ok"


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"{name}: ok")
        except Exception as e:
            failed += 1
            print(f"{name}: FAILED {type(e).__name__}: {e}")
    print("client tests passed" if not failed else f"{failed} client tests FAILED")
    sys.exit(1 if failed else 0)
