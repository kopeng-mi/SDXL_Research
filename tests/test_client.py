"""Tests for engine/client.py against the real worker loop with a stub engine (no torch).

Run: python tests/test_client.py  (needs psutil and pillow)
Cases in XFAIL need worker-side changes (batch B); they are expected to fail until then.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")
os.environ["SDXLR_SRC"] = SRC
os.environ.setdefault("SDXLR_WORK", tempfile.mkdtemp(prefix="sdxlr_client_test_"))
sys.path.insert(0, SRC)

JOB = {"prompt": "x", "steps": 20, "cfg": 5, "width": 64, "height": 64}
LOAD = {"kind": "sdxl", "path": "/m/a.safetensors"}


def stub_models():
    from PIL import Image
    m = types.ModuleType("sdxlr.engine.models")

    class Job:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Engine:
        def __init__(self):
            self.pipe = None
            self.kind = self.model_path = self.vae_mode = None
            self.adapters = {}

        def unload(self):
            self.pipe = None
            self.kind = self.model_path = None

        def load(self, path, vae="fp16fix", status=lambda s: None):
            for i in range(20):  # 2 s of stage events, like a real load
                status(f"Loading {i}")
                time.sleep(0.1)
            self.pipe = object()
            self.kind, self.model_path, self.vae_mode = "sdxl", path, vae
            return "loaded"

        def generate(self, job):
            for i in range(20):
                job.on_step(i + 1, 20)
                time.sleep(0.1)
            return [Image.new("RGB", (64, 64))], 1, "ok"

    m.Job, m.Engine = Job, Engine
    return m


def stub_entry(conn, cancel_event, parent_pid):
    sys.path.insert(0, os.environ["SDXLR_SRC"])
    sys.modules["sdxlr.engine.models"] = stub_models()
    import sdxlr.engine as pkg
    pkg.models = sys.modules["sdxlr.engine.models"]
    from sdxlr.engine import worker
    worker.main(conn, cancel_event, parent_pid)


def new_client():
    from sdxlr.engine import client
    client._child_entry = stub_entry
    return client.EngineClient()


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
    c = new_client()
    try:
        c.call("load", LOAD)
        assert c.kind == "sdxl" and c.model_path == "/m/a.safetensors" and c.loaded, (c.kind, c.model_path)
        c.call("unload")
        assert not c.loaded and c.kind is None
    finally:
        c.kill()


def test_cancel_then_generate_twice() -> None:
    from sdxlr.activity import Task
    c = new_client()
    try:
        c.call("load", LOAD)
        t, res = Task("generate", "g1"), {}
        th = threading.Thread(target=lambda: res.update(r=gen_task(c, t)), daemon=True)
        th.start()
        time.sleep(0.8)
        t.cancel()
        th.join()
        assert res["r"] == "Cancelled", res
        t0 = time.time()
        assert gen(c, "g2") == "ok"  # waits for the cancelled op to stop, no stale Cancelled
        assert gen(c, "g3") == "ok"  # the old cancel watch must not kill this one
        assert time.time() - t0 < 12
    finally:
        c.kill()


def gen_task(c, task):
    from sdxlr.activity import Cancelled
    try:
        c.call("generate", {"job": JOB, "meta": {}}, task)
        return "ok"
    except Cancelled:
        return "Cancelled"


def test_worker_killed_while_idle() -> None:
    from sdxlr.engine.client import WorkerDied
    c = new_client()
    try:
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
    finally:
        c.kill()


def test_kill_with_reason() -> None:
    from sdxlr.engine.client import WorkerDied
    c = new_client()
    try:
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
    finally:
        c.kill()


def test_heartbeat_while_idle() -> None:
    c = new_client()
    try:
        c.call("load", LOAD)
        c.beat = {}
        time.sleep(3.5)
        assert c.beat.get("type") == "beat", c.beat
    finally:
        c.kill()


def test_load_result_carries_state() -> None:  # batch B: the worker sends state with the load result
    c = new_client()
    try:
        r = c.call("load", LOAD)
        assert "state" in r, r
    finally:
        c.kill()


XFAIL = {"test_load_result_carries_state"}

if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"{name}: ok" + ("  (UNEXPECTED PASS: remove from XFAIL)" if name in XFAIL else ""))
        except Exception as e:
            if name in XFAIL:
                print(f"{name}: xfail (expected until batch B): {type(e).__name__}")
            else:
                failed += 1
                print(f"{name}: FAILED {type(e).__name__}: {e}")
    print("client tests passed" if not failed else f"{failed} client tests FAILED")
    sys.exit(1 if failed else 0)
