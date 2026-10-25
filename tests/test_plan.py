"""Plan math and Drive sync. Run: python tests/test_plan.py"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sdxlr import drive, loraformat, plan  # noqa: E402


def test_auto_repeats_keep_views() -> None:
    assert plan.auto(11, 15, 990) == (6, 15)
    assert plan.auto(40, 15, 990) == (2, 15)
    assert plan.auto(200, 15, 990) == (1, 5)
    assert plan.auto(1, 15, 990) == (20, 15)  # capped
    assert plan.auto(11, 10, 440) == (4, 10)


def test_train_plan() -> None:
    p = plan.train("sdxl", "person", 11, 1024, 3, 15)
    assert (p.repeats, p.epochs, p.steps, p.est) == (6, 15, 330, False)
    assert round(p.seconds / 60) == 30
    z = plan.train("zimage", "person", 11, 512, 1, 10)
    assert z.steps == 440 and round(z.seconds / 60) == 47
    assert plan.train("sdxl", "person", 200, 1024, 3, 15).steps == 335
    small = plan.train("sdxl", "style", 3, 768, 3, 10)
    assert small.est and len(small.warnings) == 2
    assert plan.train("sdxl", "person", 11, 1024, 3, 15, repeats=2).repeats == 2


def test_step_time_scales() -> None:
    t, est = plan.step_time("sdxl", 768, 3)
    assert est and abs(t - 5.1 * 0.5625) < 1e-9


def test_buckets_and_checks() -> None:
    assert plan.bucket(1024, 1024, 1024, 64) == (1024, 1024)
    assert plan.bucket(3000, 4000, 1024, 64) == (832, 1152)
    b = plan.buckets([(1000, 1000), (1200, 1200), (3000, 4000)], "sdxl", 1024)
    assert list(b.values()) == [2, 1]
    assert plan.too_small(400, 900, 1024) and not plan.too_small(600, 900, 1024)


def test_caption_math() -> None:
    assert plan.caption_quant(4) == "fp16" and plan.caption_quant(8) == "nf4"
    assert plan.caption_batch("nf4") == 2
    assert plan.vision_tokens(1024, 1024) == 576
    assert plan.vision_tokens(1024, 1024, 1024) == 1024
    assert plan.over_budget("w " * 56, "sdxl") and not plan.over_budget("w " * 55, "sdxl")
    assert abs(plan.lora_scale(16, 24) - 0.6667) < 1e-3


def test_drive_sync() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="sdxlr_drive_test_"))
    src, dst = tmp / "src", tmp / "dst"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("a")
    (src / "sub" / "b.txt").write_text("b")
    assert drive.pull(src, dst) == 2 and drive.pull(src, dst) == 0
    os.utime(src / "a.txt", ns=(1, 1))
    assert drive.pull(src, dst) == 1
    assert drive.mirror(src / "a.txt", tmp / "m").read_text() == "a"

    out, st = tmp / "out", tmp / "state"
    for d in ("me-000002-state", "me-000004-state", "other-000009-state"):
        (out / d).mkdir(parents=True)
        (out / d / "x.bin").write_text(d)
    st.mkdir()
    (st / "me-000001-state").mkdir()
    (st / "me-000003-state.tmp").mkdir()  # an interrupted copy
    assert drive.push_state(out, st, "me").name == "me-000004-state"
    assert sorted(p.name for p in st.iterdir()) == ["me-000004-state"]
    local, n = drive.resume(st, tmp / "local", "me")
    assert n == 4 and (local / "x.bin").read_text() == "me-000004-state"
    assert drive.done("me_r4-000002-state", "me") == 6
    assert drive.resume(tmp / "none", tmp / "local", "me") == (None, 0)

    (out / "me-000004.safetensors").write_text("w")
    assert drive.mirror_new(out, tmp / "loras") == 0  # still being written
    s = drive.Syncer(out, "me", tmp / "loras", tmp / "st2", every=999)
    s.stop()
    assert (tmp / "loras" / "me-000004.safetensors").exists() and (tmp / "st2" / "me-000004-state").exists()



def _st(path: Path, keys: list[str]) -> None:
    import json
    import struct
    head = json.dumps({k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"\0\0")


def test_vpred_and_pick() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="sdxlr_pick_test_"))
    _st(tmp / "v.safetensors", ["v_pred", "ztsnr", "model.diffusion_model.x"])
    _st(tmp / "e.safetensors", ["model.diffusion_model.x"])
    assert loraformat.is_vpred(tmp / "v.safetensors") and not loraformat.is_vpred(tmp / "e.safetensors")
    run = tmp / "run"
    run.mkdir()
    for f in ("me-000001", "me-000002", "me"):
        _st(run / f"{f}.safetensors", ["x"])
    (run / "me-000002.json").write_text("{}")
    assert [e for e, _ in loraformat.pick_epochs(run, "last:2")] == [2, 3]
    assert loraformat.pick(run, 2, tmp / "out", "me").name == "me_e2.safetensors"
    assert (tmp / "out" / "me_e2.json").exists()
    assert loraformat.pick(run, 0, tmp / "out", "me").name == "me_e3.safetensors"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
