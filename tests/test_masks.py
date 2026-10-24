"""Tests for the mask upload route (service/masks.py).

Run: uv run --no-project -q --with nicegui==3.18.0 --with pillow --with numpy --with pyyaml python -I tests/test_masks.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

os.environ["SDXLR_WORK"] = tempfile.mkdtemp(prefix="sdxlr_masks_test_")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from sdxlr.service import masks  # noqa: E402

app = FastAPI()
masks.register(app)
client = TestClient(app)


def png(size=(64, 48), mode="L", color=255) -> bytes:
    b = io.BytesIO()
    Image.new(mode, size, color).save(b, "PNG")
    return b.getvalue()


def post(code: str, data: bytes):
    return client.post(f"/api/mask/{code}", content=data)


def test_good_upload_is_stored_as_l() -> None:
    r = post("1234", png(mode="RGB", color=(255, 255, 255)))
    assert r.status_code == 200, r.text
    mid = r.json()["id"]
    p = masks.path_for(mid)
    assert p is not None and p.exists() and p.parent.name == "1234" and p.is_relative_to(masks.MASKS)
    assert Image.open(p).mode == "L"


def test_rgba_alpha_is_the_paint() -> None:
    b = io.BytesIO()
    Image.new("RGBA", (10, 10), (255, 255, 255, 0)).save(b, "PNG")
    mid = post("1234", b.getvalue()).json()["id"]
    assert Image.open(masks.path_for(mid)).getextrema() == (0, 0)  # transparent = not painted


def test_bad_session_codes() -> None:
    for code in ("12", "12345", "abcd", "12a4", "%2e%2e", "..%2f.."):
        assert post(code, png()).status_code in (404, 405), code
    assert not (masks.MASKS.parent.parent / "x").exists()


def test_not_a_png() -> None:
    assert post("1234", b"GIF89a" + b"0" * 100).status_code == 415
    assert post("1234", masks.PNG_MAGIC + b"garbage").status_code == 415
    assert post("1234", b"").status_code == 415


def test_oversize() -> None:
    assert post("1234", masks.PNG_MAGIC + b"0" * (masks.MAX_BYTES + 1)).status_code == 413
    assert post("1234", png((masks.MAX_SIDE + 1, 8))).status_code == 422


def test_huge_png_is_refused_before_decoding() -> None:
    data = png((9000, 9000))  # a few KB of PNG that would decode to 81 MB (built before load is watched)
    calls = []
    orig = Image.Image.load
    Image.Image.load = lambda self, *a, **k: (calls.append(1), orig(self, *a, **k))[1]  # noqa: E731
    try:
        r = post("1234", data)
    finally:
        Image.Image.load = orig
    assert r.status_code == 422 and not calls, (r.status_code, calls)


def test_bad_content_length_is_400() -> None:
    r = client.post("/api/mask/1234", content=png(), headers={"content-length": "abc"})
    assert r.status_code in (400, 422), r.status_code  # never a 500


def test_path_for_accepts_only_our_ids() -> None:
    for bad in ("../../etc/passwd", "1234-../x", "1234-" + "g" * 32, "1234-abc", "12345-" + "a" * 32, "", None,
                "1234-" + "a" * 32 + "/../x"):
        assert masks.path_for(bad) is None, bad
    assert masks.path_for("1234-" + "a" * 32) == masks.MASKS / "1234" / ("a" * 32 + ".png")


def test_cleanup_keeps_newest() -> None:
    for _ in range(masks.KEEP_PER_SESSION + 5):
        masks.store("4321", png())
    assert len(list((masks.MASKS / "4321").glob("*.png"))) <= masks.KEEP_PER_SESSION


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"{name}: ok")
        except Exception as e:
            failed += 1
            print(f"{name}: FAILED {type(e).__name__}: {e}")
    print("mask tests passed" if not failed else f"{failed} mask tests FAILED")
    sys.exit(1 if failed else 0)
