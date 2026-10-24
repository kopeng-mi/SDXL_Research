"""Dataset import: duplicates skipped, same-name files kept apart. Run: python tests/test_data.py"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sdxlr import data  # noqa: E402


def test_import_dedupes_and_never_overwrites() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="sdxlr_data_test_"))
    a, b, dest = tmp / "a", tmp / "b", tmp / "ds"
    a.mkdir()
    b.mkdir()
    Image.new("RGB", (300, 300), (200, 10, 10)).save(a / "photo.png")
    (a / "photo.txt").write_text("red", encoding="utf-8")
    Image.new("RGB", (300, 300), (10, 200, 10)).save(b / "photo.png")  # same name, other photo
    (b / "photo.txt").write_text("green", encoding="utf-8")
    Image.new("RGB", (300, 300), (200, 10, 10)).save(b / "copy_of_red.png")  # same bytes as a/photo.png
    summary = data.import_sources([str(a), str(b)], dest)
    names = sorted(p.name for p in dest.iterdir())
    assert names == ["photo.png", "photo.txt", "photo_2.png", "photo_2.txt"], names
    assert (dest / "photo_2.txt").read_text(encoding="utf-8") == "green"
    assert "2 images" in summary and "1 duplicate" in summary, summary
    again = data.import_sources([str(a)], dest)  # importing the same folder again adds nothing
    assert "2 images" in again and "1 duplicate" in again, again


if __name__ == "__main__":
    test_import_dedupes_and_never_overwrites()
    print("test_import_dedupes_and_never_overwrites: ok\ndata tests passed")
