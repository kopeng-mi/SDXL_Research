"""Notebook-side helpers. Stdlib only: imported by the Colab kernel, never torch.

Colab's web UI does not show output written by child processes to the real
stdout, so every GPU job is run through `run()`, which pipes it into the cell.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VENV_PY = "/content/venv/bin/python"


def run(args: list[str], cwd: str | Path = REPO, progress_every: float = 5.0, py: str = VENV_PY) -> None:
    """Run `python args...` in the venv (or another interpreter: `py`) and stream its output into the cell.

    tqdm redraws with '\\r'; those updates are printed at most every
    `progress_every` seconds so the cell does not fill with progress bars.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
           "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.Popen([py, "-u", *map(str, args)], cwd=cwd, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    buf, last = b"", 0.0
    try:
        while chunk := proc.stdout.read1(4096):
            buf += chunk
            while True:
                i = min((buf.find(c) for c in (b"\n", b"\r") if c in buf), default=-1)
                if i < 0:
                    break
                line, sep, buf = buf[:i], buf[i:i + 1], buf[i + 1:]
                text = line.decode("utf-8", "replace").rstrip()
                if not text:
                    continue
                if sep == b"\r":  # progress redraw
                    if time.time() - last < progress_every:
                        continue
                    last = time.time()
                print(text, flush=True)
        if buf.strip():
            print(buf.decode("utf-8", "replace").rstrip(), flush=True)
    except KeyboardInterrupt:  # Colab's stop button: stop the child cleanly
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("stopped", flush=True)
        return
    if proc.wait() != 0:
        raise SystemExit(f"failed with exit code {proc.returncode}")


def serve(args: list[str], max_restarts: int = 5) -> None:
    """Run the app and start it again if it crashes (not when you stop the cell).

    More than `max_restarts` crashes within 10 minutes means something is
    broken for good: stop and leave the log in the cell.
    """
    crashes: list[float] = []
    while True:
        try:
            run(args)
            return  # stopped from the cell, or exited cleanly
        except SystemExit as e:
            crashes = [t for t in crashes if time.time() - t < 600] + [time.time()]
            if len(crashes) > max_restarts:
                print(f"the app crashed {len(crashes)} times in 10 minutes; not restarting ({e})", flush=True)
                raise
            print(f"the app stopped ({e}); starting it again in 5 s (the address stays the same)", flush=True)
            time.sleep(5)


SECRETS = ("HF_TOKEN", "CIVITAI_TOKEN", "TS_AUTHKEY")


def load_secrets() -> dict[str, bool]:
    """Export tokens from Colab Secrets into the env (child processes inherit them).
    Returns which were found; values are never printed."""
    found = {}
    for key in SECRETS:
        if not os.environ.get(key):
            try:
                from google.colab import userdata  # type: ignore
                os.environ[key] = userdata.get(key)
            except Exception:
                pass
        found[key] = bool(os.environ.get(key))
    return found


def _thumb(p: Path, side: int) -> bytes:
    """Small JPEG of a photo (the Colab kernel has Pillow), so the notebook stays light."""
    import io
    from PIL import Image, ImageOps
    with Image.open(p) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def show_images(paths: list[str | Path], width: int = 256, captions: str | Path | bool = True) -> None:
    """Thumbnails in the cell, each with its .txt caption below (next to the image, or in the `captions` dir)."""
    import base64
    import html
    from IPython.display import HTML, display  # type: ignore
    parts = []
    for p in map(Path, paths):
        img = base64.b64encode(_thumb(p, width * 2)).decode()
        mime = "jpeg"
        txt = Path(captions) / f"{p.stem}.txt" if isinstance(captions, (str, Path)) else p.with_suffix(".txt")
        cap = html.escape(txt.read_text(encoding="utf-8")) if captions and txt.exists() else ""
        parts.append(f'<figure style="display:inline-block;width:{width}px;margin:4px;vertical-align:top">'
                     f'<img src="data:image/{mime};base64,{img}" style="width:100%"/>'
                     f'<figcaption style="font-size:11px">{html.escape(p.name)}<br>{cap}</figcaption></figure>')
    display(HTML("".join(parts)))


def drive_mount(root: str | Path | None = None) -> Path:
    """Mount Google Drive in the kernel and create the SDXLR folders."""
    from sdxlr import drive
    return drive.mount(root or drive.ROOT)


def hf_token() -> str | None:
    load_secrets()
    return os.environ.get("HF_TOKEN")


if __name__ == "__main__":
    run(sys.argv[1:])
