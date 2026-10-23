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


def run(args: list[str], cwd: str | Path = REPO, progress_every: float = 5.0) -> None:
    """Run `python args...` in the venv and stream its output into the cell.

    tqdm redraws with '\\r'; those updates are printed at most every
    `progress_every` seconds so the cell does not fill with progress bars.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
           "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": str(REPO / "src")}
    proc = subprocess.Popen([VENV_PY, "-u", *map(str, args)], cwd=cwd, env=env,
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


SECRETS = ("HF_TOKEN", "CIVITAI_TOKEN")


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


def hf_token() -> str | None:
    load_secrets()
    return os.environ.get("HF_TOKEN")


if __name__ == "__main__":
    run(sys.argv[1:])
