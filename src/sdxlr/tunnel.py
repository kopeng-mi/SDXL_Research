"""Public link through a Cloudflare quick tunnel (no account; trycloudflare.com).

WebSockets work through it (the UI's live updates); the URL changes every launch
and anyone with it can open the app. The binary is fetched once into
/content/bin. Port 8080 is refused: on Colab that is the notebook server itself.
"""

from __future__ import annotations

import re
import subprocess
import threading
import urllib.request
from pathlib import Path

VERSION = "2026.10.3"
BIN = Path("/content/bin/cloudflared")
URL = f"https://github.com/cloudflare/cloudflared/releases/download/{VERSION}/cloudflared-linux-amd64"


def _binary() -> Path:
    if not BIN.exists():
        BIN.parent.mkdir(parents=True, exist_ok=True)
        tmp = BIN.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.chmod(0o755)
        tmp.replace(BIN)
    return BIN


def start(port: int, on_url) -> subprocess.Popen:
    """Start the tunnel; on_url(url) is called once the public URL is known."""
    if port == 8080:
        raise ValueError("port 8080 is Colab's notebook server; use another port")
    proc = subprocess.Popen([str(_binary()), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def read():
        sent = False
        for line in proc.stdout:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m and not sent:
                sent = True
                on_url(m[0])

    threading.Thread(target=read, daemon=True).start()
    return proc
