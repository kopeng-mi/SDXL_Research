"""Private, fixed address through Tailscale: http://sdxlr:7860 on your tailnet.

Colab has no TUN device, so tailscaled runs in userspace networking mode:
connections to this node's tailnet address are forwarded to local ports, which
is all the app needs. The auth key comes from Colab Secrets (TS_AUTHKEY); use a
reusable + ephemeral key so old Colab nodes disappear by themselves and the
name stays "sdxlr". The static binaries are fetched once into /content/bin.
"""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

VERSION = "1.104.2"
BIN = Path("/content/bin")
SOCK = "/content/ts/tailscaled.sock"
HOSTNAME = "sdxlr"


def _binaries() -> tuple[Path, Path]:
    d, c = BIN / "tailscaled", BIN / "tailscale"
    if not (d.exists() and c.exists()):
        BIN.mkdir(parents=True, exist_ok=True)
        tgz = BIN / "ts.tgz"
        urllib.request.urlretrieve(f"https://pkgs.tailscale.com/stable/tailscale_{VERSION}_amd64.tgz", tgz)
        with tarfile.open(tgz) as t:
            for m in t.getmembers():
                if Path(m.name).name in ("tailscale", "tailscaled"):
                    m.name = Path(m.name).name
                    t.extract(m, BIN)
        tgz.unlink()
    return d, c


def cli(*args: str, timeout: float = 60) -> subprocess.CompletedProcess:
    return subprocess.run([str(BIN / "tailscale"), f"--socket={SOCK}", *args], capture_output=True, text=True,
                          timeout=timeout)


def up(port: int, authkey: str | None = None) -> str:
    """Join the tailnet as `sdxlr`; returns the address to open."""
    key = authkey or os.environ.get("TS_AUTHKEY")
    if not key:
        raise RuntimeError("TS_AUTHKEY is not set (add it in Colab Secrets)")
    d, _ = _binaries()
    Path(SOCK).parent.mkdir(parents=True, exist_ok=True)
    if cli("status", timeout=10).returncode != 0:  # daemon not running yet
        subprocess.Popen([str(d), "--tun=userspace-networking", "--state=mem:", f"--socket={SOCK}"],
                         stdout=open("/content/ts/tailscaled.log", "a"), stderr=subprocess.STDOUT,
                         start_new_session=True)
        for _ in range(30):
            time.sleep(1)
            if Path(SOCK).exists():
                break
    r = cli("up", f"--auth-key={key}", f"--hostname={HOSTNAME}", "--accept-dns=false", timeout=90)
    if r.returncode != 0:
        raise RuntimeError(f"tailscale up failed: {(r.stderr or r.stdout).strip()[-300:]}")
    st = json.loads(cli("status", "--json").stdout)
    dns = st["Self"].get("DNSName", "").rstrip(".")
    ip = (st["Self"].get("TailscaleIPs") or ["?"])[0]
    name = dns.split(".")[0] if dns else HOSTNAME
    return f"http://{name}:{port}  (or http://{ip}:{port}, full name {dns})"
