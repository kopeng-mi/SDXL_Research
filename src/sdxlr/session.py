"""4-digit session codes: the UI state saved on the VM, restored after a dropped connection.

Each browser gets a code (kept in the browser's localStorage, shown in the
header). Form values and the gallery are saved under it every few seconds, so
reopening the link restores everything automatically; typing the code restores
it on another device or a new link. Jobs run on the VM and keep going while
nobody is connected. Codes live on the VM only (a new VM starts fresh).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .paths import SESS

DIR = SESS


def _path(code: str) -> Path:
    return DIR / f"{code}.json"


def valid(code: str | None) -> bool:
    return bool(code) and len(code) == 4 and code.isdigit()


def new_code() -> str:
    DIR.mkdir(parents=True, exist_ok=True)
    taken = {p.stem for p in DIR.glob("*.json")}
    free = [f"{i:04d}" for i in range(1000, 10000) if f"{i:04d}" not in taken]
    code = random.choice(free)
    _path(code).write_text("{}")
    return code


def latest() -> str | None:
    """Most recently saved session code on this VM."""
    files = sorted(DIR.glob("*.json"), key=lambda p: p.stat().st_mtime) if DIR.exists() else []
    return files[-1].stem if files else None


def load(code: str) -> dict | None:
    p = _path(code) if valid(code) else None
    if not p or not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save(code: str, data: dict) -> None:
    if not valid(code):
        return
    DIR.mkdir(parents=True, exist_ok=True)
    tmp = _path(code).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
    tmp.replace(_path(code))  # atomic: a crash never leaves half a file


def update(code: str, **kv) -> None:
    d = load(code) or {}
    d.update(kv)
    save(code, d)
