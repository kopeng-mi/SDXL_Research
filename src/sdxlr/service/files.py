"""Cached listings of everything under WORK, refreshed in a background thread.

Rule: no UI timer or handler scans the disk. They read this module's cached
lists (newest-first history with metadata, LoRAs, checkpoints, datasets with
caption counts, recent prompts, thumbs pre-warmed); a thread refreshes them
every REFRESH_SECS, or at once when something calls poke(). Uploads, deletes
and job outputs show up within a refresh even if nobody clicks.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .. import history
from ..fetch import list_loras, list_models, list_zimage
from ..paths import DATASETS, LORAS, MODELS, OUTPUTS, RUNS, THUMBS, WORK, Z_MODELS, ZIMAGE

IMG = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
REFRESH_SECS = 2.5
FULL_RESCAN_SECS = 20.0  # rescan even when no directory changed (files edited in place)
SERVED = ("outputs", "thumbs", "compare")  # the only dirs under WORK the browser can fetch
PLACEHOLDER = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=="

_lock = threading.Lock()
_wake = threading.Event()
_poked = threading.Event()
_want: set[tuple[str, int]] = set()  # thumbs asked for before they existed: made by the refresh thread
_sig: list = [None, 0.0]  # directory signature and time of the last full scan
_started = threading.Event()
_cache: dict = {"history": [], "prompts": [], "loras": [], "models": [], "zimage": [], "datasets": []}


def _ensure() -> None:
    """First read before the thread ran: scan once synchronously (startup, not a timer)."""
    if not _started.is_set():
        _started.set()
        try:
            _refresh()
        except Exception:
            pass


# ---- cache access (what the UI reads) ----
def items(limit: int = 200) -> list[tuple[Path, dict]]:
    """Newest-first (png, meta) from History."""
    _ensure()
    with _lock:
        return _cache["history"][:limit]


def meta(path: str) -> dict | None:
    """The cached metadata of one output image (no disk read)."""
    p = str(path)
    with _lock:
        return next((m for q, m in _cache["history"] if str(q) == p), None)


def prompts(limit: int = 30) -> list[str]:
    with _lock:
        return _cache["prompts"][:limit]


def lora_names() -> list[str]:
    _ensure()
    with _lock:
        return [p.name for p in _cache["loras"]]


def lora_path(name: str) -> str | None:
    _ensure()
    with _lock:
        return next((str(p) for p in _cache["loras"] if p.name == name), None)


def model_names() -> list[str]:
    _ensure()
    with _lock:
        return [p.name for p in _cache["models"]]


def model_path(name: str) -> str:
    _ensure()
    with _lock:
        return next((str(p) for p in _cache["models"] if p.name == name), name)


def z_names() -> list[str]:
    _ensure()
    with _lock:
        return [*Z_MODELS, *(p.name for p in _cache["zimage"])]


def z_path(name: str) -> str:
    _ensure()
    with _lock:
        return Z_MODELS.get(name) or next((str(p) for p in _cache["zimage"] if p.name == name), name)


def datasets() -> list[str]:
    _ensure()
    with _lock:
        return list(_cache["datasets"])


def images(ds: str) -> list[Path]:
    _ensure()
    with _lock:
        return list(_cache.get("ds", {}).get(ds, []))


def ds_counts(ds: str) -> tuple[int, int]:
    """(photos, photos with a caption)."""
    _ensure()
    with _lock:
        return _cache.get("caps", {}).get(ds, (0, 0))


def caption(ds: str, name: str) -> str:
    """A photo's caption text (empty string when it has none)."""
    _ensure()
    with _lock:
        return _cache.get("cap_text", {}).get(f"{ds}/{name}", "")


def files_on_vm() -> list[tuple[str, Path]]:
    """Everything in the Library listing: (kind, path), cached."""
    _ensure()
    with _lock:
        return ([("SDXL checkpoint", p) for p in _cache["models"]]
                + [("Z-Image model", p) for p in _cache["zimage"]]
                + [("LoRA", p) for p in _cache["loras"]])


def url(p) -> str:
    """Browser URL of a file under WORK (served at /work)."""
    return f"/work/{Path(p).relative_to(WORK).as_posix()}?v={int(Path(p).stat().st_mtime)}"


def _thumb_path(p: Path, size: int) -> Path:
    return THUMBS / Path(p).relative_to(WORK).with_suffix(f".{size}.jpg")


def thumb(p: Path, size: int = 320) -> str:
    """URL of a thumbnail. Never decodes an image here (this runs on the event loop): a missing
    thumb is queued for the refresh thread and the original shows until it is made."""
    t = _thumb_path(p, size)
    try:
        if t.stat().st_mtime >= Path(p).stat().st_mtime:
            return url(t)
    except OSError:
        pass
    with _lock:
        _want.add((str(p), size))
    _wake.set()
    return url(p) if Path(p).relative_to(WORK).parts[0] in SERVED else PLACEHOLDER


def _make_thumb(p: Path, t: Path, size: int) -> None:
    from PIL import Image
    t.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(p)
    im.draft("RGB", (size * 2, size * 2))  # JPEGs decode at a fraction of the size
    im = im.convert("RGB")
    im.thumbnail((size, size))
    part = t.with_name(t.name + ".part")  # the browser never sees a half-written thumb
    im.save(part, "JPEG", quality=85)
    os.replace(part, t)


def _make_missing(pairs) -> None:
    for p, size in pairs:
        try:
            t = _thumb_path(p, size)
            if not t.exists() or t.stat().st_mtime < Path(p).stat().st_mtime:
                _make_thumb(Path(p), t, size)
        except Exception:
            pass  # half-written or vanished file; the next pass retries


# ---- refresh ----
def poke() -> None:
    """Something changed on disk: rescan at once instead of waiting."""
    _poked.set()
    _wake.set()


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _signature() -> tuple:
    """Directory mtimes that change when a file is added or removed (one stat each, no listing of files)."""
    dirs = [OUTPUTS, LORAS, RUNS, MODELS, MODELS / "zimage", ZIMAGE, DATASETS]
    try:
        dirs += sorted((d for d in OUTPUTS.iterdir() if d.is_dir()), reverse=True)[:3]  # the newest day folders
        dirs += [d for d in RUNS.iterdir() if d.is_dir()] + [d for d in DATASETS.iterdir() if d.is_dir()]
    except OSError:
        pass
    return tuple((str(d), _mtime(d)) for d in dirs)


def _refresh(force: bool = False) -> None:
    sig, now = _signature(), time.time()
    if not force and sig == _sig[0] and now - _sig[1] < FULL_RESCAN_SECS:
        with _lock:
            pending = list(_want)
            _want.clear()
        _make_missing(pending)
        return
    _sig[0], _sig[1] = sig, now
    his = history.items(220)
    loras, models, zimage = list_loras(), list_models(), list_zimage()
    ds = {}
    caps = {}
    cap_text = {}
    if DATASETS.exists():
        for d in sorted(p for p in DATASETS.iterdir() if p.is_dir()):
            imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG)
            ds[d.name] = imgs
            n = 0
            for p in imgs:
                t = p.with_suffix(".txt")
                if t.exists():
                    n += 1
                    try:
                        cap_text[f"{d.name}/{p.name}"] = t.read_text(encoding="utf-8").strip()
                    except OSError:
                        pass
            caps[d.name] = (len(imgs), n)
    seen, prompts = set(), []
    for _, m in his:
        p = m.get("prompt_raw") or m.get("prompt", "")
        if p and p not in seen:
            seen.add(p)
            prompts.append(p)
        if len(prompts) >= 40:
            break
    with _lock:
        _cache.update(history=his, prompts=prompts, loras=loras, models=models, zimage=zimage,
                      datasets=sorted(ds), ds=ds, caps=caps, cap_text=cap_text)
        pending = list(_want)
        _want.clear()
    # thumbs for every listed image, newest first, made here and not on the event loop
    _make_missing(pending + [(p, size) for p, _ in his for size in (160, 320)]
                  + [(p, 320) for imgs in ds.values() for p in imgs])


def _refresh_loop() -> None:
    while True:
        try:
            _refresh(force=_poked.is_set())
            _poked.clear()
        except Exception:
            pass  # a half-written file or a vanishing dir; the next pass re-reads
        _wake.wait(REFRESH_SECS)
        _wake.clear()


def start_refresh() -> None:
    threading.Thread(target=_refresh_loop, daemon=True).start()


# direct scans stay available for one-off actions (import checks, sync)
def scan_images(ds: str) -> list[Path]:
    d = DATASETS / ds if ds else None
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG) if d and d.exists() else []
