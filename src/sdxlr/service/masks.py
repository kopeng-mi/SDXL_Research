"""Painted masks: the upload route and the store behind it.

The mask painter (browser) POSTs a PNG to /api/mask/<session code>; it is
checked, stored as an 8-bit L PNG under WORK/uploads/masks/<code>/ and an id
comes back. Only that id travels afterwards: path_for() is the one place that
turns an id into a path, and it accepts nothing but the id format this module
makes, so no client text is ever used as a path.

The mask has the size of the display copy the painter showed; the worker
resizes it to its work copy (and refuses a different aspect).
"""

from __future__ import annotations

import io
import re
import time
import uuid
from pathlib import Path

from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request  # module level: FastAPI resolves the handler's annotations from here

from .. import session
from ..paths import UPLOADS

MASKS = UPLOADS / "masks"
MAX_BYTES = 8 * 2**20
MAX_SIDE = 4096
KEEP_PER_SESSION = 30
MAX_AGE_SECS = 24 * 3600
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
ID_RE = re.compile(r"^(\d{4})-([0-9a-f]{32})$")


class MaskError(ValueError):
    def __init__(self, status: int, msg: str) -> None:
        super().__init__(msg)
        self.status = status


def path_for(mask_id: str | None) -> Path | None:
    """The stored file of a mask id, or None when it is not an id this module made."""
    m = ID_RE.match(mask_id or "")
    return MASKS / m[1] / f"{m[2]}.png" if m else None


def store(code: str, data: bytes) -> str:
    """Validate a posted PNG and save it as 8-bit L; returns the mask id."""
    if not session.valid(code):
        raise MaskError(404, "unknown session")
    if len(data) > MAX_BYTES:
        raise MaskError(413, "the mask is too big")
    if not data.startswith(PNG_MAGIC):
        raise MaskError(415, "the mask must be a PNG")
    try:
        img = Image.open(io.BytesIO(data))  # reads the header only
        if max(img.size) > MAX_SIDE or min(img.size) < 1:  # before load(): a huge PNG is never decoded
            raise MaskError(422, f"the mask must be at most {MAX_SIDE} px per side")
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise MaskError(415, "the mask is not a readable PNG") from None
    # a painter canvas may arrive as RGB(A): its alpha carries the paint when it has any, else the brightness
    if img.mode in ("RGBA", "LA"):
        a = img.getchannel("A")
        img = a if a.getextrema() != (255, 255) else img.convert("L")
    else:
        img = img.convert("L")
    d = MASKS / code
    d.mkdir(parents=True, exist_ok=True)
    mask_id = f"{code}-{uuid.uuid4().hex}"
    img.save(d / f"{mask_id.split('-')[1]}.png")
    cleanup(code)
    return mask_id


def cleanup(code: str) -> None:
    """Keep the newest masks of a session and nothing older than a day."""
    files = sorted((MASKS / code).glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    now = time.time()
    for i, f in enumerate(files):
        if i >= KEEP_PER_SESSION or now - f.stat().st_mtime > MAX_AGE_SECS:
            f.unlink(missing_ok=True)


def register(app) -> None:
    """Add POST /api/mask/{code} to the NiceGUI app."""
    @app.post("/api/mask/{code}")
    async def mask_upload(code: str, request: Request):
        if not session.valid(code):
            raise HTTPException(404, "unknown session")
        try:
            declared = int(request.headers.get("content-length") or 0)
        except ValueError:
            raise HTTPException(400, "bad content-length") from None
        if declared > MAX_BYTES:
            raise HTTPException(413, "the mask is too big")
        buf = bytearray()
        async for chunk in request.stream():  # stop reading at the cap, never hold more than MAX_BYTES
            buf += chunk
            if len(buf) > MAX_BYTES:
                raise HTTPException(413, "the mask is too big")
        try:
            return {"id": await run_in_threadpool(store, code, bytes(buf))}
        except MaskError as e:
            raise HTTPException(e.status, str(e)) from None
