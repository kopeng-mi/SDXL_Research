"""The mask painter: a browser-side canvas over the image (see mask_painter.js).

Python sets the tool and brush; the browser keeps the mask. `export()` posts it
to /api/mask/<session> and returns the stored id (or raises with a plain message).
"""

from __future__ import annotations

from nicegui import ui


class MaskPainter(ui.element, component="mask_painter.js"):
    def __init__(self, src: str, tool: str = "brush", size: int = 48, mask_url: str | None = None,
                 on_changed=None, on_device=None) -> None:
        super().__init__()
        self._props.update(src=src, tool=tool, size=size, maskurl=mask_url or "")
        if on_changed:
            self.on("changed", lambda e: on_changed(e.args))
        if on_device:
            self.on("device", lambda e: on_device(e.args))

    def set(self, **props) -> None:
        """Change tool / size without redrawing the element."""
        changed = {k: v for k, v in props.items() if self._props.get(k) != v}
        if changed:
            self._props.update(changed)
            self.update()

    def undo(self) -> None:
        self.run_method("undo")

    def clear(self) -> None:
        self.run_method("clear")

    def invert(self) -> None:
        self.run_method("invert")

    async def export(self, code: str) -> str:
        """Send the mask to the app; the stored id. Raises ValueError (plain message) on any failure."""
        r = await self.run_method("export", f"/api/mask/{code}", timeout=30)
        if not r:
            raise ValueError("The page did not answer. Reload it and paint again.")
        if r.get("error"):
            raise ValueError(r["error"])
        return r["id"]
