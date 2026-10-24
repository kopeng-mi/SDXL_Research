"""Before/after slider (client side, see before_after.js)."""

from __future__ import annotations

from nicegui import ui


class BeforeAfter(ui.element, component="before_after.js"):
    def __init__(self, before: str, after: str) -> None:
        super().__init__()
        self._props.update(before=before, after=after)
