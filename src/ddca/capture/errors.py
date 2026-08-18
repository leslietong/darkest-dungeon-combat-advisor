"""Typed errors for window discovery and screenshot capture."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ddca.capture.windows import WindowInfo


class CaptureError(Exception):
    """Base error for read-only window capture failures."""


class WindowNotFoundError(CaptureError):
    """No visible titled window matched the requested title."""


class AmbiguousWindowError(CaptureError):
    """More than one window matched the requested title."""

    def __init__(self, query: str, matches: Sequence[WindowInfo]) -> None:
        self.query = query
        self.matches = list(matches)
        details = "\n".join(
            "  hwnd={hwnd} title={title!r} pos=({left},{top}) size={width}x{height}".format(
                hwnd=window.hwnd,
                title=window.title,
                left=window.left,
                top=window.top,
                width=window.width,
                height=window.height,
            )
            for window in matches
        )
        super().__init__(
            f"Multiple windows match {query!r}, so capture was not started.\n"
            f"Matching windows:\n{details}\n"
            "Use a more specific --title, or list exact titles with:\n"
            "  python -m ddca.capture.cli list"
        )


class MinimizedWindowError(CaptureError):
    """The matched window is minimized and must not be captured."""


class InvalidWindowGeometryError(CaptureError):
    """The matched window has a zero-size or otherwise invalid rectangle."""


class EmptyCaptureError(CaptureError):
    """The screenshot buffer was empty after capture."""


class ScreenshotWriteError(CaptureError):
    """Saving the PNG or metadata file failed."""
