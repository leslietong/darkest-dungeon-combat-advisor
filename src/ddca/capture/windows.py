"""Enumerate and locate visible Windows desktop windows without modifying them."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Sequence
from ctypes import wintypes
from dataclasses import dataclass

from ddca.capture.errors import (
    AmbiguousWindowError,
    CaptureError,
    InvalidWindowGeometryError,
    MinimizedWindowError,
    WindowNotFoundError,
)

try:
    import win32gui
except ImportError as exc:  # pragma: no cover - exercised only when pywin32 is missing
    raise ImportError(
        "pywin32 is required to enumerate Windows desktop windows. "
        'Install this project with: python -m pip install -e ".[dev]"'
    ) from exc

_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87
_E_ACCESSDENIED = 0x80070005
_PROCESS_PER_MONITOR_DPI_AWARE = 2
_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)


@dataclass(frozen=True)
class WindowInfo:
    """Descriptor for one visible top-level desktop window."""

    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int
    minimized: bool

    @property
    def right(self) -> int:
        """Exclusive right edge in screen pixels."""

        return self.left + self.width

    @property
    def bottom(self) -> int:
        """Exclusive bottom edge in screen pixels."""

        return self.top + self.height


def ensure_process_dpi_aware() -> None:
    """Make this process DPI-aware so window rectangles match physical pixels.

    Safe to call when DPI awareness has already been set. This function does not
    activate, focus, resize, or otherwise modify any window.
    """

    if sys.platform != "win32":
        raise OSError(
            "Window capture is only supported on Windows. "
            "This project targets Darkest Dungeon on Windows Steam."
        )

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    set_context = getattr(user32, "SetProcessDpiAwarenessContext", None)
    if set_context is not None:
        set_context.argtypes = [wintypes.HANDLE]
        set_context.restype = wintypes.BOOL
        ctypes.set_last_error(0)
        if set_context(_DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2):
            return
        last_error = ctypes.get_last_error()
        if last_error in (0, _ERROR_ACCESS_DENIED, _ERROR_INVALID_PARAMETER):
            return

    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        result = int(shcore.SetProcessDpiAwareness(_PROCESS_PER_MONITOR_DPI_AWARE))
        if result in (0, _E_ACCESSDENIED):
            return
    except (AttributeError, OSError):
        pass

    try:
        user32.SetProcessDPIAware.restype = wintypes.BOOL
        if user32.SetProcessDPIAware():
            return
    except (AttributeError, OSError):
        pass

    print(
        "Warning: could not set DPI awareness. Capture coordinates may be wrong "
        "on displays that use Windows scaling. Restart the terminal and retry.",
        file=sys.stderr,
    )


def list_visible_windows() -> list[WindowInfo]:
    """Return visible top-level windows that have a non-empty title."""

    ensure_process_dpi_aware()
    windows: list[WindowInfo] = []

    def _callback(hwnd: int, _: object) -> bool:
        if not win32gui.IsWindowVisible(hwnd):
            return True
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return True
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        windows.append(
            WindowInfo(
                hwnd=int(hwnd),
                title=title,
                left=int(left),
                top=int(top),
                width=int(right - left),
                height=int(bottom - top),
                minimized=bool(win32gui.IsIconic(hwnd)),
            )
        )
        return True

    try:
        win32gui.EnumWindows(_callback, None)
    except Exception as exc:
        raise CaptureError(
            f"Failed while enumerating desktop windows: {exc}. "
            "If the game is running elevated, start PowerShell as Administrator "
            "and run this command again. This tool never modifies the game window."
        ) from exc
    return windows


def find_window(
    title: str,
    windows: Sequence[WindowInfo] | None = None,
) -> WindowInfo:
    """Locate one window by title.

    Matching is case-insensitive. An exact title match is preferred. If no exact
    match exists, a unique substring match is accepted. Multiple matches are an
    error; this function never silently picks one.
    """

    query = title.strip()
    if not query:
        raise WindowNotFoundError(
            "A non-empty window title is required. "
            "List visible titles with: python -m ddca.capture.cli list"
        )

    candidates = list(windows) if windows is not None else list_visible_windows()
    folded_query = query.casefold()
    exact_matches = [window for window in candidates if window.title.casefold() == folded_query]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise AmbiguousWindowError(query, exact_matches)

    substring_matches = [
        window for window in candidates if folded_query in window.title.casefold()
    ]
    if len(substring_matches) == 1:
        return substring_matches[0]
    if len(substring_matches) > 1:
        raise AmbiguousWindowError(query, substring_matches)

    raise WindowNotFoundError(
        f"No visible window matched title {query!r}. "
        "Start Darkest Dungeon so its window is visible, then run "
        "`python -m ddca.capture.cli list` and pass an exact --title from that list."
    )


def assert_window_capturable(window: WindowInfo) -> None:
    """Reject minimized or zero-size windows without changing them."""

    if window.minimized:
        raise MinimizedWindowError(
            f"Window {window.title!r} (hwnd={window.hwnd}) is minimized. "
            "Restore it so the game is visible on screen, then retry. "
            "This tool does not restore, focus, or capture minimized windows."
        )
    if window.width <= 0 or window.height <= 0:
        raise InvalidWindowGeometryError(
            f"Window {window.title!r} (hwnd={window.hwnd}) has an invalid "
            f"rectangle ({window.width}x{window.height}). "
            "Restore the window to a normal visible size and retry."
        )
