"""Read-only Windows window capture for Darkest Dungeon Combat Advisor."""

from ddca.capture.errors import (
    AmbiguousWindowError,
    CaptureError,
    EmptyCaptureError,
    InvalidWindowGeometryError,
    MinimizedWindowError,
    ScreenshotWriteError,
    WindowNotFoundError,
)
from ddca.capture.screenshot import (
    CaptureMetadata,
    CaptureResult,
    capture_and_save,
    capture_window,
    save_capture,
)
from ddca.capture.windows import (
    WindowInfo,
    assert_window_capturable,
    find_window,
    list_visible_windows,
)

__all__ = [
    "AmbiguousWindowError",
    "CaptureError",
    "CaptureMetadata",
    "CaptureResult",
    "EmptyCaptureError",
    "InvalidWindowGeometryError",
    "MinimizedWindowError",
    "ScreenshotWriteError",
    "WindowInfo",
    "WindowNotFoundError",
    "assert_window_capturable",
    "capture_and_save",
    "capture_window",
    "find_window",
    "list_visible_windows",
    "save_capture",
]
