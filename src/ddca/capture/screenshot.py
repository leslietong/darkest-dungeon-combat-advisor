"""Capture a visible window rectangle as an OpenCV BGR image and save metadata."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mss
import mss.exception
import numpy as np
from numpy.typing import NDArray

from ddca import __version__
from ddca.capture.errors import EmptyCaptureError, ScreenshotWriteError
from ddca.capture.windows import WindowInfo, assert_window_capturable

try:
    import cv2
except ImportError as exc:  # pragma: no cover - exercised only when OpenCV is missing
    raise ImportError(
        "opencv-python is required to save PNG screenshots. "
        'Install this project with: python -m pip install -e ".[dev]"'
    ) from exc

CAPTURE_VERSION = __version__
_Region = dict[str, int]
_Grabber = Callable[[_Region], NDArray[np.uint8]]
_Clock = Callable[[], datetime]


@dataclass(frozen=True)
class CaptureMetadata:
    """Sidecar metadata written next to a captured PNG."""

    title: str
    handle: int
    rectangle: dict[str, int]
    image_dimensions: dict[str, int]
    captured_at_utc: str
    capture_version: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class CaptureResult:
    """Paths and metadata produced by one screenshot save."""

    image_path: Path
    metadata_path: Path
    metadata: CaptureMetadata
    image: NDArray[np.uint8]


def bgra_to_bgr(bgra: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Convert an mss-style BGRA screenshot to an OpenCV BGR image."""

    if bgra.ndim != 3 or bgra.shape[2] != 4:
        raise EmptyCaptureError(
            f"Expected a BGRA screenshot with shape (H, W, 4); received shape {bgra.shape}. "
            "Confirm the target window is visible and retry the capture."
        )
    return np.ascontiguousarray(bgra[:, :, :3])


def capture_window(
    window: WindowInfo,
    *,
    grab_bgra: _Grabber | None = None,
) -> NDArray[np.uint8]:
    """Capture `window` as a BGR NumPy array.

    The window is not activated, focused, moved, or resized.
    """

    assert_window_capturable(window)
    region: _Region = {
        "left": window.left,
        "top": window.top,
        "width": window.width,
        "height": window.height,
    }
    grab = grab_bgra if grab_bgra is not None else _grab_bgra_with_mss
    try:
        bgra = grab(region)
    except (OSError, ValueError, mss.exception.ScreenShotError) as exc:
        raise EmptyCaptureError(
            f"Failed to capture pixels from window {window.title!r} "
            f"(hwnd={window.hwnd}, left={window.left}, top={window.top}, "
            f"width={window.width}, height={window.height}): {exc}. "
            "Confirm the window is visible on this desktop and is not minimized."
        ) from exc

    bgr = bgra_to_bgr(np.asarray(bgra, dtype=np.uint8))
    if bgr.size == 0 or bgr.shape[0] <= 0 or bgr.shape[1] <= 0:
        raise EmptyCaptureError(
            f"Capture of window {window.title!r} (hwnd={window.hwnd}) produced an empty image. "
            "Restore the window so it is visible on a connected display and retry."
        )
    return bgr


def save_capture(
    image: NDArray[np.uint8],
    window: WindowInfo,
    output_dir: Path,
    *,
    clock: _Clock | None = None,
    capture_version: str = CAPTURE_VERSION,
) -> CaptureResult:
    """Write a PNG and matching JSON metadata using an injected or UTC clock."""

    if image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise EmptyCaptureError(
            f"Refusing to save an invalid BGR image with shape {getattr(image, 'shape', None)}. "
            "Capture the window again after restoring it to a visible size."
        )

    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ScreenshotWriteError(
            f"Could not create output directory {output_dir.resolve()}: {exc}. "
            "Choose a writable --output-dir and confirm you have permission to create folders there."
        ) from exc

    captured_at = _utc_now(clock)
    stem = captured_at.strftime("%Y%m%dT%H%M%SZ")
    image_path = output_dir / f"capture_{stem}.png"
    metadata_path = output_dir / f"capture_{stem}.json"
    if image_path.exists() or metadata_path.exists():
        raise ScreenshotWriteError(
            f"Refusing to overwrite existing capture files: {image_path} / {metadata_path}. "
            "Wait one second and retry, or choose another --output-dir."
        )

    write_ok = cv2.imwrite(str(image_path), image)
    if not write_ok:
        raise ScreenshotWriteError(
            f"OpenCV failed to write PNG {image_path.resolve()}. "
            "Check that the path is valid, the disk is not full, and you have write permission."
        )
    if not image_path.is_file() or image_path.stat().st_size <= 0:
        raise ScreenshotWriteError(
            f"OpenCV reported success but {image_path.resolve()} is missing or empty. "
            "Check disk space and antivirus locks, then retry."
        )

    metadata = CaptureMetadata(
        title=window.title,
        handle=int(window.hwnd),
        rectangle={
            "left": window.left,
            "top": window.top,
            "width": window.width,
            "height": window.height,
        },
        image_dimensions={
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        },
        captured_at_utc=_format_utc(captured_at),
        capture_version=capture_version,
    )
    try:
        metadata_path.write_text(
            json.dumps(metadata.to_dict(), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise ScreenshotWriteError(
            f"Wrote PNG {image_path.resolve()} but failed to write metadata "
            f"{metadata_path.resolve()}: {exc}. "
            "Check free disk space and write permission, then retry."
        ) from exc

    return CaptureResult(
        image_path=image_path,
        metadata_path=metadata_path,
        metadata=metadata,
        image=image,
    )


def capture_and_save(
    window: WindowInfo,
    output_dir: Path,
    *,
    clock: _Clock | None = None,
    grab_bgra: _Grabber | None = None,
) -> CaptureResult:
    """Capture one window and save a timestamped PNG plus JSON metadata."""

    image = capture_window(window, grab_bgra=grab_bgra)
    return save_capture(image, window, output_dir, clock=clock)


def _grab_bgra_with_mss(region: _Region) -> NDArray[np.uint8]:
    with mss.mss() as sct:
        shot = sct.grab(region)
    return np.asarray(shot, dtype=np.uint8)


def _utc_now(clock: _Clock | None) -> datetime:
    current = clock() if clock is not None else datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
