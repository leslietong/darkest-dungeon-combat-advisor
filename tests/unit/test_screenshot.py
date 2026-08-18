"""Unit tests for screenshot conversion, saving, and metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ddca.capture.errors import EmptyCaptureError, ScreenshotWriteError
from ddca.capture.screenshot import (
    CaptureMetadata,
    bgra_to_bgr,
    capture_window,
    save_capture,
)
from ddca.capture.windows import WindowInfo

FIXED_CLOCK = datetime(2026, 8, 18, 19, 50, 0, tzinfo=timezone.utc)


def make_window(**overrides: object) -> WindowInfo:
    values: dict[str, object] = {
        "hwnd": 42,
        "title": "Darkest Dungeon",
        "left": 8,
        "top": 16,
        "width": 4,
        "height": 2,
        "minimized": False,
    }
    values.update(overrides)
    return WindowInfo(
        hwnd=int(values["hwnd"]),
        title=str(values["title"]),
        left=int(values["left"]),
        top=int(values["top"]),
        width=int(values["width"]),
        height=int(values["height"]),
        minimized=bool(values["minimized"]),
    )


def test_bgra_to_bgr_conversion() -> None:
    bgra = np.zeros((2, 3, 4), dtype=np.uint8)
    bgra[..., 0] = 10
    bgra[..., 1] = 20
    bgra[..., 2] = 30
    bgra[..., 3] = 255

    bgr = bgra_to_bgr(bgra)

    assert bgr.shape == (2, 3, 3)
    assert bgr.dtype == np.uint8
    assert np.array_equal(bgr[..., 0], np.full((2, 3), 10, dtype=np.uint8))
    assert np.array_equal(bgr[..., 1], np.full((2, 3), 20, dtype=np.uint8))
    assert np.array_equal(bgr[..., 2], np.full((2, 3), 30, dtype=np.uint8))
    assert bgr.flags["C_CONTIGUOUS"]


def test_bgra_to_bgr_rejects_wrong_channel_count() -> None:
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(EmptyCaptureError, match="Expected a BGRA screenshot"):
        bgra_to_bgr(bgr)


def test_capture_window_converts_mocked_mss_buffer() -> None:
    window = make_window()
    bgra = np.zeros((window.height, window.width, 4), dtype=np.uint8)
    bgra[..., 0] = 1
    bgra[..., 1] = 2
    bgra[..., 2] = 3
    bgra[..., 3] = 255

    bgr = capture_window(window, grab_bgra=lambda region: bgra)

    assert bgr.shape == (2, 4, 3)
    assert np.array_equal(bgr[0, 0], np.array([1, 2, 3], dtype=np.uint8))


def test_capture_window_rejects_empty_image() -> None:
    window = make_window()
    empty = np.zeros((0, 4, 4), dtype=np.uint8)

    with pytest.raises(EmptyCaptureError, match="empty image"):
        capture_window(window, grab_bgra=lambda region: empty)


def test_screenshot_write_failure(tmp_path: Path) -> None:
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    window = make_window()

    with patch("ddca.capture.screenshot.cv2.imwrite", return_value=False):
        with pytest.raises(ScreenshotWriteError, match="OpenCV failed to write PNG"):
            save_capture(image, window, tmp_path, clock=lambda: FIXED_CLOCK)

    assert list(tmp_path.iterdir()) == []


def test_metadata_generation(tmp_path: Path) -> None:
    image = np.full((2, 4, 3), 7, dtype=np.uint8)
    window = make_window()

    result = save_capture(image, window, tmp_path, clock=lambda: FIXED_CLOCK)

    payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    expected = CaptureMetadata(
        title="Darkest Dungeon",
        handle=42,
        rectangle={"left": 8, "top": 16, "width": 4, "height": 2},
        image_dimensions={"width": 4, "height": 2},
        captured_at_utc="2026-08-18T19:50:00Z",
        capture_version="0.1.0",
    )
    assert payload == expected.to_dict()
    assert result.metadata == expected


def test_stable_timestamp_filenames_with_injected_clock(tmp_path: Path) -> None:
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    window = make_window()

    result = save_capture(image, window, tmp_path, clock=lambda: FIXED_CLOCK)

    assert result.image_path == tmp_path / "capture_20260818T195000Z.png"
    assert result.metadata_path == tmp_path / "capture_20260818T195000Z.json"
    assert result.image_path.is_file()
    assert result.image_path.stat().st_size > 0


def test_refuses_to_overwrite_existing_capture(tmp_path: Path) -> None:
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    window = make_window()
    save_capture(image, window, tmp_path, clock=lambda: FIXED_CLOCK)

    with pytest.raises(ScreenshotWriteError, match="Refusing to overwrite"):
        save_capture(image, window, tmp_path, clock=lambda: FIXED_CLOCK)
