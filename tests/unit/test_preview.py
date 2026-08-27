"""Unit tests for labeled calibration previews."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ddca.vision.calibration import REQUIRED_REGIONS, parse_calibration
from ddca.vision.errors import PreviewError
from ddca.vision.preview import draw_calibration_preview, save_calibration_preview


def _profile():
    payload = {
        "schema_version": 1,
        "profile_id": "preview-test",
        "game": "darkest_dungeon_1",
        "language": "en",
        "display_mode": "windowed",
        "coordinate_space": "captured_window_normalized",
        "reference_size": {"width": 80, "height": 60},
        "regions": {
            name: {"x": 0.1, "y": 0.2, "width": 0.4, "height": 0.3}
            for name in REQUIRED_REGIONS
        },
    }
    payload["regions"]["game_frame"] = {
        "x": 0.0,
        "y": 0.0,
        "width": 1.0,
        "height": 1.0,
    }
    return parse_calibration(payload, source="memory")


def test_preview_does_not_mutate_input_and_draws_rectangles() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    original = image.copy()
    profile = _profile()

    preview = draw_calibration_preview(image, profile)

    assert image.shape == preview.shape
    assert np.array_equal(image, original)
    assert not np.array_equal(preview, image)
    assert preview[0, 0].sum() > 0  # game_frame touches the origin


def test_save_preview_writes_png(tmp_path: Path) -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    output = tmp_path / "nested" / "preview.png"

    path = save_calibration_preview(image, _profile(), output)

    assert path == output
    assert output.is_file()
    assert output.stat().st_size > 0


def test_preview_rejects_non_bgr_image() -> None:
    gray = np.zeros((60, 80), dtype=np.uint8)

    with pytest.raises(PreviewError, match="BGR"):
        draw_calibration_preview(gray, _profile())
