"""Unit tests for normalized and pixel rectangles."""

from __future__ import annotations

import pytest

from ddca.vision.errors import InvalidCalibrationError
from ddca.vision.geometry import NormalizedRect, PixelRect


def test_pixel_round_trip_at_reference_size() -> None:
    pixel = PixelRect(left=8, top=30, width=1095, height=616)

    normalized = pixel.to_normalized(1111, 654)
    restored = normalized.to_pixel(1111, 654)

    assert restored == pixel


def test_nest_maps_local_rect_into_parent() -> None:
    parent = NormalizedRect(x=0.10, y=0.20, width=0.50, height=0.40)
    local = NormalizedRect(x=0.10, y=0.25, width=0.20, height=0.50)

    nested = parent.nest(local)

    assert nested.x == pytest.approx(0.15)
    assert nested.y == pytest.approx(0.30)
    assert nested.width == pytest.approx(0.10)
    assert nested.height == pytest.approx(0.20)


def test_normalized_rect_rejects_overflow() -> None:
    rect = NormalizedRect(x=0.8, y=0.0, width=0.3, height=1.0)

    with pytest.raises(InvalidCalibrationError, match="outside 0-1"):
        rect.validate(context="test")


def test_normalized_rect_rejects_empty() -> None:
    rect = NormalizedRect(x=0.1, y=0.1, width=0.0, height=0.2)

    with pytest.raises(InvalidCalibrationError, match="must be positive"):
        rect.validate(context="test")


def test_pixel_rect_rejects_overflow() -> None:
    rect = PixelRect(left=100, top=0, width=50, height=10)

    with pytest.raises(InvalidCalibrationError, match="does not fit"):
        rect.to_normalized(120, 10)


def test_contains_accepts_nested_rect() -> None:
    parent = NormalizedRect(x=0.10, y=0.20, width=0.50, height=0.40)
    inner = NormalizedRect(x=0.15, y=0.25, width=0.10, height=0.10)

    assert parent.contains(inner)
    assert not inner.contains(parent)


def test_numpy_slices_match_exclusive_bounds() -> None:
    rect = PixelRect(left=8, top=30, width=10, height=5)
    rows, cols = rect.numpy_slices()

    assert rows == slice(30, 35)
    assert cols == slice(8, 18)
