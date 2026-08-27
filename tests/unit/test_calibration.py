"""Unit tests for YAML calibration loading and region mapping."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from ddca.vision.calibration import (
    REQUIRED_REGIONS,
    crop_region,
    load_calibration,
    parse_calibration,
)
from ddca.vision.errors import InvalidCalibrationError, UnknownRegionError
from ddca.vision.geometry import NormalizedRect, PixelRect

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_PROFILE = REPO_ROOT / "configs" / "calibration" / "windowed_1111x654.yaml"


def _minimal_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "profile_id": "test",
        "game": "darkest_dungeon_1",
        "language": "en",
        "display_mode": "windowed",
        "coordinate_space": "captured_window_normalized",
        "reference_size": {"width": 100, "height": 100},
        "notes": "unit test",
        "regions": {
            name: {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
            for name in REQUIRED_REGIONS
        },
    }
    payload.update(overrides)
    return payload


def test_bundled_profile_loads_and_matches_known_game_frame() -> None:
    profile = load_calibration(BUNDLED_PROFILE)

    assert profile.reference_width == 1111
    assert profile.reference_height == 654
    assert set(REQUIRED_REGIONS) <= set(profile.regions)
    game_frame = profile.pixel_region("game_frame", 1111, 654)
    assert (game_frame.left, game_frame.top, game_frame.width, game_frame.height) == (
        8,
        30,
        1095,
        616,
    )


def test_relative_to_is_resolved_into_image_space() -> None:
    regions = {
        name: {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
        for name in REQUIRED_REGIONS
    }
    regions["game_frame"] = {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
    regions["heroes"] = {
        "relative_to": "game_frame",
        "x": 0.1,
        "y": 0.2,
        "width": 0.5,
        "height": 0.4,
    }
    for rank_name, x in (
        ("hero_rank_4", 0.0),
        ("hero_rank_3", 0.25),
        ("hero_rank_2", 0.5),
        ("hero_rank_1", 0.75),
    ):
        regions[rank_name] = {
            "relative_to": "heroes",
            "x": x,
            "y": 0.0,
            "width": 0.25,
            "height": 1.0,
        }
    profile = parse_calibration(_minimal_payload(regions=regions), source="memory")

    heroes = profile.region("heroes")
    assert heroes.x == pytest.approx(0.1)
    assert heroes.width == pytest.approx(0.5)
    hero_rank_1 = profile.region("hero_rank_1")
    assert hero_rank_1.x == pytest.approx(0.1 + 0.75 * 0.5)
    assert hero_rank_1.width == pytest.approx(0.25 * 0.5)


def test_missing_required_region() -> None:
    payload = _minimal_payload()
    regions = dict(payload["regions"])  # type: ignore[arg-type]
    del regions["action_bar"]
    payload["regions"] = regions

    with pytest.raises(InvalidCalibrationError, match="missing required regions: action_bar"):
        parse_calibration(payload, source="memory")


def test_relative_to_cycle_is_rejected() -> None:
    payload = _minimal_payload()
    regions = dict(payload["regions"])  # type: ignore[arg-type]
    regions["heroes"] = {
        "relative_to": "combat_area",
        "x": 0.0,
        "y": 0.0,
        "width": 1.0,
        "height": 1.0,
    }
    regions["combat_area"] = {
        "relative_to": "heroes",
        "x": 0.0,
        "y": 0.0,
        "width": 1.0,
        "height": 1.0,
    }
    payload["regions"] = regions

    with pytest.raises(InvalidCalibrationError, match="cycle"):
        parse_calibration(payload, source="memory")


def test_unknown_region_name() -> None:
    profile = parse_calibration(_minimal_payload(), source="memory")

    with pytest.raises(UnknownRegionError, match="Unknown region"):
        profile.region("minimap")


def test_size_warning_for_unvalidated_resolution() -> None:
    profile = parse_calibration(_minimal_payload(), source="memory")

    assert profile.size_warnings(100, 100) == []
    warnings = profile.size_warnings(1920, 1080)
    assert len(warnings) == 1
    assert "1920x1080" in warnings[0]


def test_crop_region_returns_expected_shape() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[5:10, 2:8] = (1, 2, 3)
    from ddca.vision.geometry import PixelRect

    cropped = crop_region(image, PixelRect(left=2, top=5, width=6, height=5))
    assert cropped.shape == (5, 6, 3)
    assert np.array_equal(cropped[0, 0], np.array([1, 2, 3], dtype=np.uint8))


def test_load_calibration_rejects_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(": this is not valid: [", encoding="utf-8")

    with pytest.raises(InvalidCalibrationError, match="Invalid YAML"):
        load_calibration(path)


def test_load_calibration_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InvalidCalibrationError, match="Could not read"):
        load_calibration(tmp_path / "missing.yaml")


def test_yaml_round_trip_parse(tmp_path: Path) -> None:
    payload = _minimal_payload()
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    profile = load_calibration(path)
    assert profile.profile_id == "test"
    assert "action_bar" in profile.regions
    assert "skill_bar" not in profile.regions
