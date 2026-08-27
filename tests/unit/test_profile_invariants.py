"""Regression tests for the bundled 1111x654 battle calibration profile."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ddca.vision.calibration import (
    REQUIRED_REGIONS,
    VALIDATION_CONFIRMED,
    VALIDATION_PROVISIONAL,
    load_calibration,
    parse_calibration,
)
from ddca.vision.cli import main
from ddca.vision.errors import InvalidCalibrationError
from ddca.vision.preview import save_calibration_preview

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_PROFILE = REPO_ROOT / "configs" / "calibration" / "windowed_1111x654.yaml"


def _profile():
    return load_calibration(BUNDLED_PROFILE)


def _pixel(name: str):
    profile = _profile()
    return profile.pixel_region(name, profile.reference_width, profile.reference_height)


def test_all_normalized_rects_stay_in_unit_interval() -> None:
    profile = _profile()
    for name, rect in profile.regions.items():
        rect.validate(context=name)
        assert 0.0 <= rect.x < 1.0
        assert 0.0 <= rect.y < 1.0
        assert rect.x + rect.width <= 1.000001
        assert rect.y + rect.height <= 1.000001


def test_resolved_pixel_rects_stay_inside_game_frame() -> None:
    profile = _profile()
    frame = profile.pixel_region("game_frame", 1111, 654)
    for name in profile.regions:
        if name == "game_frame":
            continue
        child = profile.pixel_region(name, 1111, 654)
        assert frame.contains(child), name


def test_game_frame_matches_confirmed_client_area() -> None:
    frame = _pixel("game_frame")
    assert (frame.left, frame.top, frame.width, frame.height) == (8, 30, 1095, 616)


def test_hero_ranks_run_left_to_right_from_four_to_one() -> None:
    lefts = [_pixel(f"hero_rank_{rank}").left for rank in (4, 3, 2, 1)]
    assert lefts == sorted(lefts)
    assert _pixel("hero_rank_4").left < _pixel("hero_rank_1").left


def test_enemy_ranks_run_left_to_right_from_one_to_four() -> None:
    lefts = [_pixel(f"enemy_rank_{rank}").left for rank in (1, 2, 3, 4)]
    assert lefts == sorted(lefts)
    assert _pixel("enemy_rank_1").left < _pixel("enemy_rank_4").left


def test_four_hero_health_regions_exist_in_left_to_right_rank_order() -> None:
    names = [f"hero_health_rank_{rank}" for rank in (4, 3, 2, 1)]
    assert all(name in _profile().regions for name in names)
    lefts = [_pixel(name).left for name in names]
    assert lefts == sorted(lefts)


def test_four_enemy_health_regions_exist_in_rank_order() -> None:
    names = [f"enemy_health_rank_{rank}" for rank in (1, 2, 3, 4)]
    assert all(name in _profile().regions for name in names)
    lefts = [_pixel(name).left for name in names]
    assert lefts == sorted(lefts)


def test_skill_slots_and_move_are_left_to_right_inside_action_bar() -> None:
    profile = _profile()
    action_bar = profile.region("action_bar")
    names = [
        "skill_slot_1",
        "skill_slot_2",
        "skill_slot_3",
        "skill_slot_4",
        "move_action_slot",
    ]
    lefts = []
    for name in names:
        slot = profile.region(name)
        assert action_bar.contains(slot), name
        pixel = _pixel(name)
        assert pixel.width > 0
        assert pixel.height > 0
        lefts.append(pixel.left)
    assert lefts == sorted(lefts)


def test_action_bar_replaces_skill_bar_and_includes_move() -> None:
    profile = _profile()
    assert "action_bar" in profile.regions
    assert "move_action_slot" in profile.regions
    assert "skill_bar" not in profile.regions
    action = _pixel("action_bar")
    assert (action.left, action.top, action.width, action.height) == (302, 458, 219, 48)
    move = _pixel("move_action_slot")
    assert (move.left, move.top, move.width, move.height) == (479, 461, 42, 42)
    assert move.right <= action.right
    expected_skills = {
        "skill_slot_1": (304, 461, 42, 42),
        "skill_slot_2": (348, 461, 42, 42),
        "skill_slot_3": (391, 461, 42, 42),
        "skill_slot_4": (435, 461, 42, 42),
    }
    for name, expected in expected_skills.items():
        pixel = _pixel(name)
        assert (pixel.left, pixel.top, pixel.width, pixel.height) == expected


def test_round_counter_is_separate_from_active_hero_regions() -> None:
    round_box = _pixel("round_counter")
    panel = _pixel("current_hero_panel")
    marker = _pixel("active_hero_marker")
    assert round_box.bottom <= panel.top
    assert round_box.bottom <= marker.top
    assert round_box.top < 140
    assert panel.top > 400


def test_provisional_regions_are_not_reported_as_confirmed() -> None:
    profile = _profile()
    assert "hero_status_icons" not in profile.regions
    assert "enemy_status_icons" not in profile.regions
    assert "turn_indicator" not in profile.regions
    assert profile.region_validation_status("active_hero_marker") == VALIDATION_PROVISIONAL
    assert profile.region_validation_status("enemy_health_rank_3") == VALIDATION_PROVISIONAL
    assert profile.region_validation_status("enemy_health_rank_4") == VALIDATION_PROVISIONAL
    assert not profile.is_visually_confirmed("active_hero_marker")
    assert profile.is_visually_confirmed("game_frame")
    assert profile.is_visually_confirmed("action_bar")
    assert profile.is_visually_confirmed("move_action_slot")
    assert profile.region_validation_status("round_counter") == VALIDATION_CONFIRMED


def test_preview_command_succeeds_with_strict_size(tmp_path: Path) -> None:
    image = np.zeros((654, 1111, 3), dtype=np.uint8)
    image_path = tmp_path / "shot.png"
    output_path = tmp_path / "preview.png"
    save_calibration_preview(image, _profile(), image_path)

    exit_code = main(
        [
            "preview",
            "--image",
            str(image_path),
            "--calibration",
            str(BUNDLED_PROFILE),
            "--output",
            str(output_path),
            "--strict-size",
        ]
    )
    assert exit_code == 0
    assert output_path.is_file()


def test_unknown_relative_to_is_rejected() -> None:
    payload = {
        "schema_version": 1,
        "profile_id": "bad-parent",
        "game": "darkest_dungeon_1",
        "language": "en",
        "display_mode": "windowed",
        "coordinate_space": "captured_window_normalized",
        "reference_size": {"width": 100, "height": 100},
        "regions": {
            name: {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
            for name in REQUIRED_REGIONS
        },
    }
    payload["regions"]["action_bar"] = {
        "relative_to": "missing_parent",
        "x": 0.0,
        "y": 0.0,
        "width": 1.0,
        "height": 1.0,
    }
    try:
        parse_calibration(payload, source="memory")
    except InvalidCalibrationError as exc:
        assert "relative_to" in str(exc)
        assert "missing_parent" in str(exc)
    else:
        raise AssertionError("expected InvalidCalibrationError")


def test_containment_violation_is_rejected() -> None:
    payload = {
        "schema_version": 1,
        "profile_id": "bad-containment",
        "game": "darkest_dungeon_1",
        "language": "en",
        "display_mode": "windowed",
        "coordinate_space": "captured_window_normalized",
        "reference_size": {"width": 100, "height": 100},
        "regions": {
            name: {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
            for name in REQUIRED_REGIONS
        },
    }
    payload["regions"]["action_bar"] = {
        "x": 0.0,
        "y": 0.0,
        "width": 0.2,
        "height": 0.2,
    }
    for slot_name in ("skill_slot_1", "skill_slot_2", "skill_slot_3", "skill_slot_4"):
        payload["regions"][slot_name] = {
            "relative_to": "action_bar",
            "x": 0.0,
            "y": 0.0,
            "width": 1.0,
            "height": 1.0,
        }
    payload["regions"]["move_action_slot"] = {
        "x": 0.7,
        "y": 0.7,
        "width": 0.2,
        "height": 0.2,
    }
    try:
        parse_calibration(payload, source="memory")
    except InvalidCalibrationError as exc:
        assert "not contained" in str(exc)
        assert "move_action_slot" in str(exc)
    else:
        raise AssertionError("expected InvalidCalibrationError")
