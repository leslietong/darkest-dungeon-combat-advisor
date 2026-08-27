"""Unit tests for vision CLI helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ddca.vision.cli import cmd_from_pixels, cmd_validate, main
from ddca.vision.preview import save_calibration_preview
from ddca.vision.calibration import REQUIRED_REGIONS, parse_calibration


def test_from_pixels_prints_normalized_fields(capsys) -> None:
    exit_code = cmd_from_pixels(8, 30, 1095, 616, 1111, 654)

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "x: 0.007201" in captured
    assert "y: 0.045872" in captured
    assert "width: 0.985599" in captured
    assert "height: 0.941896" in captured


def test_validate_command_on_bundled_profile(capsys) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    yaml_path = repo_root / "configs" / "calibration" / "windowed_1111x654.yaml"

    exit_code = cmd_validate(yaml_path)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "windowed_1111x654" in output
    assert "action_bar" in output
    assert "move_action_slot" in output
    assert "skill_bar" not in output


def test_main_preview_writes_output(tmp_path: Path, capsys) -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    image_path = tmp_path / "shot.png"
    yaml_path = tmp_path / "profile.yaml"
    output_path = tmp_path / "out.png"

    payload = {
        "schema_version": 1,
        "profile_id": "cli-test",
        "game": "darkest_dungeon_1",
        "language": "en",
        "display_mode": "windowed",
        "coordinate_space": "captured_window_normalized",
        "reference_size": {"width": 80, "height": 60},
        "regions": {
            name: {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
            for name in REQUIRED_REGIONS
        },
    }
    profile = parse_calibration(payload, source="memory")
    save_calibration_preview(image, profile, image_path)

    import yaml

    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    exit_code = main(
        [
            "preview",
            "--image",
            str(image_path),
            "--calibration",
            str(yaml_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.is_file()
    assert str(output_path.resolve()) in capsys.readouterr().out


def test_strict_size_mismatch_fails(tmp_path: Path) -> None:
    image = np.zeros((50, 70, 3), dtype=np.uint8)
    image_path = tmp_path / "shot.png"
    yaml_path = tmp_path / "profile.yaml"
    payload = {
        "schema_version": 1,
        "profile_id": "cli-test",
        "game": "darkest_dungeon_1",
        "language": "en",
        "display_mode": "windowed",
        "coordinate_space": "captured_window_normalized",
        "reference_size": {"width": 80, "height": 60},
        "regions": {
            name: {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
            for name in REQUIRED_REGIONS
        },
    }
    import yaml
    from ddca.vision.calibration import parse_calibration
    from ddca.vision.preview import save_calibration_preview as save

    save(image, parse_calibration(payload, source="memory"), image_path)
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    exit_code = main(
        [
            "preview",
            "--image",
            str(image_path),
            "--calibration",
            str(yaml_path),
            "--output",
            str(tmp_path / "out.png"),
            "--strict-size",
        ]
    )
    assert exit_code == 1
