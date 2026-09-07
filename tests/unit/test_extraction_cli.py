"""CLI tests for static ROI extraction. Synthetic images only."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from ddca.vision.calibration import REQUIRED_REGIONS
from ddca.vision.extraction import MANIFEST_FILENAME
from ddca.vision.extraction_cli import main


def _write_synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :] = (7, 8, 9)
    image_path = tmp_path / "source.png"
    cv2.imwrite(str(image_path), image)
    payload = {
        "schema_version": 1,
        "profile_id": "extract-cli",
        "game": "darkest_dungeon_1",
        "language": "en",
        "display_mode": "windowed",
        "coordinate_space": "captured_window_normalized",
        "reference_size": {"width": 60, "height": 40},
        "regions": {
            name: {"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}
            for name in REQUIRED_REGIONS
        },
    }
    yaml_path = tmp_path / "profile.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return image_path, yaml_path


def test_cli_success_writes_manifest_and_keeps_input(tmp_path: Path, capsys) -> None:
    image_path, yaml_path = _write_synthetic_inputs(tmp_path)
    before = image_path.read_bytes()
    output_dir = tmp_path / "out"
    exit_code = main(
        [
            "--image",
            str(image_path),
            "--profile",
            str(yaml_path),
            "--output-dir",
            str(output_dir),
            "--region",
            "game_frame",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert (output_dir / MANIFEST_FILENAME).is_file()
    assert str((output_dir / MANIFEST_FILENAME).resolve()) in captured.out
    assert "Extracted 1 region" in captured.out
    assert image_path.read_bytes() == before
    payload = json.loads((output_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert payload["frame"]["source_image_name"] == image_path.name
    assert str(image_path.resolve()) not in json.dumps(payload)
    assert ":\\" not in json.dumps(payload)


def test_cli_duplicate_regions_keep_first_seen_order(tmp_path: Path) -> None:
    image_path, yaml_path = _write_synthetic_inputs(tmp_path)
    output_dir = tmp_path / "out"
    exit_code = main(
        [
            "--image",
            str(image_path),
            "--profile",
            str(yaml_path),
            "--output-dir",
            str(output_dir),
            "--region",
            "round_counter",
            "--region",
            "game_frame",
            "--region",
            "round_counter",
        ]
    )
    assert exit_code == 0
    payload = json.loads((output_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert [item["region_name"] for item in payload["regions"]] == ["round_counter", "game_frame"]


def test_cli_unknown_region_is_nonzero(tmp_path: Path) -> None:
    image_path, yaml_path = _write_synthetic_inputs(tmp_path)
    exit_code = main(
        [
            "--image",
            str(image_path),
            "--profile",
            str(yaml_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--region",
            "minimap",
        ]
    )
    assert exit_code == 1


def test_cli_missing_image_is_nonzero(tmp_path: Path) -> None:
    _image_path, yaml_path = _write_synthetic_inputs(tmp_path)
    exit_code = main(
        [
            "--image",
            str(tmp_path / "missing.png"),
            "--profile",
            str(yaml_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert exit_code == 1
