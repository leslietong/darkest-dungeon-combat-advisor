"""CLI tests for Phase 3B-2 action-slot availability. Synthetic images only."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np

from ddca.vision.action_state import ACTION_SLOT_REGIONS
from ddca.vision.action_state_cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "vision" / "action_slot_state_v1.yaml"
FRAME_ID = "c" * 64
CALIB_SHA = "d" * 64


def _write_extraction(tmp_path: Path) -> Path:
    root = tmp_path / "extraction"
    crops = root / "crops"
    crops.mkdir(parents=True)
    for index, name in enumerate(ACTION_SLOT_REGIONS):
        image = np.zeros((42, 42, 3), dtype=np.uint8)
        if name == "skill_slot_2":
            tone = ((np.indices((42, 42))[0] + np.indices((42, 42))[1]) % 70 + 35).astype(np.uint8)
            image[4:38, 4:38] = np.stack([tone[4:38, 4:38]] * 3, axis=-1)
        else:
            rows, cols = np.indices((34, 34))
            shade = 0.55 + 0.45 * ((rows + cols) % 9) / 8.0
            base = np.array((20, 160 + index * 10, 40), dtype=np.float64)
            image[4:38, 4:38] = np.clip(base * shade[..., None], 0, 255).astype(np.uint8)
        assert cv2.imwrite(str(crops / f"{name}.png"), image)
    regions = []
    for name in ACTION_SLOT_REGIONS:
        regions.append(
            {
                "region_name": name,
                "relative_to": "action_bar",
                "validation_status": "confirmed",
                "pixel": {"left": 0, "top": 0, "width": 42, "height": 42},
                "crop_path": f"crops/{name}.png",
                "width": 42,
                "height": 42,
                "evidence": {
                    "region_name": name,
                    "x": 0,
                    "y": 0,
                    "width": 42,
                    "height": 42,
                    "confidence": None,
                },
            }
        )
    payload = {
        "schema_version": 1,
        "frame": {
            "frame_id": FRAME_ID,
            "source_image_name": "source.png",
            "source_image_sha256": FRAME_ID,
            "source_image_width": 1111,
            "source_image_height": 654,
            "captured_at_utc": None,
        },
        "calibration": {"profile_id": "windowed_1111x654", "profile_sha256": CALIB_SHA},
        "extracted_at_utc": "2026-09-07T00:00:00Z",
        "regions": regions,
    }
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_cli_success_writes_report_and_table(tmp_path: Path, capsys) -> None:
    root = _write_extraction(tmp_path)
    output = tmp_path / "report.json"
    exit_code = main(
        [
            "--extraction-dir",
            str(root),
            "--config",
            str(DEFAULT_CONFIG),
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert output.is_file()
    assert "skill_slot_1" in captured.out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [item["slot"] for item in payload["classifications"]] == list(ACTION_SLOT_REGIONS)
    assert payload["classifications"][1]["state"] == "disabled"
    encoded = json.dumps(payload)
    assert str(root.resolve()) not in encoded
    hashes = [item["crop_sha256"] for item in payload["classifications"]]
    assert len(hashes) == 5
    assert all(re.fullmatch(r"^[0-9a-f]{64}$", value) for value in hashes)
    for item in payload["classifications"]:
        crop = root / "crops" / f"{item['slot']}.png"
        assert item["crop_sha256"] == hashlib.sha256(crop.read_bytes()).hexdigest()
        assert item["crop_sha256"][:8] in captured.out
        assert "normalized_margin" in item


def test_cli_missing_extraction_is_nonzero(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--extraction-dir",
            str(tmp_path / "missing"),
            "--config",
            str(DEFAULT_CONFIG),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert exit_code == 1


def test_cli_overwrite_required(tmp_path: Path) -> None:
    root = _write_extraction(tmp_path)
    output = tmp_path / "report.json"
    assert main(["--extraction-dir", str(root), "--config", str(DEFAULT_CONFIG), "--output", str(output)]) == 0
    assert main(["--extraction-dir", str(root), "--config", str(DEFAULT_CONFIG), "--output", str(output)]) == 1
    assert (
        main(
            [
                "--extraction-dir",
                str(root),
                "--config",
                str(DEFAULT_CONFIG),
                "--output",
                str(output),
                "--overwrite",
            ]
        )
        == 0
    )
