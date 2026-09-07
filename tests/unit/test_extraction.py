"""Unit tests for static ROI extraction. Synthetic images only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from ddca.vision.calibration import (
    COORDINATE_SPACE,
    REQUIRED_REGIONS,
    VALIDATION_CONFIRMED,
    VALIDATION_PROVISIONAL,
    CalibrationProfile,
    load_calibration,
)
from ddca.vision.errors import ExtractionError, UnknownRegionError
from ddca.vision.extraction import (
    EXTRACTION_SCHEMA_VERSION,
    ExtractionManifest,
    crop_filename,
    extract_regions,
    independent_crop,
    resolve_pixel_rect,
    sha256_file,
)
from ddca.vision.geometry import NormalizedRect

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLED_PROFILE = REPO_ROOT / "configs" / "calibration" / "windowed_1111x654.yaml"
REQUIRED_EXTRACT_NAMES: tuple[str, ...] = (
    "game_frame",
    "hero_rank_4",
    "hero_rank_3",
    "hero_rank_2",
    "hero_rank_1",
    "enemy_rank_1",
    "enemy_rank_2",
    "enemy_rank_3",
    "enemy_rank_4",
    "hero_health_rank_4",
    "hero_health_rank_3",
    "hero_health_rank_2",
    "hero_health_rank_1",
    "enemy_health_rank_1",
    "enemy_health_rank_2",
    "enemy_health_rank_3",
    "enemy_health_rank_4",
    "action_bar",
    "skill_slot_1",
    "skill_slot_2",
    "skill_slot_3",
    "skill_slot_4",
    "move_action_slot",
    "round_counter",
    "current_hero_panel",
    "active_hero_marker",
)
EXTRACTED_AT = datetime(2026, 9, 6, 23, 50, 0, tzinfo=timezone.utc)
CAPTURED_AT = datetime(2026, 8, 20, 3, 0, 16, tzinfo=timezone.utc)


def _patterned_image(width: int, height: int) -> np.ndarray:
    rows, cols = np.indices((height, width))
    return np.stack(
        [(cols % 256).astype(np.uint8), (rows % 256).astype(np.uint8), ((cols + rows) % 256).astype(np.uint8)],
        axis=-1,
    )


def _bundled_profile() -> CalibrationProfile:
    return load_calibration(BUNDLED_PROFILE)


def _tiny_profile(
    regions: dict[str, NormalizedRect],
    *,
    width: int = 20,
    height: int = 20,
    statuses: dict[str, str] | None = None,
) -> CalibrationProfile:
    return CalibrationProfile(
        schema_version=1,
        profile_id="extract-test",
        game="darkest_dungeon_1",
        language="en",
        display_mode="windowed",
        reference_width=width,
        reference_height=height,
        coordinate_space=COORDINATE_SPACE,
        notes="synthetic extractor test",
        regions=regions,
        validation_status={name: (statuses or {}).get(name, VALIDATION_CONFIRMED) for name in regions},
    )


def _write_png(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return path


def _hash_files(tmp_path: Path, image: np.ndarray, *, name: str = "source.png") -> tuple[Path, Path]:
    image_path = _write_png(tmp_path / name, image)
    yaml_path = tmp_path / "profile.yaml"
    if not yaml_path.exists():
        yaml_path.write_text("synthetic: true\n", encoding="utf-8")
    return image_path, yaml_path


def _extract(
    tmp_path: Path,
    image: np.ndarray,
    profile: CalibrationProfile,
    *,
    output_dir: Path | None = None,
    calibration_path: Path | None = None,
    source_name: str = "source.png",
    sidecar: dict[str, object] | None = None,
    **kwargs: object,
) -> ExtractionManifest:
    dest = output_dir or tmp_path / "out"
    files_dir = tmp_path / "inputs" / dest.name
    image_path, default_yaml = _hash_files(files_dir, image, name=source_name)
    if sidecar is not None:
        image_path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return extract_regions(
        image,
        profile,
        dest,
        source_image_path=image_path,
        calibration_path=calibration_path or default_yaml,
        extracted_at=kwargs.pop("extracted_at", EXTRACTED_AT),  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def test_bundled_profile_extracts_every_configured_region(tmp_path: Path) -> None:
    profile = _bundled_profile()
    image = _patterned_image(profile.reference_width, profile.reference_height)
    manifest = _extract(tmp_path, image, profile, calibration_path=BUNDLED_PROFILE)
    output_dir = tmp_path / "out"

    for name in REQUIRED_EXTRACT_NAMES:
        extracted = manifest.region(name)
        expected = resolve_pixel_rect(profile, name, image.shape[1], image.shape[0])
        assert extracted.pixel == expected
        assert extracted.width == expected.width
        assert extracted.height == expected.height
        crop = cv2.imread(str(output_dir / extracted.crop_path), cv2.IMREAD_COLOR)
        assert crop is not None
        assert crop.shape[0] == expected.height
        assert crop.shape[1] == expected.width
        source = image[expected.top : expected.bottom, expected.left : expected.right]
        assert np.array_equal(crop, source)
        assert extracted.validation_status == profile.region_validation_status(name)

    assert "active_hero_marker" in profile.provisional_region_names()
    assert manifest.region("active_hero_marker").validation_status == VALIDATION_PROVISIONAL
    assert manifest.region("enemy_health_rank_3").validation_status == VALIDATION_PROVISIONAL
    assert manifest.region("enemy_health_rank_4").validation_status == VALIDATION_PROVISIONAL
    assert manifest.region("game_frame").validation_status == VALIDATION_CONFIRMED


def test_nested_child_crop_matches_source_pixels_exactly(tmp_path: Path) -> None:
    profile = _bundled_profile()
    image = _patterned_image(profile.reference_width, profile.reference_height)
    manifest = _extract(
        tmp_path,
        image,
        profile,
        calibration_path=BUNDLED_PROFILE,
        selected_regions=("skill_slot_2",),
    )
    extracted = manifest.region("skill_slot_2")
    assert extracted.relative_to == "action_bar"
    source = image[
        extracted.pixel.top : extracted.pixel.bottom,
        extracted.pixel.left : extracted.pixel.right,
    ]
    crop = cv2.imread(str(tmp_path / "out" / extracted.crop_path), cv2.IMREAD_COLOR)
    assert np.array_equal(crop, source)
    assert crop.dtype == np.uint8


def test_no_resize_or_color_conversion(tmp_path: Path) -> None:
    profile = _tiny_profile({"panel": NormalizedRect(x=0.1, y=0.2, width=0.4, height=0.3)})
    image = _patterned_image(20, 20)
    before = image.copy()
    manifest = _extract(tmp_path, image, profile)
    extracted = manifest.region("panel")
    crop = cv2.imread(str(tmp_path / "out" / extracted.crop_path), cv2.IMREAD_UNCHANGED)
    assert crop.shape == (extracted.height, extracted.width, 3)
    assert np.array_equal(image, before)
    assert crop.dtype == image.dtype


def test_manifest_order_is_deterministic(tmp_path: Path) -> None:
    profile = _tiny_profile(
        {
            "zeta": NormalizedRect(0.0, 0.0, 0.5, 0.5),
            "alpha": NormalizedRect(0.5, 0.0, 0.5, 0.5),
        }
    )
    image = _patterned_image(20, 20)
    first = _extract(tmp_path, image, profile, output_dir=tmp_path / "a")
    second = _extract(tmp_path, image, profile, output_dir=tmp_path / "b")
    assert [item.region_name for item in first.regions] == ["alpha", "zeta"]
    assert [item.region_name for item in first.regions] == [item.region_name for item in second.regions]
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(second.to_dict(), sort_keys=True)


def test_manifest_json_round_trip_preserves_utc(tmp_path: Path) -> None:
    profile = _tiny_profile({"panel": NormalizedRect(0.0, 0.0, 1.0, 1.0)})
    image = _patterned_image(20, 20)
    original = _extract(tmp_path, image, profile)
    payload = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    restored = ExtractionManifest.from_dict(payload)
    assert restored == original
    assert restored.extracted_at_utc == EXTRACTED_AT
    assert restored.extracted_at_utc.tzinfo is timezone.utc
    assert payload["extracted_at_utc"] == "2026-09-06T23:50:00Z"
    assert restored.schema_version == EXTRACTION_SCHEMA_VERSION
    assert restored.region("panel").crop_path == "crops/panel.png"
    assert not Path(restored.region("panel").crop_path).is_absolute()


def test_selected_region_extraction(tmp_path: Path) -> None:
    profile = _bundled_profile()
    image = _patterned_image(profile.reference_width, profile.reference_height)
    manifest = _extract(
        tmp_path,
        image,
        profile,
        calibration_path=BUNDLED_PROFILE,
        selected_regions=("move_action_slot", "game_frame"),
    )
    assert [item.region_name for item in manifest.regions] == ["move_action_slot", "game_frame"]
    assert manifest.region("move_action_slot").relative_to == "action_bar"


def test_unknown_region_is_rejected(tmp_path: Path) -> None:
    profile = _tiny_profile({"panel": NormalizedRect(0.0, 0.0, 1.0, 1.0)})
    image = _patterned_image(20, 20)
    with pytest.raises(UnknownRegionError, match="minimap"):
        _extract(tmp_path, image, profile, selected_regions=("minimap",))


def test_wrong_image_dimensions_are_rejected(tmp_path: Path) -> None:
    profile = _tiny_profile({"panel": NormalizedRect(0.0, 0.0, 1.0, 1.0)}, width=20, height=20)
    image = _patterned_image(24, 20)
    with pytest.raises(ExtractionError, match="does not match the profile reference size"):
        _extract(tmp_path, image, profile)


def test_out_of_bounds_rectangle_is_not_clamped(tmp_path: Path) -> None:
    profile = _tiny_profile({"overflow": NormalizedRect(x=0.8, y=0.0, width=0.4, height=0.5)})
    image = _patterned_image(20, 20)
    with pytest.raises(ExtractionError, match="does not clamp"):
        _extract(tmp_path, image, profile)


def test_empty_geometry_is_rejected(tmp_path: Path) -> None:
    profile = _tiny_profile({"speck": NormalizedRect(x=0.0, y=0.0, width=0.001, height=0.001)})
    image = _patterned_image(20, 20)
    with pytest.raises(ExtractionError, match="empty"):
        _extract(tmp_path, image, profile)


def test_path_traversal_region_names_are_rejected(tmp_path: Path) -> None:
    profile = _tiny_profile({"panel": NormalizedRect(0.0, 0.0, 1.0, 1.0)})
    image = _patterned_image(20, 20)
    with pytest.raises(ExtractionError, match="safe crop filename"):
        _extract(tmp_path, image, profile, selected_regions=("../secret",))
    with pytest.raises(ExtractionError, match="safe crop filename"):
        crop_filename("skill/slot")


def test_overwrite_requires_explicit_flag(tmp_path: Path) -> None:
    profile = _tiny_profile({"panel": NormalizedRect(0.0, 0.0, 1.0, 1.0)})
    image = _patterned_image(20, 20)
    output_dir = tmp_path / "out"
    _extract(tmp_path, image, profile, output_dir=output_dir)
    stray = output_dir / "notes.txt"
    stray.write_text("keep me", encoding="utf-8")
    extra_crop = output_dir / "crops" / "unrelated.png"
    extra_crop.write_bytes(b"not-a-planned-crop")
    with pytest.raises(ExtractionError, match="Refusing to overwrite"):
        _extract(tmp_path, image, profile, output_dir=output_dir)
    _extract(tmp_path, image, profile, output_dir=output_dir, overwrite=True)
    assert stray.read_text(encoding="utf-8") == "keep me"
    assert extra_crop.read_bytes() == b"not-a-planned-crop"
    assert (output_dir / "crops" / "panel.png").is_file()


def test_source_array_is_not_modified(tmp_path: Path) -> None:
    profile = _tiny_profile({"panel": NormalizedRect(0.0, 0.0, 0.5, 0.5)})
    image = _patterned_image(20, 20)
    fingerprint = image.tobytes()
    _extract(tmp_path, image, profile)
    assert image.tobytes() == fingerprint


def test_manifest_contains_no_recognition_or_combat_state(tmp_path: Path) -> None:
    profile = _tiny_profile({"panel": NormalizedRect(0.0, 0.0, 1.0, 1.0)})
    image = _patterned_image(20, 20)
    manifest = _extract(tmp_path, image, profile)
    encoded = json.dumps(manifest.to_dict())
    assert "CombatState" not in encoded
    assert "ocr" not in encoded.lower()
    assert "win_probability" not in encoded
    assert manifest.region("panel").evidence.confidence is None
    assert b"ndarray" not in encoded.encode("ascii")


def test_pixel_rects_match_phase_2_mapping_without_clamp() -> None:
    profile = _bundled_profile()
    for name in REQUIRED_REGIONS + ("active_hero_marker",):
        expected = profile.pixel_region(name, profile.reference_width, profile.reference_height)
        actual = resolve_pixel_rect(profile, name, profile.reference_width, profile.reference_height)
        assert actual == expected, name


def test_source_and_calibration_sha256_and_stable_frame_id(tmp_path: Path) -> None:
    profile = _tiny_profile({"panel": NormalizedRect(0.0, 0.0, 1.0, 1.0)})
    image = _patterned_image(20, 20)
    first = _extract(tmp_path, image, profile, output_dir=tmp_path / "a")
    second = _extract(tmp_path, image, profile, output_dir=tmp_path / "b")
    image_path = tmp_path / "inputs" / "a" / "source.png"
    yaml_path = tmp_path / "inputs" / "a" / "profile.yaml"
    expected_image = hashlib.sha256(image_path.read_bytes()).hexdigest()
    expected_yaml = hashlib.sha256(yaml_path.read_bytes()).hexdigest()
    assert sha256_file(image_path) == expected_image
    assert first.frame.source_image_sha256 == expected_image
    assert first.frame.frame_id == expected_image
    assert first.calibration.profile_sha256 == expected_yaml
    assert first.calibration.profile_id == "extract-test"
    assert second.frame.frame_id == first.frame.frame_id
    assert second.calibration.profile_sha256 == first.calibration.profile_sha256


def test_captured_at_comes_only_from_matching_sidecar(tmp_path: Path) -> None:
    profile = _tiny_profile({"panel": NormalizedRect(0.0, 0.0, 1.0, 1.0)})
    image = _patterned_image(20, 20)
    matching = _extract(
        tmp_path,
        image,
        profile,
        output_dir=tmp_path / "match",
        sidecar={
            "image_dimensions": {"width": 20, "height": 20},
            "captured_at_utc": "2026-08-20T03:00:16Z",
        },
    )
    assert matching.frame.captured_at_utc == CAPTURED_AT
    assert matching.extracted_at_utc == EXTRACTED_AT
    assert matching.extracted_at_utc != matching.frame.captured_at_utc

    mismatched = _extract(
        tmp_path,
        image,
        profile,
        output_dir=tmp_path / "mismatch",
        sidecar={
            "image_dimensions": {"width": 1111, "height": 654},
            "captured_at_utc": "2026-08-20T03:00:16Z",
        },
    )
    assert mismatched.frame.captured_at_utc is None
    assert mismatched.extracted_at_utc == EXTRACTED_AT

    missing = _extract(tmp_path, image, profile, output_dir=tmp_path / "missing")
    assert missing.frame.captured_at_utc is None
    assert missing.extracted_at_utc == EXTRACTED_AT


def test_manifest_does_not_store_absolute_source_path(tmp_path: Path) -> None:
    profile = _tiny_profile({"panel": NormalizedRect(0.0, 0.0, 1.0, 1.0)})
    image = _patterned_image(20, 20)
    manifest = _extract(tmp_path, image, profile, source_name="battle.png")
    payload = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    encoded = json.dumps(payload)
    assert payload["frame"]["source_image_name"] == "battle.png"
    assert payload["frame"]["source_image_name"] == Path(payload["frame"]["source_image_name"]).name
    assert "calibration_path" not in payload
    assert "image_path" not in encoded
    assert str(tmp_path.resolve()) not in encoded
    assert ":\\" not in encoded
    assert "Users" not in encoded


def test_validation_status_is_serialized_per_region(tmp_path: Path) -> None:
    profile = _tiny_profile(
        {
            "panel": NormalizedRect(0.0, 0.0, 0.5, 1.0),
            "marker": NormalizedRect(0.5, 0.0, 0.5, 1.0),
        },
        statuses={"panel": VALIDATION_CONFIRMED, "marker": VALIDATION_PROVISIONAL},
    )
    image = _patterned_image(20, 20)
    manifest = _extract(tmp_path, image, profile)
    payload = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    by_name = {item["region_name"]: item["validation_status"] for item in payload["regions"]}
    assert by_name == {"panel": "confirmed", "marker": "provisional"}
    assert manifest.region("marker").validation_status == VALIDATION_PROVISIONAL


def test_multi_region_failure_does_not_write_final_manifest(tmp_path: Path) -> None:
    profile = _tiny_profile(
        {
            "panel": NormalizedRect(0.0, 0.0, 0.5, 0.5),
            "overflow": NormalizedRect(x=0.8, y=0.0, width=0.4, height=0.5),
        }
    )
    image = _patterned_image(20, 20)
    output_dir = tmp_path / "out"
    with pytest.raises(ExtractionError, match="does not clamp"):
        _extract(tmp_path, image, profile, output_dir=output_dir, selected_regions=("panel", "overflow"))
    assert not (output_dir / "manifest.json").exists()
    assert not (output_dir / "crops" / "panel.png").exists()


def test_png_write_failure_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _tiny_profile({"panel": NormalizedRect(0.0, 0.0, 1.0, 1.0)})
    image = _patterned_image(20, 20)
    files_dir = tmp_path / "inputs"
    image_path, yaml_path = _hash_files(files_dir, image)
    output_dir = tmp_path / "out"
    monkeypatch.setattr("ddca.vision.extraction.cv2.imwrite", lambda *_args, **_kwargs: False)
    with pytest.raises(ExtractionError, match="Failed to write lossless crop PNG"):
        extract_regions(
            image,
            profile,
            output_dir,
            source_image_path=image_path,
            calibration_path=yaml_path,
            extracted_at=EXTRACTED_AT,
        )
    assert not (output_dir / "manifest.json").exists()


def test_duplicate_regions_preserve_first_seen_order(tmp_path: Path) -> None:
    profile = _tiny_profile(
        {
            "zeta": NormalizedRect(0.0, 0.0, 0.5, 0.5),
            "alpha": NormalizedRect(0.5, 0.0, 0.5, 0.5),
        }
    )
    image = _patterned_image(20, 20)
    manifest = _extract(tmp_path, image, profile, selected_regions=("zeta", "alpha", "zeta"))
    assert [item.region_name for item in manifest.regions] == ["zeta", "alpha"]
    assert len(manifest.regions) == 2


def test_crop_copy_is_independent_of_source_image() -> None:
    image = _patterned_image(20, 20)
    pixel = resolve_pixel_rect(
        _tiny_profile({"panel": NormalizedRect(0.0, 0.0, 0.5, 0.5)}),
        "panel",
        20,
        20,
    )
    original = image[pixel.top, pixel.left].copy()
    crop = independent_crop(image, pixel)
    assert crop.dtype == image.dtype
    assert crop.ndim == image.ndim
    assert crop.shape[2] == image.shape[2]
    assert not np.shares_memory(crop, image)
    crop[0, 0] = (1, 2, 3)
    assert np.array_equal(image[pixel.top, pixel.left], original)


def test_manifest_json_uses_deterministic_key_order(tmp_path: Path) -> None:
    profile = _tiny_profile({"panel": NormalizedRect(0.0, 0.0, 1.0, 1.0)})
    image = _patterned_image(20, 20)
    _extract(tmp_path, image, profile)
    text = (tmp_path / "out" / "manifest.json").read_text(encoding="utf-8")
    parsed = json.loads(text)
    assert text == json.dumps(parsed, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
    assert list(parsed.keys()) == sorted(parsed.keys())
    assert list(parsed["frame"].keys()) == sorted(parsed["frame"].keys())
    assert list(parsed["calibration"].keys()) == sorted(parsed["calibration"].keys())
