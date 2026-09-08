"""Synthetic tests for action-slot availability classification."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from ddca.combat.enums import ActionSlot, EvidenceSource, ObservationStatus
from ddca.combat.evidence import FrameReference, RegionEvidence
from ddca.combat.observed import observed
from ddca.vision.action_state import (
    ACTION_SLOT_REGIONS,
    CONFIDENCE_CAP,
    DEFAULT_ACTION_STATE_CONFIG_PATH,
    ActionSlotClassification,
    ActionSlotFeatures,
    class_side_normalized_margin,
    classify_action_slot_features,
    classify_action_slot_image,
    classify_extraction_dir,
    final_confidence,
    input_quality,
    load_action_slot_state_config,
    load_extraction_manifest,
    write_action_slot_state_report,
)
from ddca.vision.errors import RecognitionError

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_YAML = REPO_ROOT / "configs" / "calibration" / "windowed_1111x654.yaml"
CONFIG_YAML = REPO_ROOT / "configs" / "vision" / "action_slot_state_v1.yaml"
FAKE_SHA = "a" * 64
FRAME = FrameReference(capture_id=FAKE_SHA, image_width=1111, image_height=654)
EVIDENCE = RegionEvidence(region_name="skill_slot_1", x=0, y=0, width=42, height=42)


def _config_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "algorithm_id": "action-slot-chroma-v1",
        "expected_crop_width": 42,
        "expected_crop_height": 42,
        "inner_border_margin": 4,
        "foreground_value_threshold": 18,
        "minimum_foreground_ratio": 0.08,
        "minimum_brightness_std": 8.0,
        "chromatic_saturation_threshold": 28,
        "disabled_max_chroma": 0.12,
        "available_min_chroma": 0.28,
        "confidence_distance_scale": 0.35,
        "notes": "synthetic",
    }
    payload.update(overrides)
    return payload


def _write_config(path: Path, **overrides: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_config_payload(**overrides), sort_keys=False), encoding="utf-8")
    return path


def _load_config(tmp_path: Path, **overrides: object):
    return load_action_slot_state_config(_write_config(tmp_path / "config.yaml", **overrides))


def _colored_icon(bgr: tuple[int, int, int] = (20, 180, 40)) -> np.ndarray:
    """Paint a chromatic inner icon with brightness texture so it is not uniform."""

    image = np.zeros((42, 42, 3), dtype=np.uint8)
    rows, cols = np.indices((34, 34))
    shade = 0.55 + 0.45 * ((rows + cols) % 9) / 8.0
    pixel = np.clip(np.asarray(bgr, dtype=np.float64) * shade[..., None], 0, 255)
    image[4:38, 4:38] = pixel.astype(np.uint8)
    return image


def _grey_icon(seed: int = 0) -> np.ndarray:
    image = np.zeros((42, 42, 3), dtype=np.uint8)
    grey = np.random.default_rng(seed).integers(40, 180, size=(34, 34), dtype=np.uint8)
    image[4:38, 4:38, 0] = grey
    image[4:38, 4:38, 1] = grey
    image[4:38, 4:38, 2] = grey
    return image


def _black_icon() -> np.ndarray:
    return np.zeros((42, 42, 3), dtype=np.uint8)


def _uniform_icon() -> np.ndarray:
    return np.full((42, 42, 3), 120, dtype=np.uint8)


def _colored_border_grey_inner() -> np.ndarray:
    image = np.full((42, 42, 3), (0, 220, 220), dtype=np.uint8)
    grey = np.random.default_rng(1).integers(50, 170, size=(34, 34), dtype=np.uint8)
    image[4:38, 4:38, 0] = grey
    image[4:38, 4:38, 1] = grey
    image[4:38, 4:38, 2] = grey
    return image


def _grey_border_colored_inner() -> np.ndarray:
    image = np.full((42, 42, 3), 90, dtype=np.uint8)
    image[4:38, 4:38] = _colored_icon((20, 40, 200))[4:38, 4:38]
    return image


def _abstention_icon() -> np.ndarray:
    image = _grey_icon(seed=2)
    image[10:18, 4:38] = (40, 70, 190)
    return image


def _classify(tmp_path: Path, image: np.ndarray, slot: ActionSlot = ActionSlot.SKILL_1):
    return classify_action_slot_image(
        image,
        _load_config(tmp_path),
        slot=slot,
        frame_id=FAKE_SHA,
        validation_status="confirmed",
        crop_sha256=FAKE_SHA,
    )


def _measured_features(*, chroma: float, foreground_ratio: float = 0.50, brightness_std: float = 40.0) -> ActionSlotFeatures:
    return ActionSlotFeatures(
        foreground_ratio=foreground_ratio,
        brightness_mean=80.0,
        brightness_median=80.0,
        brightness_std=brightness_std,
        mean_saturation=120.0 if chroma > 0.2 else 0.0,
        median_saturation=120.0 if chroma > 0.2 else 0.0,
        chromatic_pixel_ratio=chroma,
        colourfulness=40.0 if chroma > 0.2 else 0.0,
    )


def _classify_chroma(tmp_path: Path, chroma: float, slot: ActionSlot = ActionSlot.SKILL_1) -> ActionSlotClassification:
    return classify_action_slot_features(
        _measured_features(chroma=chroma),
        _load_config(tmp_path),
        slot=slot,
        frame_id=FAKE_SHA,
        crop_width=42,
        crop_height=42,
        validation_status="confirmed",
        crop_sha256=FAKE_SHA,
    )


def _region_payload(name: str, *, width: int = 42, height: int = 42, status: str = "confirmed") -> dict[str, object]:
    return {
        "region_name": name,
        "relative_to": "action_bar",
        "validation_status": status,
        "pixel": {"left": 0, "top": 0, "width": width, "height": height},
        "crop_path": f"crops/{name}.png",
        "width": width,
        "height": height,
        "evidence": {
            "region_name": name,
            "x": 0,
            "y": 0,
            "width": width,
            "height": height,
            "confidence": None,
        },
    }


def _manifest_payload(regions: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "frame": {
            "frame_id": FAKE_SHA,
            "source_image_name": "source.png",
            "source_image_sha256": FAKE_SHA,
            "source_image_width": 1111,
            "source_image_height": 654,
            "captured_at_utc": None,
        },
        "calibration": {"profile_id": "windowed_1111x654", "profile_sha256": FAKE_SHA},
        "extracted_at_utc": "2026-09-07T00:00:00Z",
        "regions": regions if regions is not None else [_region_payload(name) for name in ACTION_SLOT_REGIONS],
    }


def _write_extraction(
    tmp_path: Path,
    images: dict[str, np.ndarray] | None = None,
    *,
    manifest: dict[str, object] | None = None,
) -> Path:
    root = tmp_path / "extract"
    crops = root / "crops"
    crops.mkdir(parents=True)
    defaults = {name: _colored_icon() for name in ACTION_SLOT_REGIONS}
    defaults[ActionSlot.SKILL_2.value] = _grey_icon()
    if images:
        defaults.update(images)
    for name, image in defaults.items():
        assert cv2.imwrite(str(crops / f"{name}.png"), image)
    payload = manifest if manifest is not None else _manifest_payload()
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_colored_inner_icon_is_available(tmp_path: Path) -> None:
    result = _classify(tmp_path, _colored_icon())
    assert result.state == "available"
    assert result.is_available.status is ObservationStatus.OBSERVED
    assert result.is_available.value is True
    assert result.is_available.source is EvidenceSource.CLASSIFIER
    assert result.features.chroma_score >= 0.28
    assert 0.0 <= result.is_available.confidence.score <= CONFIDENCE_CAP
    assert 0.0 <= result.normalized_margin <= 1.0


def test_grey_textured_icon_is_disabled(tmp_path: Path) -> None:
    result = _classify(tmp_path, _grey_icon())
    assert result.state == "disabled"
    assert result.is_available.status is ObservationStatus.OBSERVED
    assert result.is_available.value is False
    assert result.features.chroma_score < 0.12
    assert result.features.brightness_std >= 8.0


def test_black_crop_is_unknown_not_disabled(tmp_path: Path) -> None:
    result = _classify(tmp_path, _black_icon())
    assert result.state == "unknown"
    assert result.is_available.status is ObservationStatus.UNKNOWN
    assert result.is_available.value is None
    assert "unknown rather than disabled" in result.decision_reason or "foreground" in result.decision_reason


def test_uniform_crop_is_unknown(tmp_path: Path) -> None:
    result = _classify(tmp_path, _uniform_icon())
    assert result.state == "unknown"
    assert result.is_available.value is None
    assert "uniform" in result.decision_reason or "std" in result.decision_reason


def test_abstention_band_is_unknown(tmp_path: Path) -> None:
    result = _classify(tmp_path, _abstention_icon())
    assert 0.12 < result.features.chroma_score < 0.28
    assert result.state == "unknown"
    assert "abstention band" in result.decision_reason
    assert result.is_available.value is None


def test_colored_border_does_not_make_grey_icon_available(tmp_path: Path) -> None:
    result = _classify(tmp_path, _colored_border_grey_inner())
    assert result.state == "disabled"
    assert result.is_available.value is False


def test_grey_border_does_not_hide_colored_inner_icon(tmp_path: Path) -> None:
    result = _classify(tmp_path, _grey_border_colored_inner())
    assert result.state == "available"
    assert result.is_available.value is True


def test_wrong_crop_size_is_explicit_failure(tmp_path: Path) -> None:
    with pytest.raises(RecognitionError, match="not resized"):
        _classify(tmp_path, np.zeros((20, 20, 3), dtype=np.uint8))


def test_malformed_and_numeric_config_rejected(tmp_path: Path) -> None:
    (tmp_path / "empty.yaml").write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(RecognitionError, match="must be a YAML mapping"):
        load_action_slot_state_config(tmp_path / "empty.yaml")
    with pytest.raises(RecognitionError, match="must be an integer"):
        _load_config(tmp_path / "bool_int", expected_crop_width=True)
    with pytest.raises(RecognitionError, match="must be a number"):
        _load_config(tmp_path / "bool_float", minimum_foreground_ratio=True)
    with pytest.raises(RecognitionError, match="finite"):
        _load_config(tmp_path / "nan", minimum_brightness_std=math.nan)
    with pytest.raises(RecognitionError, match="finite"):
        _load_config(tmp_path / "inf", available_min_chroma=math.inf)
    with pytest.raises(RecognitionError, match="must be <"):
        _load_config(tmp_path / "order", disabled_max_chroma=0.50, available_min_chroma=0.28)
    with pytest.raises(RecognitionError, match="must be > 0"):
        _load_config(tmp_path / "zero_disabled", disabled_max_chroma=0.0)
    with pytest.raises(RecognitionError, match="must be < 1"):
        _load_config(tmp_path / "available_one", available_min_chroma=1.0)
    with pytest.raises(RecognitionError, match="must be a number"):
        _load_config(tmp_path / "bool_disabled", disabled_max_chroma=True)


def test_bundled_config_hash_and_threshold_order() -> None:
    before = CALIBRATION_YAML.read_bytes()
    config = load_action_slot_state_config(CONFIG_YAML)
    assert config.available_min_chroma > config.disabled_max_chroma
    assert config.sha256 == hashlib.sha256(CONFIG_YAML.read_bytes()).hexdigest()
    assert CALIBRATION_YAML.read_bytes() == before
    assert DEFAULT_ACTION_STATE_CONFIG_PATH.as_posix().endswith("action_slot_state_v1.yaml")


def test_missing_manifest_fields_and_unsupported_schema(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "manifest.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(RecognitionError, match="missing required provenance"):
        load_extraction_manifest(missing)
    unsupported = tmp_path / "unsupported"
    unsupported.mkdir()
    (unsupported / "manifest.json").write_text(json.dumps({"schema_version": 99, "regions": []}), encoding="utf-8")
    with pytest.raises(RecognitionError, match="Unsupported extraction schema_version"):
        load_extraction_manifest(unsupported)


def test_missing_and_duplicate_slots_rejected(tmp_path: Path) -> None:
    missing = _write_extraction(
        tmp_path / "missing",
        manifest=_manifest_payload([_region_payload("skill_slot_1")]),
    )
    with pytest.raises(RecognitionError, match="missing required action slot"):
        classify_extraction_dir(missing, load_action_slot_state_config(CONFIG_YAML))
    duplicate_regions = [_region_payload(name) for name in ACTION_SLOT_REGIONS]
    duplicate_regions.append(_region_payload("skill_slot_1"))
    duplicate = tmp_path / "dup"
    duplicate.mkdir()
    (duplicate / "manifest.json").write_text(json.dumps(_manifest_payload(duplicate_regions)), encoding="utf-8")
    with pytest.raises(RecognitionError, match="duplicate action slot"):
        load_extraction_manifest(duplicate)


def test_path_traversal_rejected(tmp_path: Path) -> None:
    payload = _manifest_payload()
    for item in payload["regions"]:
        if item["region_name"] == "skill_slot_1":
            item["crop_path"] = "../secret.png"
    root = _write_extraction(tmp_path, manifest=payload)
    with pytest.raises(RecognitionError, match="inside the extraction directory"):
        classify_extraction_dir(root, load_action_slot_state_config(CONFIG_YAML))


def test_unreadable_crop_rejected(tmp_path: Path) -> None:
    root = _write_extraction(tmp_path)
    (root / "crops" / "skill_slot_3.png").write_bytes(b"not-a-png")
    with pytest.raises(RecognitionError, match="unreadable"):
        classify_extraction_dir(root, load_action_slot_state_config(CONFIG_YAML))


def test_manifest_crop_dimension_mismatch_rejected(tmp_path: Path) -> None:
    images = {name: _colored_icon() for name in ACTION_SLOT_REGIONS}
    images["skill_slot_4"] = np.zeros((20, 20, 3), dtype=np.uint8)
    root = _write_extraction(tmp_path, images)
    with pytest.raises(RecognitionError, match="manifest records"):
        classify_extraction_dir(root, load_action_slot_state_config(CONFIG_YAML))


def test_stable_five_slot_order_and_observation_mapping(tmp_path: Path) -> None:
    root = _write_extraction(tmp_path)
    report = classify_extraction_dir(root, load_action_slot_state_config(CONFIG_YAML))
    assert [item.slot for item in report.classifications] == list(ActionSlot)
    assert [item.slot.value for item in report.classifications] == list(ACTION_SLOT_REGIONS)
    observations = report.observations
    assert observations[0].is_available.value is True
    assert observations[1].is_available.value is False
    assert observations[1].skill_id.status is ObservationStatus.UNKNOWN
    assert observations[4].slot is ActionSlot.MOVE
    assert observations[4].skill_id.status is ObservationStatus.NOT_APPLICABLE
    assert all(item.legal_target_ranks.status is ObservationStatus.UNKNOWN for item in observations)
    projected = report.classifications[0].to_action_observation(FRAME, EVIDENCE)
    assert projected.is_available.value is True
    assert projected.kind.value == "skill"


def test_unknown_maps_to_unknown_none(tmp_path: Path) -> None:
    images = {name: _black_icon() for name in ACTION_SLOT_REGIONS}
    root = _write_extraction(tmp_path, images)
    report = classify_extraction_dir(root, load_action_slot_state_config(CONFIG_YAML))
    for item in report.classifications:
        assert item.state == "unknown"
        assert item.is_available.status is ObservationStatus.UNKNOWN
        assert item.is_available.value is None


def test_overwrite_refusal_and_atomic_deterministic_json(tmp_path: Path) -> None:
    root = _write_extraction(tmp_path)
    report = classify_extraction_dir(root, load_action_slot_state_config(CONFIG_YAML))
    output = tmp_path / "out" / "report.json"
    write_action_slot_state_report(report, output)
    original = output.read_text(encoding="utf-8")
    parsed = json.loads(original)
    assert original == json.dumps(parsed, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
    assert list(parsed.keys()) == sorted(parsed.keys())
    assert [item["slot"] for item in parsed["classifications"]] == list(ACTION_SLOT_REGIONS)
    with pytest.raises(RecognitionError, match="Refusing to overwrite"):
        write_action_slot_state_report(report, output)
    assert output.read_text(encoding="utf-8") == original
    write_action_slot_state_report(report, output, overwrite=True)
    assert not (output.parent / ".report.json.tmp").exists()
    encoded = json.dumps(parsed)
    assert "win_probability" not in encoded
    assert "utility" not in encoded
    assert "CombatState" not in encoded
    assert "skill_name" not in encoded
    assert "Noxious" not in encoded
    assert ":\\" not in encoded


def test_calibration_yaml_unchanged_by_classifier(tmp_path: Path) -> None:
    before = CALIBRATION_YAML.read_bytes()
    load_action_slot_state_config(CONFIG_YAML)
    classify_action_slot_image(
        _colored_icon(),
        _load_config(tmp_path),
        slot=ActionSlot.SKILL_1,
        frame_id=FAKE_SHA,
        validation_status="confirmed",
        crop_sha256=FAKE_SHA,
    )
    assert CALIBRATION_YAML.read_bytes() == before
    assert yaml.safe_load(before)["profile_id"] == "windowed_1111x654"


def test_report_does_not_create_combat_state(tmp_path: Path) -> None:
    root = _write_extraction(tmp_path)
    report = classify_extraction_dir(root, load_action_slot_state_config(CONFIG_YAML))
    encoded = json.dumps(report.to_dict())
    assert "CombatState" not in encoded
    assert all(item.skill_id.value is None for item in report.observations)


def test_chroma_zero_has_strong_disabled_normalized_margin(tmp_path: Path) -> None:
    result = _classify_chroma(tmp_path, 0.0)
    assert result.state == "disabled"
    assert result.normalized_margin == pytest.approx(1.0)
    assert result.normalized_margin > 0.9
    assert result.is_available.confidence.score > 0.5
    payload = result.to_dict()
    assert payload["normalized_margin"] == pytest.approx(1.0)
    assert payload["quality"] == pytest.approx(result.quality)


def test_chroma_one_has_strong_available_normalized_margin(tmp_path: Path) -> None:
    result = _classify_chroma(tmp_path, 1.0)
    assert result.state == "available"
    assert result.normalized_margin == pytest.approx(1.0)
    assert result.normalized_margin > 0.9
    assert result.is_available.confidence.score > 0.5


def test_equivalent_normalized_margins_yield_equal_confidence(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    margin = 0.5
    disabled_chroma = config.disabled_max_chroma * (1.0 - margin)
    available_chroma = config.available_min_chroma + margin * (1.0 - config.available_min_chroma)
    disabled = classify_action_slot_features(
        _measured_features(chroma=disabled_chroma),
        config,
        slot=ActionSlot.SKILL_1,
        frame_id=FAKE_SHA,
        crop_width=42,
        crop_height=42,
        validation_status="confirmed",
        crop_sha256=FAKE_SHA,
    )
    available = classify_action_slot_features(
        _measured_features(chroma=available_chroma),
        config,
        slot=ActionSlot.SKILL_1,
        frame_id=FAKE_SHA,
        crop_width=42,
        crop_height=42,
        validation_status="confirmed",
        crop_sha256=FAKE_SHA,
    )
    assert class_side_normalized_margin(disabled_chroma, config, state="disabled") == pytest.approx(margin)
    assert class_side_normalized_margin(available_chroma, config, state="available") == pytest.approx(margin)
    assert disabled.state == "disabled"
    assert available.state == "available"
    assert disabled.quality == pytest.approx(available.quality)
    assert disabled.normalized_margin == pytest.approx(available.normalized_margin)
    assert disabled.is_available.confidence.score == pytest.approx(available.is_available.confidence.score)
    assert disabled.is_available.confidence.score == pytest.approx(
        final_confidence(margin, disabled.quality, config.confidence_distance_scale)
    )


def test_exact_disabled_max_is_unknown(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    result = classify_action_slot_features(
        _measured_features(chroma=config.disabled_max_chroma),
        config,
        slot=ActionSlot.SKILL_1,
        frame_id=FAKE_SHA,
        crop_width=42,
        crop_height=42,
        validation_status="confirmed",
        crop_sha256=FAKE_SHA,
    )
    assert result.state == "unknown"
    assert result.is_available.status is ObservationStatus.UNKNOWN
    assert result.is_available.value is None
    assert result.normalized_margin == 0.0
    assert "including exact decision boundaries" in result.decision_reason


def test_exact_available_min_is_unknown(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    result = classify_action_slot_features(
        _measured_features(chroma=config.available_min_chroma),
        config,
        slot=ActionSlot.SKILL_1,
        frame_id=FAKE_SHA,
        crop_width=42,
        crop_height=42,
        validation_status="confirmed",
        crop_sha256=FAKE_SHA,
    )
    assert result.state == "unknown"
    assert result.is_available.status is ObservationStatus.UNKNOWN
    assert result.is_available.value is None
    assert result.normalized_margin == 0.0


def test_confidence_monotonic_away_from_class_boundary(tmp_path: Path) -> None:
    config = _load_config(tmp_path)
    disabled_chromas = (0.11, 0.06, 0.0)
    disabled_results = [_classify_chroma(tmp_path, chroma) for chroma in disabled_chromas]
    assert [item.state for item in disabled_results] == ["disabled"] * 3
    disabled_margins = [item.normalized_margin for item in disabled_results]
    disabled_scores = [item.is_available.confidence.score for item in disabled_results]
    assert disabled_margins == sorted(disabled_margins)
    assert disabled_scores == sorted(disabled_scores)
    assert len(set(disabled_margins)) == 3
    available_chromas = (0.29, 0.64, 1.0)
    available_results = [_classify_chroma(tmp_path, chroma) for chroma in available_chromas]
    assert [item.state for item in available_results] == ["available"] * 3
    available_margins = [item.normalized_margin for item in available_results]
    available_scores = [item.is_available.confidence.score for item in available_results]
    assert available_margins == sorted(available_margins)
    assert available_scores == sorted(available_scores)
    assert len(set(available_margins)) == 3
    assert all(item.quality == pytest.approx(input_quality(item.features, config)) for item in disabled_results)


def test_every_confidence_remains_inside_cap(tmp_path: Path) -> None:
    for chroma in (index / 100.0 for index in range(101)):
        result = _classify_chroma(tmp_path, chroma)
        assert 0.0 <= result.is_available.confidence.score <= CONFIDENCE_CAP
        assert 0.0 <= result.normalized_margin <= 1.0
        assert 0.0 <= result.quality <= 1.0


def test_zero_disabled_max_rejected(tmp_path: Path) -> None:
    with pytest.raises(RecognitionError, match="must be > 0"):
        _load_config(tmp_path / "zero_disabled_max", disabled_max_chroma=0.0)


def test_available_min_one_rejected(tmp_path: Path) -> None:
    with pytest.raises(RecognitionError, match="must be < 1"):
        _load_config(tmp_path / "available_min_one", available_min_chroma=1.0)


def _direct_classification(**overrides: object) -> ActionSlotClassification:
    payload: dict[str, object] = {
        "slot": ActionSlot.SKILL_1,
        "is_available": observed(True, score=0.5, source=EvidenceSource.CLASSIFIER),
        "algorithm_id": "action-slot-chroma-v1",
        "config_sha256": FAKE_SHA,
        "frame_id": FAKE_SHA,
        "crop_width": 42,
        "crop_height": 42,
        "validation_status": "confirmed",
        "features": _measured_features(chroma=0.9),
        "crop_sha256": FAKE_SHA,
        "quality": 1.0,
        "normalized_margin": 0.8,
        "decision_reason": "direct construction",
    }
    payload.update(overrides)
    return ActionSlotClassification(**payload)  # type: ignore[arg-type]


def test_crop_sha256_matches_exact_png_bytes(tmp_path: Path) -> None:
    root = _write_extraction(tmp_path)
    before = {
        name: (root / "crops" / f"{name}.png").read_bytes() for name in ACTION_SLOT_REGIONS
    }
    report = classify_extraction_dir(root, load_action_slot_state_config(CONFIG_YAML))
    after = {
        name: (root / "crops" / f"{name}.png").read_bytes() for name in ACTION_SLOT_REGIONS
    }
    assert before == after
    for item in report.classifications:
        crop_bytes = before[item.slot.value]
        expected = hashlib.sha256(crop_bytes).hexdigest()
        assert item.crop_sha256 == expected
        assert re.fullmatch(r"^[0-9a-f]{64}$", item.crop_sha256)


def test_repeated_classification_keeps_the_same_crop_hash(tmp_path: Path) -> None:
    root = _write_extraction(tmp_path)
    config = load_action_slot_state_config(CONFIG_YAML)
    first = classify_extraction_dir(root, config)
    second = classify_extraction_dir(root, config)
    assert [item.crop_sha256 for item in first.classifications] == [
        item.crop_sha256 for item in second.classifications
    ]


def test_changing_crop_bytes_changes_hash(tmp_path: Path) -> None:
    root = _write_extraction(tmp_path)
    config = load_action_slot_state_config(CONFIG_YAML)
    original = classify_extraction_dir(root, config)
    original_hash = original.classifications[0].crop_sha256
    replacement = root / "crops" / "skill_slot_1.png"
    assert cv2.imwrite(
        str(replacement),
        _colored_icon((40, 30, 210)),
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    )
    changed = classify_extraction_dir(root, config)
    changed_hash = changed.classifications[0].crop_sha256
    assert changed_hash != original_hash
    assert changed_hash == hashlib.sha256(replacement.read_bytes()).hexdigest()


def test_invalid_crop_sha256_rejected_on_direct_construction() -> None:
    with pytest.raises(RecognitionError, match="64-character lowercase SHA-256"):
        _direct_classification(crop_sha256="not-a-hash")
    with pytest.raises(RecognitionError, match="64-character lowercase SHA-256"):
        _direct_classification(crop_sha256="A" * 64)
    with pytest.raises(RecognitionError, match="64-character lowercase SHA-256"):
        _direct_classification(crop_sha256="a" * 63)
    with pytest.raises(RecognitionError, match="64-character lowercase SHA-256"):
        _direct_classification(crop_sha256="a" * 65)


def test_output_json_contains_five_crop_hashes_and_no_absolute_paths(tmp_path: Path) -> None:
    root = _write_extraction(tmp_path)
    report = classify_extraction_dir(root, load_action_slot_state_config(CONFIG_YAML))
    output = tmp_path / "report.json"
    write_action_slot_state_report(report, output)
    encoded = output.read_text(encoding="utf-8")
    payload = json.loads(encoded)
    hashes = [item["crop_sha256"] for item in payload["classifications"]]
    assert len(hashes) == 5
    assert all(re.fullmatch(r"^[0-9a-f]{64}$", value) for value in hashes)
    assert len(set(hashes)) >= 2
    assert str(root.resolve()) not in encoded
    assert str(output.resolve()) not in encoded
    assert ":\\" not in encoded
    assert not any("crop_path" in item for item in payload["classifications"])
    for item in payload["classifications"]:
        assert "normalized_margin" in item
        assert "quality" in item
