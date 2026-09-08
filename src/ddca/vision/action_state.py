"""Deterministic action-slot availability from Phase 3B-1 crops.

This module classifies the four skill slots and Move as available, disabled,
or unknown. It does not identify skill names, compute legal targets, or build
a CombatState.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from ddca.combat.actions import SLOT_ORDER
from ddca.combat.enums import ActionSlot, EvidenceSource, ObservationStatus, action_kind_for_slot
from ddca.combat.errors import CombatModelError
from ddca.combat.evidence import FrameReference, RegionEvidence
from ddca.combat.observations import ActionObservation
from ddca.combat.observed import ObservedValue, not_applicable, observed, unknown
from ddca.vision.calibration import VALIDATION_CONFIRMED
from ddca.vision.errors import ExtractionError, RecognitionError
from ddca.vision.extraction import (
    EXTRACTION_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    SHA256_HEX,
    ExtractedRegion,
    ExtractionManifest,
    sha256_file,
)

ACTION_STATE_SCHEMA_VERSION = 1
CONFIDENCE_CAP = 0.85
DEFAULT_ACTION_STATE_CONFIG_PATH = Path("configs/vision/action_slot_state_v1.yaml")
ACTION_SLOT_REGIONS: tuple[str, ...] = tuple(slot.value for slot in SLOT_ORDER)
_SLOT_BY_REGION: dict[str, ActionSlot] = {slot.value: slot for slot in SLOT_ORDER}

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only when PyYAML is missing
    raise ImportError(
        "PyYAML is required to load action-slot state config. "
        'Install this project with: python -m pip install -e ".[dev]"'
    ) from exc


@dataclass(frozen=True)
class ActionSlotStateConfig:
    """Versioned, validated thresholds for the chroma availability baseline."""

    algorithm_id: str
    expected_crop_width: int
    expected_crop_height: int
    inner_border_margin: int
    foreground_value_threshold: float
    minimum_foreground_ratio: float
    minimum_brightness_std: float
    chromatic_saturation_threshold: float
    disabled_max_chroma: float
    available_min_chroma: float
    confidence_distance_scale: float
    notes: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm_id, str) or not self.algorithm_id.strip():
            raise RecognitionError("algorithm_id must be a non-empty string.")
        object.__setattr__(self, "algorithm_id", self.algorithm_id.strip())
        if not isinstance(self.sha256, str) or len(self.sha256) != 64 or self.sha256 != self.sha256.lower():
            raise RecognitionError("ActionSlotStateConfig.sha256 must be a 64-character lowercase hex digest.")
        if not math.isfinite(self.disabled_max_chroma) or not math.isfinite(self.available_min_chroma):
            raise RecognitionError("Chroma thresholds must be finite real values.")
        if self.disabled_max_chroma <= 0.0:
            raise RecognitionError(
                "disabled_max_chroma must be > 0 so disabled-side normalized margin is defined. "
                "Invalid values are not clamped."
            )
        if self.available_min_chroma >= 1.0:
            raise RecognitionError(
                "available_min_chroma must be < 1 so available-side normalized margin is defined. "
                "Invalid values are not clamped."
            )
        if self.disabled_max_chroma >= self.available_min_chroma:
            raise RecognitionError(
                "disabled_max_chroma must be < available_min_chroma "
                "so the classifier keeps an abstention band."
            )
        if 2 * self.inner_border_margin >= min(self.expected_crop_width, self.expected_crop_height):
            raise RecognitionError(
                "inner_border_margin leaves no inner content. "
                "Reduce the margin; invalid settings are not clamped."
            )


@dataclass(frozen=True)
class ActionSlotFeatures:
    """Explainable measurements retained for review. Not gameplay scores."""

    foreground_ratio: float
    brightness_mean: float
    brightness_median: float
    brightness_std: float
    mean_saturation: float
    median_saturation: float
    chromatic_pixel_ratio: float
    colourfulness: float

    @property
    def chroma_score(self) -> float:
        """Primary decision score: inner-foreground chromatic pixel ratio."""

        return self.chromatic_pixel_ratio

    def to_dict(self) -> dict[str, float]:
        return {
            "brightness_mean": self.brightness_mean,
            "brightness_median": self.brightness_median,
            "brightness_std": self.brightness_std,
            "chroma_score": self.chroma_score,
            "chromatic_pixel_ratio": self.chromatic_pixel_ratio,
            "colourfulness": self.colourfulness,
            "foreground_ratio": self.foreground_ratio,
            "mean_saturation": self.mean_saturation,
            "median_saturation": self.median_saturation,
        }


@dataclass(frozen=True)
class ActionSlotClassification:
    """Vision-layer result for one action slot. Not a CombatState."""

    slot: ActionSlot
    is_available: ObservedValue[bool]
    algorithm_id: str
    config_sha256: str
    frame_id: str
    crop_width: int
    crop_height: int
    validation_status: str
    features: ActionSlotFeatures
    crop_sha256: str
    quality: float
    normalized_margin: float
    decision_reason: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.slot, ActionSlot):
            raise RecognitionError(f"ActionSlotClassification.slot must be an ActionSlot (got {self.slot!r}).")
        if self.slot.value not in ACTION_SLOT_REGIONS:
            raise RecognitionError(f"Unsupported action slot {self.slot.value!r}.")
        if self.is_available.source is not EvidenceSource.CLASSIFIER:
            raise RecognitionError("Action-slot availability must use EvidenceSource.CLASSIFIER.")
        if self.validation_status != VALIDATION_CONFIRMED:
            raise RecognitionError(
                f"Action slot {self.slot.value} validation_status must be confirmed "
                f"(got {self.validation_status!r})."
            )
        if not isinstance(self.crop_sha256, str) or not SHA256_HEX.fullmatch(self.crop_sha256):
            raise RecognitionError(
                "ActionSlotClassification.crop_sha256 must be a 64-character lowercase SHA-256 "
                "of the exact crop PNG bytes."
            )
        if not math.isfinite(self.quality) or self.quality < 0.0 or self.quality > 1.0:
            raise RecognitionError("ActionSlotClassification.quality must be a finite value in [0, 1].")
        if (
            not math.isfinite(self.normalized_margin)
            or self.normalized_margin < 0.0
            or self.normalized_margin > 1.0
        ):
            raise RecognitionError(
                "ActionSlotClassification.normalized_margin must be a finite value in [0, 1]."
            )

    @property
    def state(self) -> str:
        """Return available, disabled, or unknown from the observed availability."""

        if self.is_available.status is ObservationStatus.UNKNOWN:
            return "unknown"
        if self.is_available.value is True:
            return "available"
        if self.is_available.value is False:
            return "disabled"
        raise RecognitionError("Observed availability must be true, false, or unknown with value=None.")

    def to_action_observation(
        self,
        frame: FrameReference,
        evidence: RegionEvidence,
    ) -> ActionObservation:
        """Populate the existing ActionObservation without adding domain fields."""

        skill_id = (
            not_applicable(source=EvidenceSource.CLASSIFIER)
            if self.slot is ActionSlot.MOVE
            else unknown(source=EvidenceSource.CLASSIFIER)
        )
        return ActionObservation(
            frame=frame,
            slot=self.slot,
            kind=action_kind_for_slot(self.slot),
            skill_id=skill_id,
            is_available=self.is_available,
            legal_target_ranks=unknown(source=EvidenceSource.CLASSIFIER),
            evidence=(evidence,),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm_id": self.algorithm_id,
            "config_sha256": self.config_sha256,
            "crop_height": self.crop_height,
            "crop_sha256": self.crop_sha256,
            "crop_width": self.crop_width,
            "decision_reason": self.decision_reason,
            "features": self.features.to_dict(),
            "frame_id": self.frame_id,
            "normalized_margin": self.normalized_margin,
            "quality": self.quality,
            "is_available": self.is_available.to_dict(),
            "slot": self.slot.value,
            "state": self.state,
            "validation_status": self.validation_status,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ActionSlotStateReport:
    """Deterministic five-slot classification report. Does not store pixels."""

    schema_version: int
    algorithm_id: str
    config_sha256: str
    frame_id: str
    classifications: tuple[ActionSlotClassification, ...]
    observations: tuple[ActionObservation, ...]

    def __post_init__(self) -> None:
        if self.schema_version != ACTION_STATE_SCHEMA_VERSION:
            raise RecognitionError(
                f"Unsupported action-state schema_version {self.schema_version}. "
                f"This build understands version {ACTION_STATE_SCHEMA_VERSION} only."
            )
        if len(self.classifications) != len(SLOT_ORDER):
            raise RecognitionError("Action-state reports must contain exactly the five action slots.")
        names = [item.slot for item in self.classifications]
        if tuple(names) != SLOT_ORDER:
            raise RecognitionError(
                "Action-state classifications must stay in skill 1-4 then Move order."
            )
        if [item.slot for item in self.observations] != list(SLOT_ORDER):
            raise RecognitionError("ActionObservation order must match SLOT_ORDER.")

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": {
                "algorithm_id": self.algorithm_id,
                "config_sha256": self.config_sha256,
            },
            "classifications": [item.to_dict() for item in self.classifications],
            "frame_id": self.frame_id,
            "observations": [item.to_dict() for item in self.observations],
            "schema_version": self.schema_version,
        }


def load_action_slot_state_config(path: Path | str) -> ActionSlotStateConfig:
    """Load and validate a versioned action-slot classifier config."""

    config_path = Path(path)
    if not config_path.is_file():
        raise RecognitionError(f"Action-slot state config is not an existing file ({config_path.name}).")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RecognitionError(f"Could not read action-slot state config {config_path.name}: {exc}.") from exc
    if not isinstance(payload, Mapping):
        raise RecognitionError("Action-slot state config must be a YAML mapping.")
    version = payload.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != ACTION_STATE_SCHEMA_VERSION:
        raise RecognitionError(
            f"Unsupported action-slot config schema_version {version!r}. "
            f"This build understands version {ACTION_STATE_SCHEMA_VERSION} only."
        )
    width = _require_int(payload, "expected_crop_width", minimum=1)
    height = _require_int(payload, "expected_crop_height", minimum=1)
    return ActionSlotStateConfig(
        algorithm_id=str(payload.get("algorithm_id", "")),
        expected_crop_width=width,
        expected_crop_height=height,
        inner_border_margin=_require_int(payload, "inner_border_margin", minimum=0),
        foreground_value_threshold=_require_number(payload, "foreground_value_threshold", minimum=0.0, maximum=255.0),
        minimum_foreground_ratio=_require_number(payload, "minimum_foreground_ratio", minimum=0.0, maximum=1.0),
        minimum_brightness_std=_require_number(payload, "minimum_brightness_std", minimum=0.0),
        chromatic_saturation_threshold=_require_number(
            payload, "chromatic_saturation_threshold", minimum=0.0, maximum=255.0
        ),
        disabled_max_chroma=_require_number(payload, "disabled_max_chroma", exclusive_minimum=0.0, exclusive_maximum=1.0),
        available_min_chroma=_require_number(payload, "available_min_chroma", exclusive_minimum=0.0, exclusive_maximum=1.0),
        confidence_distance_scale=_require_number(payload, "confidence_distance_scale", exclusive_minimum=0.0),
        notes=str(payload.get("notes", "")).strip(),
        sha256=sha256_file(config_path),
    )


def measure_action_slot_features(image: NDArray[np.uint8], config: ActionSlotStateConfig) -> ActionSlotFeatures:
    """Measure inner-content chroma features. Does not classify skill identity."""

    empty = ActionSlotFeatures(
        foreground_ratio=0.0,
        brightness_mean=0.0,
        brightness_median=0.0,
        brightness_std=0.0,
        mean_saturation=0.0,
        median_saturation=0.0,
        chromatic_pixel_ratio=0.0,
        colourfulness=0.0,
    )
    if not isinstance(image, np.ndarray) or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        return empty
    if image.dtype != np.uint8:
        return empty

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float64)
    value = hsv[:, :, 2].astype(np.float64)
    inner = _inner_mask(image.shape[0], image.shape[1], config.inner_border_margin)
    if not np.any(inner):
        return empty

    foreground = inner & (value >= config.foreground_value_threshold)
    inner_count = int(inner.sum())
    foreground_count = int(foreground.sum())
    chromatic = foreground & (saturation >= config.chromatic_saturation_threshold)
    blue, green, red = (image[:, :, index].astype(np.float64) for index in range(3))
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)
    inner_rg = rg[inner]
    inner_yb = yb[inner]
    colourfulness = float(
        math.sqrt(float(inner_rg.std()) ** 2 + float(inner_yb.std()) ** 2)
        + 0.3 * math.sqrt(float(inner_rg.mean()) ** 2 + float(inner_yb.mean()) ** 2)
    )
    return ActionSlotFeatures(
        foreground_ratio=foreground_count / inner_count,
        brightness_mean=_masked_mean(value, foreground),
        brightness_median=_masked_median(value, foreground),
        brightness_std=float(value[inner].std()),
        mean_saturation=_masked_mean(saturation, foreground),
        median_saturation=_masked_median(saturation, foreground),
        chromatic_pixel_ratio=(int(chromatic.sum()) / foreground_count) if foreground_count else 0.0,
        colourfulness=colourfulness,
    )


def classify_action_slot_features(
    features: ActionSlotFeatures,
    config: ActionSlotStateConfig,
    *,
    slot: ActionSlot,
    frame_id: str,
    crop_width: int,
    crop_height: int,
    validation_status: str,
    crop_sha256: str,
) -> ActionSlotClassification:
    """Apply the available / disabled / unknown decision order to measured features."""

    warnings: list[str] = []
    quality = input_quality(features, config)
    normalized_margin = 0.0
    if features.foreground_ratio < config.minimum_foreground_ratio:
        availability, reason, score = _unknown_result(
            "insufficient_foreground",
            f"Inner foreground ratio {features.foreground_ratio:.3f} is below "
            f"{config.minimum_foreground_ratio:.3f}, so the crop is unknown rather than disabled.",
        )
    elif features.brightness_std < config.minimum_brightness_std:
        availability, reason, score = _unknown_result(
            "nearly_uniform",
            f"Inner brightness std {features.brightness_std:.3f} is below "
            f"{config.minimum_brightness_std:.3f}, so the crop is unknown.",
        )
    elif features.chroma_score < config.disabled_max_chroma:
        normalized_margin = class_side_normalized_margin(features.chroma_score, config, state="disabled")
        score = final_confidence(normalized_margin, quality, config.confidence_distance_scale)
        availability = observed(False, score=score, source=EvidenceSource.CLASSIFIER)
        reason = (
            f"Inner chroma score {features.chroma_score:.3f} is below the disabled "
            f"threshold {config.disabled_max_chroma:.3f}."
        )
    elif features.chroma_score > config.available_min_chroma:
        normalized_margin = class_side_normalized_margin(features.chroma_score, config, state="available")
        score = final_confidence(normalized_margin, quality, config.confidence_distance_scale)
        availability = observed(True, score=score, source=EvidenceSource.CLASSIFIER)
        reason = (
            f"Inner chroma score {features.chroma_score:.3f} is above the available "
            f"threshold {config.available_min_chroma:.3f}."
        )
    else:
        availability, reason, score = _unknown_result(
            "abstention_band",
            f"Inner chroma score {features.chroma_score:.3f} is inside the abstention band "
            f"[{config.disabled_max_chroma:.3f}, {config.available_min_chroma:.3f}], "
            "including exact decision boundaries.",
        )
        warnings.append("chroma score is inside the abstention band")

    return ActionSlotClassification(
        slot=slot,
        is_available=availability,
        algorithm_id=config.algorithm_id,
        config_sha256=config.sha256,
        frame_id=frame_id,
        crop_width=crop_width,
        crop_height=crop_height,
        validation_status=validation_status,
        features=features,
        crop_sha256=crop_sha256,
        quality=quality,
        normalized_margin=normalized_margin,
        decision_reason=reason,
        warnings=tuple(warnings),
    )


def classify_action_slot_image(
    image: NDArray[np.uint8],
    config: ActionSlotStateConfig,
    *,
    slot: ActionSlot,
    frame_id: str,
    validation_status: str,
    crop_sha256: str,
) -> ActionSlotClassification:
    """Classify one BGR crop. Wrong configured dimensions are an explicit failure."""

    _assert_crop_matches_config(image, config, slot=slot.value)
    features = measure_action_slot_features(image, config)
    return classify_action_slot_features(
        features,
        config,
        slot=slot,
        frame_id=frame_id,
        crop_width=int(image.shape[1]),
        crop_height=int(image.shape[0]),
        validation_status=validation_status,
        crop_sha256=crop_sha256,
    )


def classify_extraction_dir(
    extraction_dir: Path | str,
    config: ActionSlotStateConfig,
) -> ActionSlotStateReport:
    """Classify the five action slots from a Phase 3B-1 extraction directory."""

    root, manifest = load_extraction_manifest(extraction_dir)
    if not manifest.frame.frame_id.strip() or not config.sha256.strip():
        raise RecognitionError("frame_id and classifier config SHA-256 must be non-empty.")
    crops = _load_required_action_crops(root, manifest, config)
    classifications: list[ActionSlotClassification] = []
    observations: list[ActionObservation] = []
    frame = manifest.frame.to_frame_reference()
    for crop in crops:
        result = classify_action_slot_image(
            crop.image,
            config,
            slot=crop.slot,
            frame_id=manifest.frame.frame_id,
            validation_status=crop.region.validation_status,
            crop_sha256=crop.crop_sha256,
        )
        classifications.append(result)
        observations.append(result.to_action_observation(frame, crop.region.evidence))
    return ActionSlotStateReport(
        schema_version=ACTION_STATE_SCHEMA_VERSION,
        algorithm_id=config.algorithm_id,
        config_sha256=config.sha256,
        frame_id=manifest.frame.frame_id,
        classifications=tuple(classifications),
        observations=tuple(observations),
    )


def write_action_slot_state_report(
    report: ActionSlotStateReport,
    output_path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the classification report as deterministic JSON. Atomic replace."""

    path = Path(output_path)
    if path.exists() and not overwrite:
        raise RecognitionError(
            f"Refusing to overwrite existing action-state file ({path.name}). "
            "Pass --overwrite to replace only this report."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, report.to_dict())
    return path


def load_extraction_manifest(extraction_dir: Path | str) -> tuple[Path, ExtractionManifest]:
    """Load a Phase 3B-1 manifest from an extraction directory."""

    root = Path(extraction_dir)
    manifest_path = root if root.name == MANIFEST_FILENAME and root.is_file() else root / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise RecognitionError(
            f"Phase 3B-2 requires a Phase 3B-1 extraction directory containing {MANIFEST_FILENAME}."
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecognitionError(f"Could not read extraction manifest {manifest_path.name}: {exc}.") from exc
    if not isinstance(payload, Mapping):
        raise RecognitionError("Extraction manifest must be a JSON object.")
    version = payload.get("schema_version")
    if version != EXTRACTION_SCHEMA_VERSION:
        raise RecognitionError(
            f"Unsupported extraction schema_version {version!r}. "
            f"Phase 3B-2 understands extraction schema {EXTRACTION_SCHEMA_VERSION} only."
        )
    regions_raw = payload.get("regions")
    if isinstance(regions_raw, list):
        seen_slots: set[str] = set()
        for item in regions_raw:
            if not isinstance(item, Mapping):
                continue
            name = item.get("region_name")
            if name in ACTION_SLOT_REGIONS:
                if name in seen_slots:
                    raise RecognitionError(f"Extraction manifest contains a duplicate action slot {name!r}.")
                seen_slots.add(str(name))
    try:
        manifest = ExtractionManifest.from_dict(payload)
    except (ExtractionError, CombatModelError, TypeError, KeyError, ValueError) as exc:
        raise RecognitionError(
            f"Extraction manifest is missing required provenance or region fields: {exc}."
        ) from exc
    return manifest_path.parent, manifest


@dataclass(frozen=True)
class _LoadedActionCrop:
    slot: ActionSlot
    region: ExtractedRegion
    image: NDArray[np.uint8]
    crop_sha256: str


def _load_required_action_crops(
    root: Path,
    manifest: ExtractionManifest,
    config: ActionSlotStateConfig,
) -> tuple[_LoadedActionCrop, ...]:
    seen: set[str] = set()
    loaded: list[_LoadedActionCrop] = []
    for region_name in ACTION_SLOT_REGIONS:
        try:
            region = manifest.region(region_name)
        except ExtractionError as exc:
            raise RecognitionError(
                f"Extraction manifest is missing required action slot {region_name!r}."
            ) from exc
        if region_name in seen:
            raise RecognitionError(f"Extraction manifest contains a duplicate action slot {region_name!r}.")
        seen.add(region_name)
        if region.validation_status != VALIDATION_CONFIRMED:
            raise RecognitionError(
                f"Action slot {region_name} must have validation_status 'confirmed' "
                f"(got {region.validation_status!r}). Provisional slots are not classified."
            )
        crop_path = _resolve_crop_path(root, region.crop_path)
        if not crop_path.is_file():
            raise RecognitionError(f"Action-slot crop {region.crop_path} does not exist inside the extraction directory.")
        crop_sha256 = sha256_file(crop_path)
        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RecognitionError(f"Action-slot crop {Path(region.crop_path).name} is unreadable.")
        if image.shape[1] != region.width or image.shape[0] != region.height:
            raise RecognitionError(
                f"Crop {Path(region.crop_path).name} is {image.shape[1]}x{image.shape[0]} "
                f"but the manifest records {region.width}x{region.height}."
            )
        _assert_crop_matches_config(image, config, slot=region_name)
        slot = _SLOT_BY_REGION.get(region_name)
        if slot is None:
            raise RecognitionError(f"Region {region_name!r} does not map to a known ActionSlot.")
        loaded.append(
            _LoadedActionCrop(slot=slot, region=region, image=image, crop_sha256=crop_sha256)
        )
    return tuple(loaded)


def _resolve_crop_path(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RecognitionError(
            f"crop_path {relative!r} must stay inside the extraction directory. "
            "Absolute paths and parent traversal are rejected."
        )
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RecognitionError(
            f"crop_path {relative!r} escaped the extraction directory."
        ) from exc
    return resolved


def _assert_crop_matches_config(image: NDArray[np.uint8], config: ActionSlotStateConfig, *, slot: str) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise RecognitionError(f"Action-slot crop {slot} must be a BGR uint8 image.")
    height, width = image.shape[:2]
    if width != config.expected_crop_width or height != config.expected_crop_height:
        raise RecognitionError(
            f"Action-slot crop {slot} is {width}x{height} but the classifier config "
            f"expects {config.expected_crop_width}x{config.expected_crop_height}. "
            "Crops are not resized."
        )


def _inner_mask(height: int, width: int, margin: int) -> NDArray[np.bool_]:
    mask = np.zeros((height, width), dtype=bool)
    if height > 2 * margin and width > 2 * margin:
        mask[margin : height - margin, margin : width - margin] = True
    return mask


def _masked_mean(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
    if not np.any(mask):
        return 0.0
    return float(values[mask].mean())


def _masked_median(values: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.median(values[mask]))


def input_quality(features: ActionSlotFeatures, config: ActionSlotStateConfig) -> float:
    """Return input-quality in [0, 1]. Not a gameplay probability."""

    quality = 1.0
    if features.foreground_ratio < min(1.0, 2.0 * config.minimum_foreground_ratio):
        quality *= 0.7
    if features.brightness_std < min(255.0, 2.0 * config.minimum_brightness_std):
        quality *= 0.7
    return quality


def class_side_normalized_margin(
    chroma_score: float,
    config: ActionSlotStateConfig,
    *,
    state: str,
) -> float:
    """Normalize distance from the class decision boundary into [0, 1]."""

    if state == "disabled":
        margin = (config.disabled_max_chroma - chroma_score) / config.disabled_max_chroma
    elif state == "available":
        margin = (chroma_score - config.available_min_chroma) / (1.0 - config.available_min_chroma)
    else:
        return 0.0
    if margin < 0.0 or margin > 1.0:
        raise RecognitionError(
            f"normalized_margin {margin} is outside [0, 1] for state {state!r}. "
            "Invalid chroma/threshold combinations are not clamped."
        )
    return float(margin)


def margin_confidence(normalized_margin: float, scale: float) -> float:
    """Map a [0, 1] class-side margin through the saturating distance transform."""

    if normalized_margin < 0.0 or normalized_margin > 1.0:
        raise RecognitionError("normalized_margin must lie in [0, 1].")
    return normalized_margin / (normalized_margin + scale)


def final_confidence(normalized_margin: float, quality: float, scale: float) -> float:
    """Classifier confidence from normalized margin and input quality. Not a probability."""

    score = quality * margin_confidence(normalized_margin, scale)
    return max(0.0, min(CONFIDENCE_CAP, float(score)))


def _unknown_result(kind: str, reason: str) -> tuple[ObservedValue[bool], str, float]:
    score = 0.2 if kind == "abstention_band" else 0.15
    return unknown(score=score, source=EvidenceSource.CLASSIFIER), reason, score


def _require_int(payload: Mapping[str, object], key: str, *, minimum: int) -> int:
    raw = payload.get(key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise RecognitionError(f"{key} must be an integer (got {raw!r}). Booleans are rejected.")
    if raw < minimum:
        raise RecognitionError(f"{key} must be >= {minimum} (got {raw}). Invalid values are not clamped.")
    return raw


def _require_number(
    payload: Mapping[str, object],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: float | None = None,
    exclusive_maximum: float | None = None,
) -> float:
    raw = payload.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise RecognitionError(f"{key} must be a number (got {raw!r}). Booleans are rejected.")
    value = float(raw)
    if not math.isfinite(value):
        raise RecognitionError(f"{key} must be finite. NaN and infinities are rejected.")
    if minimum is not None and value < minimum:
        raise RecognitionError(f"{key} must be >= {minimum} (got {value}). Invalid values are not clamped.")
    if exclusive_minimum is not None and value <= exclusive_minimum:
        raise RecognitionError(f"{key} must be > {exclusive_minimum} (got {value}). Invalid values are not clamped.")
    if exclusive_maximum is not None and value >= exclusive_maximum:
        raise RecognitionError(f"{key} must be < {exclusive_maximum} (got {value}). Invalid values are not clamped.")
    if maximum is not None and value > maximum:
        raise RecognitionError(f"{key} must be <= {maximum} (got {value}). Invalid values are not clamped.")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RecognitionError(f"Could not write action-state report {path.name}: {exc}.") from exc
