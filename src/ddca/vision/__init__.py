"""Calibration and region-of-interest mapping for captured game windows."""

from __future__ import annotations

from ddca.vision.action_state import (
    ACTION_STATE_SCHEMA_VERSION,
    ActionSlotClassification,
    ActionSlotFeatures,
    ActionSlotStateConfig,
    ActionSlotStateReport,
    classify_extraction_dir,
    load_action_slot_state_config,
)
from ddca.vision.calibration import (
    DEFAULT_CALIBRATION_PATH,
    REQUIRED_REGIONS,
    VALIDATION_CONFIRMED,
    VALIDATION_PROVISIONAL,
    CalibrationProfile,
    crop_region,
    load_calibration,
)
from ddca.vision.errors import (
    CalibrationError,
    ExtractionError,
    InvalidCalibrationError,
    PreviewError,
    RecognitionError,
    UnknownRegionError,
)
from ddca.vision.extraction import (
    EXTRACTION_SCHEMA_VERSION,
    CalibrationIdentity,
    ExtractedRegion,
    ExtractionFrame,
    ExtractionManifest,
    extract_regions,
)
from ddca.vision.geometry import NormalizedRect, PixelRect
from ddca.vision.preview import draw_calibration_preview, save_calibration_preview

__all__ = [
    "ACTION_STATE_SCHEMA_VERSION",
    "DEFAULT_CALIBRATION_PATH",
    "EXTRACTION_SCHEMA_VERSION",
    "REQUIRED_REGIONS",
    "VALIDATION_CONFIRMED",
    "VALIDATION_PROVISIONAL",
    "ActionSlotClassification",
    "ActionSlotFeatures",
    "ActionSlotStateConfig",
    "ActionSlotStateReport",
    "CalibrationError",
    "CalibrationIdentity",
    "CalibrationProfile",
    "ExtractedRegion",
    "ExtractionError",
    "ExtractionFrame",
    "ExtractionManifest",
    "InvalidCalibrationError",
    "NormalizedRect",
    "PixelRect",
    "PreviewError",
    "RecognitionError",
    "UnknownRegionError",
    "classify_extraction_dir",
    "crop_region",
    "draw_calibration_preview",
    "extract_regions",
    "load_action_slot_state_config",
    "load_calibration",
    "save_calibration_preview",
]
