"""Calibration and region-of-interest mapping for captured game windows."""

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
    InvalidCalibrationError,
    PreviewError,
    UnknownRegionError,
)
from ddca.vision.geometry import NormalizedRect, PixelRect
from ddca.vision.preview import draw_calibration_preview, save_calibration_preview

__all__ = [
    "DEFAULT_CALIBRATION_PATH",
    "REQUIRED_REGIONS",
    "VALIDATION_CONFIRMED",
    "VALIDATION_PROVISIONAL",
    "CalibrationError",
    "CalibrationProfile",
    "InvalidCalibrationError",
    "NormalizedRect",
    "PixelRect",
    "PreviewError",
    "UnknownRegionError",
    "crop_region",
    "draw_calibration_preview",
    "load_calibration",
    "save_calibration_preview",
]
