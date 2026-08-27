"""Errors for calibration and region-of-interest mapping."""

from __future__ import annotations


class CalibrationError(Exception):
    """Base error for calibration and ROI failures."""


class InvalidCalibrationError(CalibrationError):
    """The YAML profile is missing fields, out of range, or inconsistent."""


class UnknownRegionError(CalibrationError):
    """A requested region name is not present in the loaded profile."""


class PreviewError(CalibrationError):
    """Drawing or saving a calibration preview failed."""
