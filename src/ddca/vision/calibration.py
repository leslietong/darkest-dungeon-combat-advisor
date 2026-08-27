"""Load, validate, and apply YAML calibration profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ddca.vision.errors import InvalidCalibrationError, UnknownRegionError
from ddca.vision.geometry import NormalizedRect, PixelRect

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only when PyYAML is missing
    raise ImportError(
        "PyYAML is required to load calibration profiles. "
        'Install this project with: python -m pip install -e ".[dev]"'
    ) from exc

SCHEMA_VERSION = 1
COORDINATE_SPACE = "captured_window_normalized"
VALIDATION_CONFIRMED = "confirmed"
VALIDATION_PROVISIONAL = "provisional"
VALID_VALIDATION_STATUSES = frozenset({VALIDATION_CONFIRMED, VALIDATION_PROVISIONAL})

REQUIRED_REGIONS: tuple[str, ...] = (
    "game_frame",
    "combat_area",
    "heroes",
    "hero_rank_4",
    "hero_rank_3",
    "hero_rank_2",
    "hero_rank_1",
    "enemies",
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
)

# Optional regions may appear in YAML. They are never required for schema validity.
OPTIONAL_REGIONS: tuple[str, ...] = ("active_hero_marker",)

# Child must resolve inside parent. Used after relative_to nesting.
CONTAINMENT_RULES: tuple[tuple[str, str], ...] = (
    ("heroes", "game_frame"),
    ("enemies", "game_frame"),
    ("combat_area", "game_frame"),
    ("hero_rank_4", "heroes"),
    ("hero_rank_3", "heroes"),
    ("hero_rank_2", "heroes"),
    ("hero_rank_1", "heroes"),
    ("enemy_rank_1", "enemies"),
    ("enemy_rank_2", "enemies"),
    ("enemy_rank_3", "enemies"),
    ("enemy_rank_4", "enemies"),
    ("hero_health_rank_4", "game_frame"),
    ("hero_health_rank_3", "game_frame"),
    ("hero_health_rank_2", "game_frame"),
    ("hero_health_rank_1", "game_frame"),
    ("enemy_health_rank_1", "game_frame"),
    ("enemy_health_rank_2", "game_frame"),
    ("enemy_health_rank_3", "game_frame"),
    ("enemy_health_rank_4", "game_frame"),
    ("action_bar", "game_frame"),
    ("skill_slot_1", "action_bar"),
    ("skill_slot_2", "action_bar"),
    ("skill_slot_3", "action_bar"),
    ("skill_slot_4", "action_bar"),
    ("move_action_slot", "action_bar"),
    ("round_counter", "game_frame"),
    ("current_hero_panel", "game_frame"),
    ("active_hero_marker", "game_frame"),
)

DEFAULT_CALIBRATION_PATH = Path("configs") / "calibration" / "windowed_1111x654.yaml"


@dataclass(frozen=True)
class CalibrationProfile:
    """A named set of image-normalized regions for one display mode."""

    schema_version: int
    profile_id: str
    game: str
    language: str
    display_mode: str
    reference_width: int
    reference_height: int
    coordinate_space: str
    notes: str
    regions: dict[str, NormalizedRect]
    validation_status: dict[str, str]

    def region(self, name: str) -> NormalizedRect:
        """Return one image-normalized region by name."""

        try:
            return self.regions[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.regions))
            raise UnknownRegionError(
                f"Unknown region {name!r}. Available regions: {known}. "
                "Edit the calibration YAML or use a name from REQUIRED_REGIONS."
            ) from exc

    def pixel_region(self, name: str, image_width: int, image_height: int) -> PixelRect:
        """Map a named region onto a captured image size."""

        return self.region(name).to_pixel(image_width, image_height)

    def region_validation_status(self, name: str) -> str:
        """Return `confirmed` or `provisional` for a region."""

        if name not in self.regions:
            self.region(name)
        return self.validation_status.get(name, VALIDATION_CONFIRMED)

    def is_visually_confirmed(self, name: str) -> bool:
        """Return True only when the region is explicitly confirmed."""

        return self.region_validation_status(name) == VALIDATION_CONFIRMED

    def provisional_region_names(self) -> tuple[str, ...]:
        """Return region names marked provisional."""

        return tuple(
            name
            for name in self.regions
            if self.region_validation_status(name) == VALIDATION_PROVISIONAL
        )

    def size_warnings(self, image_width: int, image_height: int) -> list[str]:
        """Warn when the image is not the one validated resolution."""

        if image_width == self.reference_width and image_height == self.reference_height:
            return []
        return [
            (
                f"Image size {image_width}x{image_height} differs from the validated "
                f"size {self.reference_width}x{self.reference_height} for profile "
                f"{self.profile_id!r}. Normalized coordinates will still be applied, "
                "but Phase 2 only treats the reference size as reliable. "
                "Recapture at the reference size or create a new YAML profile."
            )
        ]


def load_calibration(path: Path) -> CalibrationProfile:
    """Load and validate a calibration YAML file."""

    path = Path(path)
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InvalidCalibrationError(
            f"Could not read calibration file {path.resolve()}: {exc}. "
            f"Create or point --calibration at {DEFAULT_CALIBRATION_PATH}."
        ) from exc

    try:
        payload = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise InvalidCalibrationError(
            f"Invalid YAML in {path.resolve()}: {exc}. "
            "Fix the syntax, then run: python -m ddca.vision.cli validate "
            f"--calibration {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise InvalidCalibrationError(
            f"{path} must contain a YAML mapping at the top level, not {type(payload).__name__}."
        )
    return parse_calibration(payload, source=str(path))


def parse_calibration(payload: Mapping[str, Any], *, source: str) -> CalibrationProfile:
    """Build a profile from an already-parsed mapping."""

    schema_version = _require_int(payload, "schema_version", source=source)
    if schema_version != SCHEMA_VERSION:
        raise InvalidCalibrationError(
            f"{source}: unsupported schema_version {schema_version}. "
            f"This build understands version {SCHEMA_VERSION} only."
        )

    reference = payload.get("reference_size")
    if not isinstance(reference, Mapping):
        raise InvalidCalibrationError(
            f"{source}: reference_size must be a mapping with width and height."
        )
    reference_width = _require_int(reference, "width", source=f"{source}.reference_size")
    reference_height = _require_int(reference, "height", source=f"{source}.reference_size")
    if reference_width <= 0 or reference_height <= 0:
        raise InvalidCalibrationError(
            f"{source}: reference_size width and height must be positive."
        )

    coordinate_space = str(payload.get("coordinate_space", COORDINATE_SPACE))
    if coordinate_space != COORDINATE_SPACE:
        raise InvalidCalibrationError(
            f"{source}: coordinate_space must be {COORDINATE_SPACE!r} "
            f"(got {coordinate_space!r})."
        )

    raw_regions = payload.get("regions")
    if not isinstance(raw_regions, Mapping) or not raw_regions:
        raise InvalidCalibrationError(f"{source}: regions must be a non-empty mapping.")

    resolved, statuses = _resolve_regions(raw_regions, source=source)
    missing = [name for name in REQUIRED_REGIONS if name not in resolved]
    if missing:
        raise InvalidCalibrationError(
            f"{source}: missing required regions: {', '.join(missing)}. "
            "Add them to the YAML file. See REQUIRED_REGIONS in ddca.vision.calibration. "
            "Provisional regions such as active_hero_marker are optional."
        )

    for name, rect in resolved.items():
        rect.validate(context=f"{source} region {name!r}")
        rect.to_pixel(reference_width, reference_height)

    _assert_containment(resolved, source=source)

    return CalibrationProfile(
        schema_version=schema_version,
        profile_id=_require_str(payload, "profile_id", source=source),
        game=_require_str(payload, "game", source=source),
        language=_require_str(payload, "language", source=source),
        display_mode=_require_str(payload, "display_mode", source=source),
        reference_width=reference_width,
        reference_height=reference_height,
        coordinate_space=coordinate_space,
        notes=str(payload.get("notes", "")).strip(),
        regions=resolved,
        validation_status=statuses,
    )


def crop_region(image: NDArray[np.uint8], rect: PixelRect) -> NDArray[np.uint8]:
    """Return a copy of `image` cropped to `rect`."""

    if image.ndim not in (2, 3):
        raise InvalidCalibrationError(
            f"Expected a 2D or 3D image array; received shape {image.shape}."
        )
    height, width = image.shape[:2]
    if rect.right > width or rect.bottom > height:
        raise InvalidCalibrationError(
            f"Crop rectangle ({rect.left},{rect.top}) {rect.width}x{rect.height} "
            f"does not fit in image {width}x{height}."
        )
    rows, cols = rect.numpy_slices()
    return np.ascontiguousarray(image[rows, cols])


def _resolve_regions(
    raw_regions: Mapping[str, Any],
    *,
    source: str,
) -> tuple[dict[str, NormalizedRect], dict[str, str]]:
    locals_by_name: dict[str, tuple[NormalizedRect, str | None]] = {}
    statuses: dict[str, str] = {}
    for name, spec in raw_regions.items():
        if not isinstance(name, str) or not name.strip():
            raise InvalidCalibrationError(f"{source}: region names must be non-empty strings.")
        if not isinstance(spec, Mapping):
            raise InvalidCalibrationError(
                f"{source}: region {name!r} must be a mapping with x, y, width, height."
            )
        relative_to = spec.get("relative_to")
        if relative_to is not None and not isinstance(relative_to, str):
            raise InvalidCalibrationError(
                f"{source}: region {name!r} relative_to must be a string region name."
            )
        status = str(spec.get("validation_status", VALIDATION_CONFIRMED)).strip().lower()
        if status not in VALID_VALIDATION_STATUSES:
            raise InvalidCalibrationError(
                f"{source}: region {name!r} validation_status must be "
                f"'confirmed' or 'provisional' (got {status!r})."
            )
        rect = NormalizedRect(
            x=_require_float(spec, "x", source=f"{source}.regions.{name}"),
            y=_require_float(spec, "y", source=f"{source}.regions.{name}"),
            width=_require_float(spec, "width", source=f"{source}.regions.{name}"),
            height=_require_float(spec, "height", source=f"{source}.regions.{name}"),
        )
        rect.validate(context=f"{source} region {name!r}")
        locals_by_name[name] = (rect, relative_to)
        statuses[name] = status

    resolved: dict[str, NormalizedRect] = {}

    def _resolve(name: str, stack: tuple[str, ...]) -> NormalizedRect:
        if name in resolved:
            return resolved[name]
        if name in stack:
            cycle = " -> ".join(stack + (name,))
            raise InvalidCalibrationError(
                f"{source}: relative_to cycle detected: {cycle}. "
                "Change relative_to so regions form a tree."
            )
        if name not in locals_by_name:
            raise InvalidCalibrationError(
                f"{source}: relative_to {name!r} does not exist. "
                "Create that parent region or fix the name."
            )
        local, parent_name = locals_by_name[name]
        if parent_name is None:
            resolved[name] = local
            return local
        parent = _resolve(parent_name, stack + (name,))
        nested = parent.nest(local)
        nested.validate(context=f"{source} region {name!r} after resolving relative_to")
        resolved[name] = nested
        return nested

    for name in locals_by_name:
        _resolve(name, ())
    return resolved, statuses


def _assert_containment(resolved: Mapping[str, NormalizedRect], *, source: str) -> None:
    for child_name, parent_name in CONTAINMENT_RULES:
        if child_name not in resolved or parent_name not in resolved:
            continue
        parent = resolved[parent_name]
        child = resolved[child_name]
        if not parent.contains(child):
            raise InvalidCalibrationError(
                f"{source}: region {child_name!r} is not contained in {parent_name!r}. "
                "Reduce the child rectangle or enlarge the parent so the child stays inside."
            )


def _require_str(payload: Mapping[str, Any], key: str, *, source: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InvalidCalibrationError(
            f"{source}: missing or empty {key!r}. Add a non-empty string."
        )
    return value.strip()


def _require_int(payload: Mapping[str, Any], key: str, *, source: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidCalibrationError(
            f"{source}: {key!r} must be an integer (got {value!r})."
        )
    return int(value)


def _require_float(payload: Mapping[str, Any], key: str, *, source: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidCalibrationError(
            f"{source}: {key!r} must be a number (got {value!r})."
        )
    return float(value)
