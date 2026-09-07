"""Crop calibrated ROIs from a static screenshot. No recognition or inference."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from ddca.combat.errors import CombatModelError
from ddca.combat.evidence import FrameReference, RegionEvidence, format_utc_datetime, parse_utc_datetime
from ddca.vision.calibration import (
    CONTAINMENT_RULES,
    VALID_VALIDATION_STATUSES,
    CalibrationProfile,
    crop_region,
)
from ddca.vision.errors import ExtractionError, UnknownRegionError
from ddca.vision.geometry import PixelRect

EXTRACTION_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
CROPS_DIRNAME = "crops"
SAFE_REGION_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
PARENT_BY_REGION: dict[str, str] = {child: parent for child, parent in CONTAINMENT_RULES}


@dataclass(frozen=True)
class ExtractionFrame:
    """Source-image identity for one extraction. Paths are never stored."""

    frame_id: str
    source_image_name: str
    source_image_sha256: str
    source_image_width: int
    source_image_height: int
    captured_at_utc: datetime | None

    def __post_init__(self) -> None:
        if not isinstance(self.frame_id, str) or not SHA256_HEX.fullmatch(self.frame_id):
            raise ExtractionError(
                "ExtractionFrame.frame_id must be the 64-character lowercase SHA-256 "
                "of the exact source image bytes."
            )
        if self.frame_id != self.source_image_sha256:
            raise ExtractionError(
                "ExtractionFrame.frame_id must equal source_image_sha256. "
                "The stable identifier is the source-image digest, not a timestamp."
            )
        object.__setattr__(self, "source_image_name", _require_basename(self.source_image_name, "source_image_name"))
        object.__setattr__(self, "source_image_sha256", _require_sha256(self.source_image_sha256, "source_image_sha256"))
        for name in ("source_image_width", "source_image_height"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise ExtractionError(f"ExtractionFrame.{name} must be a positive integer (got {raw!r}).")
        captured = None if self.captured_at_utc is None else parse_utc_datetime(self.captured_at_utc)
        object.__setattr__(self, "captured_at_utc", captured)

    def to_dict(self) -> dict[str, object]:
        captured = self.captured_at_utc
        return {
            "frame_id": self.frame_id,
            "source_image_name": self.source_image_name,
            "source_image_sha256": self.source_image_sha256,
            "source_image_width": self.source_image_width,
            "source_image_height": self.source_image_height,
            "captured_at_utc": None if captured is None else format_utc_datetime(captured),
        }

    def to_frame_reference(self) -> FrameReference:
        """Project this identity onto the Phase 3A FrameReference fields.

        `image_path` is the source basename only so an absolute user path is never stored.
        `capture_id` is the source-image SHA-256, matching `frame_id`.
        """

        return FrameReference(
            capture_id=self.frame_id,
            captured_at=self.captured_at_utc,
            image_path=self.source_image_name,
            image_width=self.source_image_width,
            image_height=self.source_image_height,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExtractionFrame:
        return cls(
            frame_id=str(payload["frame_id"]),
            source_image_name=str(payload["source_image_name"]),
            source_image_sha256=str(payload["source_image_sha256"]),
            source_image_width=int(payload["source_image_width"]),  # type: ignore[arg-type]
            source_image_height=int(payload["source_image_height"]),  # type: ignore[arg-type]
            captured_at_utc=payload.get("captured_at_utc"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CalibrationIdentity:
    """Calibration file identity. The YAML path is not stored."""

    profile_id: str
    profile_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ExtractionError("CalibrationIdentity.profile_id must be a non-empty string.")
        object.__setattr__(self, "profile_id", self.profile_id.strip())
        object.__setattr__(self, "profile_sha256", _require_sha256(self.profile_sha256, "profile_sha256"))

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "profile_sha256": self.profile_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CalibrationIdentity:
        return cls(
            profile_id=str(payload["profile_id"]),
            profile_sha256=str(payload["profile_sha256"]),
        )


@dataclass(frozen=True)
class ExtractedRegion:
    """One lossless crop plus Phase 3A-compatible region evidence."""

    region_name: str
    relative_to: str | None
    validation_status: str
    pixel: PixelRect
    crop_path: str
    width: int
    height: int
    evidence: RegionEvidence

    def __post_init__(self) -> None:
        if self.region_name != self.evidence.region_name:
            raise ExtractionError(
                f"ExtractedRegion {self.region_name!r} evidence name "
                f"{self.evidence.region_name!r} must match."
            )
        if self.validation_status not in VALID_VALIDATION_STATUSES:
            raise ExtractionError(
                f"ExtractedRegion {self.region_name!r} validation_status must be "
                f"'confirmed' or 'provisional' (got {self.validation_status!r})."
            )
        if self.width != self.pixel.width or self.height != self.pixel.height:
            raise ExtractionError(
                f"ExtractedRegion {self.region_name!r} size {self.width}x{self.height} "
                f"must equal pixel rectangle {self.pixel.width}x{self.pixel.height}."
            )
        if Path(self.crop_path).is_absolute() or ".." in Path(self.crop_path).parts:
            raise ExtractionError(
                f"crop_path {self.crop_path!r} must be a relative path inside the extraction directory."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "region_name": self.region_name,
            "relative_to": self.relative_to,
            "validation_status": self.validation_status,
            "pixel": {
                "left": self.pixel.left,
                "top": self.pixel.top,
                "width": self.pixel.width,
                "height": self.pixel.height,
            },
            "crop_path": self.crop_path,
            "width": self.width,
            "height": self.height,
            "evidence": self.evidence.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExtractedRegion:
        pixel_payload = payload.get("pixel")
        if not isinstance(pixel_payload, Mapping):
            raise ExtractionError("ExtractedRegion.pixel must be a mapping.")
        evidence_payload = payload.get("evidence")
        if not isinstance(evidence_payload, Mapping):
            raise ExtractionError("ExtractedRegion.evidence must be a mapping.")
        relative_to = payload.get("relative_to")
        return cls(
            region_name=str(payload["region_name"]),
            relative_to=None if relative_to is None else str(relative_to),
            validation_status=str(payload["validation_status"]),
            pixel=PixelRect(
                left=int(pixel_payload["left"]),  # type: ignore[arg-type]
                top=int(pixel_payload["top"]),  # type: ignore[arg-type]
                width=int(pixel_payload["width"]),  # type: ignore[arg-type]
                height=int(pixel_payload["height"]),  # type: ignore[arg-type]
            ),
            crop_path=str(payload["crop_path"]),
            width=int(payload["width"]),  # type: ignore[arg-type]
            height=int(payload["height"]),  # type: ignore[arg-type]
            evidence=RegionEvidence.from_dict(evidence_payload),
        )


@dataclass(frozen=True)
class ExtractionManifest:
    """Deterministic evidence manifest for one static extraction. No pixel arrays."""

    schema_version: int
    frame: ExtractionFrame
    calibration: CalibrationIdentity
    extracted_at_utc: datetime
    regions: tuple[ExtractedRegion, ...]

    def __post_init__(self) -> None:
        if self.schema_version != EXTRACTION_SCHEMA_VERSION:
            raise ExtractionError(
                f"Unsupported extraction schema_version {self.schema_version}. "
                f"This build understands version {EXTRACTION_SCHEMA_VERSION} only."
            )
        extracted_at = parse_utc_datetime(self.extracted_at_utc)
        if extracted_at is None:
            raise ExtractionError("ExtractionManifest.extracted_at_utc must be a timezone-aware UTC datetime.")
        object.__setattr__(self, "extracted_at_utc", extracted_at)
        object.__setattr__(self, "regions", tuple(self.regions))
        names = [item.region_name for item in self.regions]
        if len(names) != len(set(names)):
            raise ExtractionError("ExtractionManifest contains duplicate region_name entries.")

    def region(self, name: str) -> ExtractedRegion:
        """Return one extracted region by calibration name."""

        for item in self.regions:
            if item.region_name == name:
                return item
        known = ", ".join(item.region_name for item in self.regions)
        raise ExtractionError(f"Unknown extracted region {name!r}. Available: {known}.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frame": self.frame.to_dict(),
            "calibration": self.calibration.to_dict(),
            "extracted_at_utc": format_utc_datetime(self.extracted_at_utc),
            "regions": [item.to_dict() for item in self.regions],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExtractionManifest:
        regions_raw = payload.get("regions")
        if not isinstance(regions_raw, list):
            raise ExtractionError("ExtractionManifest.regions must be a list.")
        frame_payload = payload.get("frame")
        if not isinstance(frame_payload, Mapping):
            raise ExtractionError("ExtractionManifest.frame must be a mapping.")
        calibration_payload = payload.get("calibration")
        if not isinstance(calibration_payload, Mapping):
            raise ExtractionError("ExtractionManifest.calibration must be a mapping.")
        extracted_at = parse_utc_datetime(payload.get("extracted_at_utc"))
        if extracted_at is None:
            raise ExtractionError("ExtractionManifest.extracted_at_utc is required.")
        return cls(
            schema_version=int(payload["schema_version"]),  # type: ignore[arg-type]
            frame=ExtractionFrame.from_dict(frame_payload),
            calibration=CalibrationIdentity.from_dict(calibration_payload),
            extracted_at_utc=extracted_at,
            regions=tuple(ExtractedRegion.from_dict(item) for item in regions_raw),
        )


def extract_regions(
    image: NDArray[np.uint8],
    profile: CalibrationProfile,
    output_dir: Path,
    *,
    selected_regions: Sequence[str] | None = None,
    calibration_path: Path | str,
    source_image_path: Path | str,
    overwrite: bool = False,
    extracted_at: datetime | None = None,
) -> ExtractionManifest:
    """Crop configured ROIs from `image` and write a deterministic evidence manifest.

    This function copies pixels only. It does not identify heroes, skills, HP, or
    any other game value, and it does not build a CombatState.

    All requested regions are resolved and copied in memory before any crop or
    manifest file is committed. The final `manifest.json` is written last and
    only after every planned PNG succeeds.

    `frame_id` is the SHA-256 of the exact source image file bytes. That same
    digest is stored as `source_image_sha256`. `captured_at_utc` is taken from a
    sibling capture-metadata JSON only after that sidecar is shown to belong to
    this image; otherwise it is null and is never replaced with extraction time.
    """

    height, width = _require_extractable_image(image, profile)
    source_path = _require_existing_file(source_image_path, "source_image_path")
    profile_path = _require_existing_file(calibration_path, "calibration_path")
    names = _selected_region_names(profile, selected_regions)
    prepared = tuple(_prepare_region(image, profile, name) for name in names)

    output_dir = Path(output_dir)
    crops_dir = output_dir / CROPS_DIRNAME
    planned = _planned_paths(output_dir, names)
    _assert_overwrite_allowed(planned, overwrite=overwrite)

    source_sha256 = sha256_file(source_path)
    captured_at = _captured_at_from_sidecar(source_path, width=width, height=height)
    frame = ExtractionFrame(
        frame_id=source_sha256,
        source_image_name=source_path.name,
        source_image_sha256=source_sha256,
        source_image_width=width,
        source_image_height=height,
        captured_at_utc=captured_at,
    )
    calibration = CalibrationIdentity(
        profile_id=profile.profile_id,
        profile_sha256=sha256_file(profile_path),
    )
    extracted = tuple(_extracted_region(item) for item in prepared)
    manifest = ExtractionManifest(
        schema_version=EXTRACTION_SCHEMA_VERSION,
        frame=frame,
        calibration=calibration,
        extracted_at_utc=extracted_at or datetime.now(timezone.utc),
        regions=extracted,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    for item in prepared:
        _write_crop_png_atomic(output_dir / item.crop_path, item.crop)
    _write_json_atomic(output_dir / MANIFEST_FILENAME, manifest.to_dict())
    return manifest


def resolve_pixel_rect(
    profile: CalibrationProfile,
    name: str,
    image_width: int,
    image_height: int,
) -> PixelRect:
    """Map a named region to pixels using the same rounding as Phase 2, without clamping."""

    normalized = profile.region(name)
    left = int(round(normalized.x * image_width))
    top = int(round(normalized.y * image_height))
    width = int(round(normalized.width * image_width))
    height = int(round(normalized.height * image_height))
    if width <= 0 or height <= 0:
        raise ExtractionError(
            f"Region {name!r} mapped to an empty pixel rectangle "
            f"{width}x{height} on a {image_width}x{image_height} image. "
            "The extractor does not pad or enlarge empty geometry."
        )
    if left < 0 or top < 0 or left + width > image_width or top + height > image_height:
        raise ExtractionError(
            f"Region {name!r} pixel rectangle ({left},{top}) {width}x{height} "
            f"is outside the {image_width}x{image_height} image. "
            "The extractor does not clamp invalid rectangles."
        )
    return PixelRect(left=left, top=top, width=width, height=height)


def independent_crop(image: NDArray[np.uint8], pixel: PixelRect) -> NDArray[np.uint8]:
    """Return a crop that keeps dtype and channel order and does not share writable memory."""

    crop = np.array(crop_region(image, pixel), copy=True, order="C")
    if crop.dtype != image.dtype:
        raise ExtractionError(
            f"Crop dtype {crop.dtype} does not match source dtype {image.dtype}. "
            "The extractor does not convert pixels."
        )
    if crop.ndim != image.ndim:
        raise ExtractionError(
            f"Crop ndim {crop.ndim} does not match source ndim {image.ndim}. "
            "The extractor does not convert channel order."
        )
    if image.ndim == 3 and crop.shape[2] != image.shape[2]:
        raise ExtractionError(
            f"Crop channel count {crop.shape[2]} does not match source channel count "
            f"{image.shape[2]}. The extractor does not convert channel order."
        )
    if np.shares_memory(crop, image):
        crop = np.array(crop, copy=True, order="C")
    if np.shares_memory(crop, image):
        raise ExtractionError("Crop still shares writable memory with the source image.")
    return crop


def sha256_file(path: Path | str) -> str:
    """Return the lowercase SHA-256 hex digest of the exact file bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def crop_filename(region_name: str) -> str:
    """Return a filesystem-safe PNG filename for a validated region name."""

    if not isinstance(region_name, str) or not SAFE_REGION_NAME.fullmatch(region_name):
        raise ExtractionError(
            f"Region name {region_name!r} is not a safe crop filename. "
            "Use calibration keys such as skill_slot_1."
        )
    if ".." in region_name or "/" in region_name or "\\" in region_name:
        raise ExtractionError(f"Region name {region_name!r} must not contain path separators.")
    return f"{region_name}.png"


def _relative_crop_path(region_name: str) -> str:
    return f"{CROPS_DIRNAME}/{crop_filename(region_name)}"


@dataclass(frozen=True)
class _PreparedRegion:
    name: str
    pixel: PixelRect
    crop: NDArray[np.uint8]
    relative_to: str | None
    validation_status: str
    crop_path: str


def _prepare_region(image: NDArray[np.uint8], profile: CalibrationProfile, name: str) -> _PreparedRegion:
    pixel = resolve_pixel_rect(profile, name, image.shape[1], image.shape[0])
    crop = independent_crop(image, pixel)
    if crop.size == 0:
        raise ExtractionError(
            f"Region {name!r} produced an empty crop. "
            "The extractor does not pad or resize invalid rectangles."
        )
    if crop.shape[0] != pixel.height or crop.shape[1] != pixel.width:
        raise ExtractionError(
            f"Region {name!r} crop shape {crop.shape[:2]} does not match "
            f"resolved rectangle {pixel.width}x{pixel.height}."
        )
    status = profile.region_validation_status(name)
    if status not in VALID_VALIDATION_STATUSES:
        raise ExtractionError(
            f"Region {name!r} has unsupported validation_status {status!r}. "
            "Use the calibration statuses 'confirmed' or 'provisional'."
        )
    return _PreparedRegion(
        name=name,
        pixel=pixel,
        crop=crop,
        relative_to=PARENT_BY_REGION.get(name),
        validation_status=status,
        crop_path=_relative_crop_path(name),
    )


def _extracted_region(item: _PreparedRegion) -> ExtractedRegion:
    return ExtractedRegion(
        region_name=item.name,
        relative_to=item.relative_to,
        validation_status=item.validation_status,
        pixel=item.pixel,
        crop_path=item.crop_path,
        width=item.pixel.width,
        height=item.pixel.height,
        evidence=RegionEvidence(
            region_name=item.name,
            x=item.pixel.left,
            y=item.pixel.top,
            width=item.pixel.width,
            height=item.pixel.height,
        ),
    )


def _selected_region_names(
    profile: CalibrationProfile,
    selected_regions: Sequence[str] | None,
) -> tuple[str, ...]:
    if selected_regions is None:
        names = tuple(sorted(profile.regions))
        for name in names:
            crop_filename(name)
        return names
    unique: list[str] = []
    seen: set[str] = set()
    for name in selected_regions:
        crop_filename(name)
        if name not in profile.regions:
            raise UnknownRegionError(
                f"Unknown region {name!r}. Available regions: {', '.join(sorted(profile.regions))}."
            )
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return tuple(unique)


def _require_extractable_image(image: NDArray[np.uint8], profile: CalibrationProfile) -> tuple[int, int]:
    if not isinstance(image, np.ndarray):
        raise ExtractionError("extract_regions requires a NumPy image array.")
    if image.size == 0 or image.ndim not in (2, 3):
        raise ExtractionError(
            f"Source image must be a non-empty 2D or 3D array (got shape {getattr(image, 'shape', None)})."
        )
    if image.dtype != np.uint8:
        raise ExtractionError(
            f"Source image must keep uint8 channel values (got dtype {image.dtype}). "
            "The extractor does not convert or enhance pixels."
        )
    height, width = image.shape[:2]
    if width != profile.reference_width or height != profile.reference_height:
        raise ExtractionError(
            f"Image size {width}x{height} does not match the profile reference size "
            f"{profile.reference_width}x{profile.reference_height} for {profile.profile_id!r}. "
            "Recapture at the accepted size or use a matching calibration profile. "
            "The extractor does not rescale the source image."
        )
    return height, width


def _require_existing_file(value: Path | str, label: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise ExtractionError(
            f"{label} is required and must be an existing file so the manifest "
            f"can hash the exact bytes (got {path.name!r})."
        )
    return path


def _require_basename(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != Path(value).name:
        raise ExtractionError(f"{label} must be a file basename, not a path (got {value!r}).")
    if Path(value).is_absolute() or ".." in Path(value).parts or "/" in value or "\\" in value:
        raise ExtractionError(f"{label} must be a file basename, not a path (got {value!r}).")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_HEX.fullmatch(value):
        raise ExtractionError(f"{label} must be a 64-character lowercase SHA-256 hex digest.")
    return value


def _captured_at_from_sidecar(image_path: Path, *, width: int, height: int) -> datetime | None:
    """Return the capture timestamp only when the sidecar is shown to belong to this image."""

    sidecar = image_path.with_suffix(".json")
    if not sidecar.is_file() or sidecar.stem != image_path.stem:
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    dimensions = payload.get("image_dimensions")
    if not isinstance(dimensions, Mapping):
        return None
    try:
        sidecar_width = int(dimensions["width"])  # type: ignore[arg-type]
        sidecar_height = int(dimensions["height"])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError):
        return None
    if sidecar_width != width or sidecar_height != height:
        return None
    raw = payload.get("captured_at_utc")
    if raw is None:
        return None
    try:
        captured = parse_utc_datetime(raw)
    except CombatModelError:
        return None
    return captured


def _planned_paths(output_dir: Path, names: Sequence[str]) -> tuple[Path, ...]:
    crops = [output_dir / _relative_crop_path(name) for name in names]
    return (output_dir / MANIFEST_FILENAME, *crops)


def _assert_overwrite_allowed(paths: Sequence[Path], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        listed = ", ".join(path.name for path in existing[:5])
        raise ExtractionError(
            f"Refusing to overwrite existing extraction files ({listed}). "
            "Pass overwrite=True or --overwrite to replace only this extraction's files."
        )


def _write_crop_png_atomic(path: Path, crop: NDArray[np.uint8]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV selects the encoder from the suffix; the temp name must end in .png.
    tmp = path.with_name(f".{path.name}.tmp.png")
    try:
        try:
            ok = cv2.imwrite(str(tmp), crop)
        except cv2.error as exc:
            raise ExtractionError(
                f"Failed to write lossless crop PNG {path.name}: {exc}. "
                "Check that the path is writable and the disk is not full."
            ) from exc
        if not ok or not tmp.is_file() or tmp.stat().st_size <= 0:
            raise ExtractionError(
                f"Failed to write lossless crop PNG {path.name}. "
                "Check that the path is writable and the disk is not full."
            )
        tmp.replace(path)
    except ExtractionError:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    except OSError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise ExtractionError(f"Could not write crop PNG {path.name}: {exc}.") from exc


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise ExtractionError(f"Could not write manifest {path.name}: {exc}.") from exc
