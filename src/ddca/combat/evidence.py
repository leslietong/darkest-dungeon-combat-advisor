"""Frame and region evidence without storing raw image pixels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from ddca.combat.errors import InvalidCombatStateError
from ddca.combat.observed import Confidence, as_confidence


def parse_utc_datetime(value: object) -> datetime | None:
    """Parse a timezone-aware UTC datetime from an object or ISO-8601 string."""

    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise InvalidCombatStateError(
                "FrameReference.captured_at must be a UTC datetime, not an empty string."
            )
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            moment = datetime.fromisoformat(text)
        except ValueError as exc:
            raise InvalidCombatStateError(
                f"FrameReference.captured_at is not a valid ISO-8601 datetime (got {value!r})."
            ) from exc
    else:
        raise InvalidCombatStateError(
            "FrameReference.captured_at must be a timezone-aware UTC datetime "
            f"or ISO-8601 string (got {type(value).__name__})."
        )
    if moment.tzinfo is None:
        raise InvalidCombatStateError(
            "FrameReference.captured_at must be timezone-aware UTC. "
            "Naive datetimes are rejected so JSON round-trips stay exact."
        )
    return moment.astimezone(timezone.utc)


def format_utc_datetime(value: datetime) -> str:
    """Serialize a UTC datetime with a trailing Z."""

    text = value.astimezone(timezone.utc).isoformat()
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    return text


@dataclass(frozen=True)
class FrameReference:
    """Identifies a captured frame. Paths are optional; pixels are never stored."""

    capture_id: str
    captured_at: datetime | None = None
    image_path: str | None = None
    image_width: int | None = None
    image_height: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.capture_id, str) or not self.capture_id.strip():
            raise InvalidCombatStateError(
                "FrameReference.capture_id must be a non-empty string. "
                "Use a timestamped capture id such as capture_20260820T030016Z."
            )
        object.__setattr__(self, "capture_id", self.capture_id.strip())
        object.__setattr__(self, "captured_at", parse_utc_datetime(self.captured_at))
        if self.image_path is not None and not isinstance(self.image_path, str):
            raise InvalidCombatStateError("FrameReference.image_path must be a string or None.")
        for name in ("image_width", "image_height"):
            raw = getattr(self, name)
            if raw is None:
                continue
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise InvalidCombatStateError(
                    f"FrameReference.{name} must be a positive integer when set (got {raw!r})."
                )

    def to_dict(self) -> dict[str, object]:
        captured_at = self.captured_at
        return {
            "capture_id": self.capture_id,
            "captured_at": None if captured_at is None else format_utc_datetime(captured_at),
            "image_path": self.image_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FrameReference:
        return cls(
            capture_id=str(payload["capture_id"]),
            captured_at=payload.get("captured_at"),
            image_path=_optional_str(payload.get("image_path")),
            image_width=_optional_int(payload.get("image_width")),
            image_height=_optional_int(payload.get("image_height")),
        )


@dataclass(frozen=True)
class RegionEvidence:
    """A named ROI reference used as evidence. Does not embed image arrays."""

    region_name: str
    x: int
    y: int
    width: int
    height: int
    confidence: Confidence | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.region_name, str) or not self.region_name.strip():
            raise InvalidCombatStateError(
                "RegionEvidence.region_name must be a non-empty string matching a calibration key."
            )
        object.__setattr__(self, "region_name", self.region_name.strip())
        for name in ("x", "y", "width", "height"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise InvalidCombatStateError(
                    f"RegionEvidence.{name} must be an integer pixel value (got {raw!r})."
                )
        if self.x < 0 or self.y < 0:
            raise InvalidCombatStateError(
                f"RegionEvidence origin ({self.x}, {self.y}) must be non-negative."
            )
        if self.width <= 0 or self.height <= 0:
            raise InvalidCombatStateError(
                "RegionEvidence width and height must be positive. "
                "Do not store a zero-size crop as evidence."
            )
        if self.confidence is not None:
            object.__setattr__(self, "confidence", as_confidence(self.confidence))

    def to_dict(self) -> dict[str, object]:
        return {
            "region_name": self.region_name,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": None if self.confidence is None else self.confidence.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RegionEvidence:
        confidence_payload = payload.get("confidence")
        confidence = (
            None
            if confidence_payload is None
            else Confidence.from_dict(confidence_payload)  # type: ignore[arg-type]
        )
        return cls(
            region_name=str(payload["region_name"]),
            x=int(payload["x"]),  # type: ignore[arg-type]
            y=int(payload["y"]),  # type: ignore[arg-type]
            width=int(payload["width"]),  # type: ignore[arg-type]
            height=int(payload["height"]),  # type: ignore[arg-type]
            confidence=confidence,
        )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidCombatStateError(f"Expected an integer or None (got {value!r}).")
    return value
