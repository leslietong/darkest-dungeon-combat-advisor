"""Per-field observed values with status, confidence, and evidence source."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

from ddca.combat.enums import EvidenceSource, ObservationStatus
from ddca.combat.errors import InvalidCombatStateError

T = TypeVar("T")


@dataclass(frozen=True)
class Confidence:
    """A probability-like score in ``[0, 1]``. Not a proven accuracy."""

    score: float

    def __post_init__(self) -> None:
        score = self.score
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise InvalidCombatStateError(
                f"Confidence score must be a number in [0, 1] (got {score!r})."
            )
        numeric = float(score)
        if not math.isfinite(numeric):
            raise InvalidCombatStateError(
                "Confidence score must be finite. NaN and infinities are not valid scores."
            )
        if numeric < 0.0 or numeric > 1.0:
            raise InvalidCombatStateError(
                f"Confidence score {numeric} is outside [0, 1]. "
                "Use 0 for no confidence and 1 for complete confidence under the current model."
            )
        object.__setattr__(self, "score", numeric)

    def to_dict(self) -> dict[str, float]:
        return {"score": self.score}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Confidence:
        if not isinstance(payload, Mapping) or "score" not in payload:
            raise InvalidCombatStateError("Confidence mapping must include a numeric 'score'.")
        score = payload["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise InvalidCombatStateError(
                f"Confidence score must be a number in [0, 1] (got {score!r})."
            )
        return cls(score=float(score))


def as_confidence(value: Confidence | float) -> Confidence:
    """Accept a Confidence or a raw score."""

    if isinstance(value, Confidence):
        return value
    return Confidence(score=value)


@dataclass(frozen=True)
class ObservedValue(Generic[T]):
    """A typed field that keeps unknown / not-visible / not-applicable explicit."""

    status: ObservationStatus
    value: T | None
    confidence: Confidence
    source: EvidenceSource = EvidenceSource.UNKNOWN

    def __post_init__(self) -> None:
        if not isinstance(self.status, ObservationStatus):
            raise InvalidCombatStateError(
                f"status must be an ObservationStatus (got {self.status!r})."
            )
        if not isinstance(self.source, EvidenceSource):
            raise InvalidCombatStateError(
                f"source must be an EvidenceSource (got {self.source!r})."
            )
        if not isinstance(self.confidence, Confidence):
            object.__setattr__(self, "confidence", as_confidence(self.confidence))
        _reject_raw_image(self.value)
        if self.status is ObservationStatus.OBSERVED:
            if self.value is None:
                raise InvalidCombatStateError(
                    "ObservedValue with status 'observed' requires a non-None value. "
                    "Use status 'unknown' when the value is not known."
                )
        elif self.value is not None:
            raise InvalidCombatStateError(
                f"ObservedValue with status {self.status.value!r} requires value=None "
                f"(got {self.value!r}). Do not store a guessed value for an unobserved field."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "value": _encode_value(self.value),
            "confidence": self.confidence.to_dict(),
            "source": self.source.value,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        decode_value: Callable[[object], T] | None = None,
    ) -> ObservedValue[T]:
        if not isinstance(payload, Mapping):
            raise InvalidCombatStateError("ObservedValue payload must be a mapping.")
        try:
            status = ObservationStatus(str(payload["status"]))
        except (KeyError, ValueError) as exc:
            raise InvalidCombatStateError(
                "ObservedValue.status must be observed, unknown, not_visible, or not_applicable."
            ) from exc
        raw_value = payload.get("value")
        decoder = decode_value if decode_value is not None else (lambda item: item)
        value: T | None
        if status is ObservationStatus.OBSERVED:
            value = decoder(raw_value)
        else:
            if raw_value is not None:
                raise InvalidCombatStateError(
                    f"ObservedValue with status {status.value!r} must serialize value as null."
                )
            value = None
        confidence_payload = payload.get("confidence", {"score": 0.0})
        if not isinstance(confidence_payload, Mapping):
            raise InvalidCombatStateError("confidence must be a mapping with a score.")
        source_raw = payload.get("source", EvidenceSource.UNKNOWN.value)
        try:
            source = EvidenceSource(str(source_raw))
        except ValueError as exc:
            raise InvalidCombatStateError(
                f"Unknown evidence source {source_raw!r}."
            ) from exc
        return cls(
            status=status,
            value=value,
            confidence=Confidence.from_dict(confidence_payload),
            source=source,
        )


def observed(
    value: T,
    *,
    score: float,
    source: EvidenceSource = EvidenceSource.MANUAL,
) -> ObservedValue[T]:
    """Build an observed field."""

    return ObservedValue(
        status=ObservationStatus.OBSERVED,
        value=value,
        confidence=Confidence(score=score),
        source=source,
    )


def unknown(
    *,
    score: float = 0.0,
    source: EvidenceSource = EvidenceSource.UNKNOWN,
) -> ObservedValue[T]:
    """Build an unknown field with no stored value."""

    return ObservedValue(
        status=ObservationStatus.UNKNOWN,
        value=None,
        confidence=Confidence(score=score),
        source=source,
    )


def not_visible(
    *,
    score: float = 0.0,
    source: EvidenceSource = EvidenceSource.UNKNOWN,
) -> ObservedValue[T]:
    """Build a field that was not visible in the current frame."""

    return ObservedValue(
        status=ObservationStatus.NOT_VISIBLE,
        value=None,
        confidence=Confidence(score=score),
        source=source,
    )


def not_applicable(
    *,
    score: float = 1.0,
    source: EvidenceSource = EvidenceSource.RULE_ENGINE,
) -> ObservedValue[T]:
    """Build a field that does not apply to this actor or action."""

    return ObservedValue(
        status=ObservationStatus.NOT_APPLICABLE,
        value=None,
        confidence=Confidence(score=score),
        source=source,
    )


def _reject_raw_image(value: object) -> None:
    """Keep Observation and CombatState free of screenshot pixels."""

    if value is None or isinstance(value, (str, int, float, bool, tuple)):
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise InvalidCombatStateError(
            "Observed values cannot store raw bytes or image buffers. "
            "Keep pixels in capture files and reference them with FrameReference."
        )
    if hasattr(value, "__array_interface__") or hasattr(value, "__array__"):
        raise InvalidCombatStateError(
            "Observed values cannot store image arrays. "
            "Phase 3A models structured combat state, not screenshots."
        )


def _encode_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, tuple):
        return [_encode_value(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, (str, int, float, bool)):
        return value
    raise InvalidCombatStateError(
        f"Cannot serialize observed value of type {type(value).__name__}. "
        "Combat models must not store raw images or other non-JSON values."
    )
