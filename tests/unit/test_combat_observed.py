"""Unit tests for ObservedValue, Confidence, and evidence envelopes."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ddca.combat.enums import EvidenceSource, ObservationStatus
from ddca.combat.errors import InvalidCombatStateError
from ddca.combat.evidence import FrameReference, RegionEvidence
from ddca.combat.observed import (
    Confidence,
    ObservedValue,
    not_applicable,
    not_visible,
    observed,
    unknown,
)


def test_observed_requires_non_none_value() -> None:
    with pytest.raises(InvalidCombatStateError, match="non-None value"):
        ObservedValue(
            status=ObservationStatus.OBSERVED,
            value=None,
            confidence=Confidence(score=1.0),
        )


@pytest.mark.parametrize(
    "status",
    [
        ObservationStatus.UNKNOWN,
        ObservationStatus.NOT_VISIBLE,
        ObservationStatus.NOT_APPLICABLE,
    ],
)
def test_non_observed_status_requires_none_value(status: ObservationStatus) -> None:
    with pytest.raises(InvalidCombatStateError, match="value=None"):
        ObservedValue(
            status=status,
            value=7,
            confidence=Confidence(score=0.1),
        )


def test_factory_helpers_set_expected_status() -> None:
    assert observed("vestal", score=0.4).status is ObservationStatus.OBSERVED
    assert unknown().status is ObservationStatus.UNKNOWN
    assert not_visible().status is ObservationStatus.NOT_VISIBLE
    assert not_applicable().status is ObservationStatus.NOT_APPLICABLE
    assert unknown().value is None
    assert not_visible().value is None
    assert not_applicable().value is None


def test_unknown_does_not_coerce_to_empty_defaults() -> None:
    field = unknown()
    assert field.value is None
    assert field.value is not False
    assert field.value != 0
    assert field.value != ""
    assert field.value != ()


def test_confidence_accepts_unit_interval() -> None:
    assert Confidence(score=0).score == 0.0
    assert Confidence(score=1).score == 1.0
    assert Confidence(score=0.33).score == pytest.approx(0.33)


@pytest.mark.parametrize("score", [-0.01, 1.01, float("nan"), float("inf"), float("-inf"), True, "0.5"])
def test_confidence_rejects_invalid_scores(score: object) -> None:
    with pytest.raises(InvalidCombatStateError, match="Confidence"):
        Confidence(score=score)  # type: ignore[arg-type]


def test_confidence_from_dict_rejects_bool() -> None:
    with pytest.raises(InvalidCombatStateError, match="Confidence"):
        Confidence.from_dict({"score": True})


def test_observed_value_round_trip_dict() -> None:
    field = observed(12, score=0.5, source=EvidenceSource.OCR)
    restored = ObservedValue.from_dict(field.to_dict(), decode_value=lambda raw: int(raw))  # type: ignore[arg-type]
    assert restored == field


def test_unknown_json_cannot_smuggle_a_value() -> None:
    payload = unknown().to_dict()
    payload["value"] = 5
    with pytest.raises(InvalidCombatStateError, match="value as null"):
        ObservedValue.from_dict(payload)


def test_raw_image_arrays_are_rejected() -> None:
    class FakeImage:
        __array_interface__ = {"shape": (2, 2)}

    with pytest.raises(InvalidCombatStateError, match="image"):
        observed(FakeImage(), score=1.0)


def test_raw_bytes_are_rejected() -> None:
    with pytest.raises(InvalidCombatStateError, match="bytes"):
        observed(b"\x00\x01", score=1.0)


def test_frame_reference_requires_capture_id() -> None:
    with pytest.raises(InvalidCombatStateError, match="capture_id"):
        FrameReference(capture_id="  ")


def test_frame_reference_rejects_naive_datetime() -> None:
    with pytest.raises(InvalidCombatStateError, match="timezone-aware UTC"):
        FrameReference(capture_id="synthetic", captured_at=datetime(2026, 8, 27, 0, 0, 0))


def test_frame_reference_normalizes_iso_string_to_utc_datetime() -> None:
    frame = FrameReference(capture_id="synthetic", captured_at="2026-08-27T03:00:16Z")
    assert frame.captured_at == datetime(2026, 8, 27, 3, 0, 16, tzinfo=timezone.utc)
    assert frame.to_dict()["captured_at"] == "2026-08-27T03:00:16Z"


def test_region_evidence_rejects_empty_size() -> None:
    with pytest.raises(InvalidCombatStateError, match="positive"):
        RegionEvidence(region_name="hero_rank_1", x=0, y=0, width=0, height=10)


def test_region_evidence_round_trip() -> None:
    region = RegionEvidence(
        region_name="skill_slot_1",
        x=304,
        y=461,
        width=42,
        height=42,
        confidence=Confidence(score=0.4),
    )
    assert RegionEvidence.from_dict(region.to_dict()) == region
