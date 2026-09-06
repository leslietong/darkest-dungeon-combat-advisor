"""Frame-bound combat observations. Pixels stay in capture files, not here."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from ddca.combat.actions import ActionState
from ddca.combat.actors import EnemyState, HeroState
from ddca.combat.effects import StatusEffect
from ddca.combat.enums import ActionKind, ActionSlot
from ddca.combat.errors import InvalidCombatStateError
from ddca.combat.evidence import FrameReference, RegionEvidence
from ddca.combat.observed import ObservedValue

T = TypeVar("T")


def _tuple_evidence(evidence: Sequence[RegionEvidence]) -> tuple[RegionEvidence, ...]:
    return tuple(evidence)


@dataclass(frozen=True)
class Observation(Generic[T]):
    """Generic observation envelope bound to a captured frame, without image arrays."""

    frame: FrameReference
    payload: T
    evidence: tuple[RegionEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.frame, FrameReference):
            raise InvalidCombatStateError("Observation.frame must be a FrameReference.")
        object.__setattr__(self, "evidence", _tuple_evidence(self.evidence))

    def to_dict(self) -> dict[str, object]:
        payload = self.payload
        to_dict = getattr(payload, "to_dict", None)
        if not callable(to_dict):
            raise InvalidCombatStateError(
                "Observation payload must provide to_dict() for JSON serialization."
            )
        return {
            "frame": self.frame.to_dict(),
            "payload": to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
        }


@dataclass(frozen=True)
class HeroObservation:
    """Hero fields read from a frame, including optional ROI evidence."""

    frame: FrameReference
    actor_id: str
    rank: ObservedValue[int]
    class_id: ObservedValue[str]
    current_hp: ObservedValue[int]
    maximum_hp: ObservedValue[int]
    current_stress: ObservedValue[int]
    maximum_stress: ObservedValue[int]
    status_effects: ObservedValue[tuple[StatusEffect, ...]]
    evidence: tuple[RegionEvidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", _tuple_evidence(self.evidence))
        self.to_state()

    def to_state(self) -> HeroState:
        return HeroState(
            actor_id=self.actor_id,
            rank=self.rank,
            class_id=self.class_id,
            current_hp=self.current_hp,
            maximum_hp=self.maximum_hp,
            current_stress=self.current_stress,
            maximum_stress=self.maximum_stress,
            status_effects=self.status_effects,
        )

    def to_dict(self) -> dict[str, object]:
        payload = self.to_state().to_dict()
        return {
            "frame": self.frame.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            **payload,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HeroObservation:
        state = HeroState.from_dict(payload)
        evidence_raw = payload.get("evidence", [])
        if not isinstance(evidence_raw, list):
            raise InvalidCombatStateError("HeroObservation.evidence must be a list.")
        return cls(
            frame=FrameReference.from_dict(payload["frame"]),  # type: ignore[arg-type]
            actor_id=state.actor_id,
            rank=state.rank,
            class_id=state.class_id,
            current_hp=state.current_hp,
            maximum_hp=state.maximum_hp,
            current_stress=state.current_stress,
            maximum_stress=state.maximum_stress,
            status_effects=state.status_effects,
            evidence=tuple(RegionEvidence.from_dict(item) for item in evidence_raw),
        )


@dataclass(frozen=True)
class EnemyObservation:
    """Enemy fields read from a frame. Large enemies use occupied_ranks."""

    frame: FrameReference
    actor_id: str
    occupied_ranks: ObservedValue[tuple[int, ...]]
    enemy_type_id: ObservedValue[str]
    current_hp: ObservedValue[int]
    maximum_hp: ObservedValue[int]
    status_effects: ObservedValue[tuple[StatusEffect, ...]]
    evidence: tuple[RegionEvidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", _tuple_evidence(self.evidence))
        self.to_state()

    def to_state(self) -> EnemyState:
        return EnemyState(
            actor_id=self.actor_id,
            occupied_ranks=self.occupied_ranks,
            enemy_type_id=self.enemy_type_id,
            current_hp=self.current_hp,
            maximum_hp=self.maximum_hp,
            status_effects=self.status_effects,
        )

    def to_dict(self) -> dict[str, object]:
        payload = self.to_state().to_dict()
        return {
            "frame": self.frame.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            **payload,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EnemyObservation:
        state = EnemyState.from_dict(payload)
        evidence_raw = payload.get("evidence", [])
        if not isinstance(evidence_raw, list):
            raise InvalidCombatStateError("EnemyObservation.evidence must be a list.")
        return cls(
            frame=FrameReference.from_dict(payload["frame"]),  # type: ignore[arg-type]
            actor_id=state.actor_id,
            occupied_ranks=state.occupied_ranks,
            enemy_type_id=state.enemy_type_id,
            current_hp=state.current_hp,
            maximum_hp=state.maximum_hp,
            status_effects=state.status_effects,
            evidence=tuple(RegionEvidence.from_dict(item) for item in evidence_raw),
        )


@dataclass(frozen=True)
class ActionObservation:
    """One action-bar slot observed in a frame."""

    frame: FrameReference
    slot: ActionSlot
    kind: ActionKind
    skill_id: ObservedValue[str]
    is_available: ObservedValue[bool]
    legal_target_ranks: ObservedValue[tuple[int, ...]]
    evidence: tuple[RegionEvidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", _tuple_evidence(self.evidence))
        self.to_state()

    def to_state(self) -> ActionState:
        return ActionState(
            slot=self.slot,
            kind=self.kind,
            skill_id=self.skill_id,
            is_available=self.is_available,
            legal_target_ranks=self.legal_target_ranks,
        )

    def to_dict(self) -> dict[str, object]:
        payload = self.to_state().to_dict()
        return {
            "frame": self.frame.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            **payload,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ActionObservation:
        state = ActionState.from_dict(payload)
        evidence_raw = payload.get("evidence", [])
        if not isinstance(evidence_raw, list):
            raise InvalidCombatStateError("ActionObservation.evidence must be a list.")
        return cls(
            frame=FrameReference.from_dict(payload["frame"]),  # type: ignore[arg-type]
            slot=state.slot,
            kind=state.kind,
            skill_id=state.skill_id,
            is_available=state.is_available,
            legal_target_ranks=state.legal_target_ranks,
            evidence=tuple(RegionEvidence.from_dict(item) for item in evidence_raw),
        )
