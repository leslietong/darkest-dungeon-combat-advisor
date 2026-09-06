"""Action-bar slots: four skills plus Move."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from ddca.combat.enums import ActionKind, ActionSlot, ObservationStatus, action_kind_for_slot
from ddca.combat.errors import InvalidCombatStateError
from ddca.combat.observed import ObservedValue
from ddca.combat.ranks import validate_rank
from ddca.combat.resources import require_int, require_non_empty_str

SLOT_ORDER: tuple[ActionSlot, ...] = (
    ActionSlot.SKILL_1,
    ActionSlot.SKILL_2,
    ActionSlot.SKILL_3,
    ActionSlot.SKILL_4,
    ActionSlot.MOVE,
)


def _decode_skill_id(raw: object) -> str:
    return require_non_empty_str(raw, context="skill_id")


def _decode_bool(raw: object) -> bool:
    if not isinstance(raw, bool):
        raise InvalidCombatStateError(f"Boolean field must be true or false (got {raw!r}).")
    return raw


def _decode_target_ranks(raw: object) -> tuple[int, ...]:
    if not isinstance(raw, list):
        raise InvalidCombatStateError(
            f"legal_target_ranks must be a list when observed (got {type(raw).__name__})."
        )
    ranks = tuple(validate_rank(require_int(item, context="legal_target_ranks"), context="legal_target_ranks") for item in raw)
    if len(set(ranks)) != len(ranks):
        raise InvalidCombatStateError(
            f"legal_target_ranks contains duplicates {ranks}. List each legal rank once."
        )
    return tuple(sorted(ranks))


@dataclass(frozen=True)
class ActionState:
    """One action-bar slot. Move is a first-class legal action, not a skill."""

    slot: ActionSlot
    kind: ActionKind
    skill_id: ObservedValue[str]
    is_available: ObservedValue[bool]
    legal_target_ranks: ObservedValue[tuple[int, ...]]

    @property
    def ui_slot(self) -> ActionSlot:
        """Calibration slot name. Alias of ``slot`` for lookup by ui_slot."""

        return self.slot

    @property
    def action_id(self) -> str:
        """Stable action identity: the Phase 2 slot string."""

        return self.slot.value

    def __post_init__(self) -> None:
        if not isinstance(self.slot, ActionSlot):
            raise InvalidCombatStateError(f"ActionState.slot must be an ActionSlot (got {self.slot!r}).")
        if not isinstance(self.kind, ActionKind):
            raise InvalidCombatStateError(f"ActionState.kind must be an ActionKind (got {self.kind!r}).")
        expected = action_kind_for_slot(self.slot)
        if self.kind is not expected:
            raise InvalidCombatStateError(
                f"ActionState slot {self.slot.value} requires kind {expected.value} "
                f"(got {self.kind.value}). Skill slots are skills; move_action_slot is Move."
            )
        if self.slot is ActionSlot.MOVE and self.skill_id.status is ObservationStatus.OBSERVED:
            raise InvalidCombatStateError(
                "Move actions cannot observe a skill_id. Use status 'not_applicable'."
            )
        if (
            self.slot is not ActionSlot.MOVE
            and self.skill_id.status is ObservationStatus.NOT_APPLICABLE
        ):
            raise InvalidCombatStateError(
                f"Skill slot {self.slot.value} cannot mark skill_id as not_applicable. "
                "Use 'unknown' when the icon is not yet recognized."
            )
        if self.skill_id.status is ObservationStatus.OBSERVED:
            require_non_empty_str(self.skill_id.value, context=f"ActionState {self.slot.value} skill_id")
        if self.legal_target_ranks.status is ObservationStatus.OBSERVED:
            if self.legal_target_ranks.value is None:
                raise InvalidCombatStateError("Observed legal_target_ranks require a tuple.")
            object.__setattr__(
                self,
                "legal_target_ranks",
                replace(
                    self.legal_target_ranks,
                    value=_decode_target_ranks(list(self.legal_target_ranks.value)),
                ),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "slot": self.slot.value,
            "kind": self.kind.value,
            "skill_id": self.skill_id.to_dict(),
            "is_available": self.is_available.to_dict(),
            "legal_target_ranks": self.legal_target_ranks.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ActionState:
        if not isinstance(payload, Mapping):
            raise InvalidCombatStateError("ActionState payload must be a mapping.")
        try:
            slot = ActionSlot(str(payload["slot"]))
            kind = ActionKind(str(payload["kind"]))
        except (KeyError, ValueError) as exc:
            raise InvalidCombatStateError(
                "ActionState requires valid slot and kind strings matching the calibration names."
            ) from exc
        return cls(
            slot=slot,
            kind=kind,
            skill_id=ObservedValue.from_dict(payload["skill_id"], decode_value=_decode_skill_id),  # type: ignore[arg-type]
            is_available=ObservedValue.from_dict(payload["is_available"], decode_value=_decode_bool),  # type: ignore[arg-type]
            legal_target_ranks=ObservedValue.from_dict(
                payload["legal_target_ranks"],  # type: ignore[arg-type]
                decode_value=_decode_target_ranks,
            ),
        )
