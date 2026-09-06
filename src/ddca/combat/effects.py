"""Status-effect records identified by stable catalog strings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ddca.combat.errors import InvalidCombatStateError
from ddca.combat.observed import ObservedValue
from ddca.combat.resources import require_non_empty_str, validate_non_negative_resource


@dataclass(frozen=True)
class StatusEffect:
    """One status effect on an actor. IDs are stable strings, not a closed enum."""

    effect_id: str
    stacks: ObservedValue[int]
    duration_rounds: ObservedValue[int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "effect_id",
            require_non_empty_str(self.effect_id, context="StatusEffect.effect_id"),
        )
        validate_non_negative_resource(self.stacks, context=f"StatusEffect {self.effect_id} stacks")
        validate_non_negative_resource(
            self.duration_rounds,
            context=f"StatusEffect {self.effect_id} duration_rounds",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "effect_id": self.effect_id,
            "stacks": self.stacks.to_dict(),
            "duration_rounds": self.duration_rounds.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> StatusEffect:
        if not isinstance(payload, Mapping):
            raise InvalidCombatStateError("StatusEffect payload must be a mapping.")
        return cls(
            effect_id=str(payload["effect_id"]),
            stacks=ObservedValue.from_dict(payload["stacks"], decode_value=_decode_int),  # type: ignore[arg-type]
            duration_rounds=ObservedValue.from_dict(
                payload["duration_rounds"],  # type: ignore[arg-type]
                decode_value=_decode_int,
            ),
        )


def decode_status_effects(raw: object) -> tuple[StatusEffect, ...]:
    """Decode a JSON list of status effects into a tuple."""

    if not isinstance(raw, list):
        raise InvalidCombatStateError(
            f"status_effects must be a list when observed (got {type(raw).__name__})."
        )
    effects = tuple(StatusEffect.from_dict(item) for item in raw)
    return canonicalize_status_effects(effects)


def canonicalize_status_effects(effects: Sequence[StatusEffect]) -> tuple[StatusEffect, ...]:
    """Sort by effect_id and reject duplicate catalog IDs on the same actor."""

    ordered = tuple(sorted(effects, key=lambda effect: effect.effect_id))
    seen: set[str] = set()
    for effect in ordered:
        if effect.effect_id in seen:
            raise InvalidCombatStateError(
                f"Duplicate status effect {effect.effect_id!r} on the same actor. "
                "Collapse stacks into one StatusEffect instead of repeating the id."
            )
        seen.add(effect.effect_id)
    return ordered


def _decode_int(raw: object) -> int:
    from ddca.combat.resources import require_int

    return require_int(raw, context="integer field")
