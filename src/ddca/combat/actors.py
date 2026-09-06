"""Hero and enemy combat-state records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from ddca.combat.effects import (
    StatusEffect,
    canonicalize_status_effects,
    decode_status_effects,
)
from ddca.combat.enums import ObservationStatus
from ddca.combat.errors import InvalidCombatStateError
from ddca.combat.observed import ObservedValue
from ddca.combat.ranks import validate_occupied_ranks, validate_rank
from ddca.combat.resources import (
    require_actor_id,
    require_non_empty_str,
    validate_current_maximum,
)


def _canonicalize_observed_effects(
    field: ObservedValue[tuple[StatusEffect, ...]],
) -> ObservedValue[tuple[StatusEffect, ...]]:
    if field.status is not ObservationStatus.OBSERVED or field.value is None:
        return field
    return replace(field, value=canonicalize_status_effects(field.value))


def _decode_rank(raw: object) -> int:
    from ddca.combat.resources import require_int

    return validate_rank(require_int(raw, context="rank"), context="rank")


def _decode_occupied_ranks(raw: object) -> tuple[int, ...]:
    from ddca.combat.resources import require_int

    if not isinstance(raw, list):
        raise InvalidCombatStateError(
            f"occupied_ranks must be a list when observed (got {type(raw).__name__})."
        )
    ranks = [require_int(item, context="occupied_ranks") for item in raw]
    return validate_occupied_ranks(ranks, context="occupied_ranks")


def _decode_class_id(raw: object) -> str:
    return require_non_empty_str(raw, context="class_id")


def _decode_enemy_type_id(raw: object) -> str:
    return require_non_empty_str(raw, context="enemy_type_id")


def _decode_int(raw: object) -> int:
    from ddca.combat.resources import require_int

    return require_int(raw, context="integer field")


@dataclass(frozen=True)
class HeroState:
    """Interpreted hero snapshot. Each hero occupies exactly one rank when observed."""

    actor_id: str
    rank: ObservedValue[int]
    class_id: ObservedValue[str]
    current_hp: ObservedValue[int]
    maximum_hp: ObservedValue[int]
    current_stress: ObservedValue[int]
    maximum_stress: ObservedValue[int]
    status_effects: ObservedValue[tuple[StatusEffect, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", require_actor_id(self.actor_id, context="HeroState.actor_id"))
        if self.rank.status is ObservationStatus.OBSERVED:
            validate_rank(self.rank.value, context=f"HeroState {self.actor_id} rank")  # type: ignore[arg-type]
        if self.class_id.status is ObservationStatus.OBSERVED:
            require_non_empty_str(self.class_id.value, context=f"HeroState {self.actor_id} class_id")
        validate_current_maximum(
            self.current_hp,
            self.maximum_hp,
            resource="HP",
            context=f"HeroState {self.actor_id}",
        )
        validate_current_maximum(
            self.current_stress,
            self.maximum_stress,
            resource="stress",
            context=f"HeroState {self.actor_id}",
        )
        object.__setattr__(self, "status_effects", _canonicalize_observed_effects(self.status_effects))

    def to_dict(self) -> dict[str, object]:
        return {
            "actor_id": self.actor_id,
            "rank": self.rank.to_dict(),
            "class_id": self.class_id.to_dict(),
            "current_hp": self.current_hp.to_dict(),
            "maximum_hp": self.maximum_hp.to_dict(),
            "current_stress": self.current_stress.to_dict(),
            "maximum_stress": self.maximum_stress.to_dict(),
            "status_effects": self.status_effects.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> HeroState:
        if not isinstance(payload, Mapping):
            raise InvalidCombatStateError("HeroState payload must be a mapping.")
        return cls(
            actor_id=str(payload["actor_id"]),
            rank=ObservedValue.from_dict(payload["rank"], decode_value=_decode_rank),  # type: ignore[arg-type]
            class_id=ObservedValue.from_dict(payload["class_id"], decode_value=_decode_class_id),  # type: ignore[arg-type]
            current_hp=ObservedValue.from_dict(payload["current_hp"], decode_value=_decode_int),  # type: ignore[arg-type]
            maximum_hp=ObservedValue.from_dict(payload["maximum_hp"], decode_value=_decode_int),  # type: ignore[arg-type]
            current_stress=ObservedValue.from_dict(payload["current_stress"], decode_value=_decode_int),  # type: ignore[arg-type]
            maximum_stress=ObservedValue.from_dict(payload["maximum_stress"], decode_value=_decode_int),  # type: ignore[arg-type]
            status_effects=ObservedValue.from_dict(
                payload["status_effects"],  # type: ignore[arg-type]
                decode_value=decode_status_effects,
            ),
        )


@dataclass(frozen=True)
class EnemyState:
    """Interpreted enemy snapshot. Large enemies occupy a contiguous rank block."""

    actor_id: str
    occupied_ranks: ObservedValue[tuple[int, ...]]
    enemy_type_id: ObservedValue[str]
    current_hp: ObservedValue[int]
    maximum_hp: ObservedValue[int]
    status_effects: ObservedValue[tuple[StatusEffect, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "actor_id",
            require_actor_id(self.actor_id, context="EnemyState.actor_id"),
        )
        if self.occupied_ranks.status is ObservationStatus.OBSERVED:
            if self.occupied_ranks.value is None:
                raise InvalidCombatStateError(
                    f"EnemyState {self.actor_id}: observed occupied_ranks require a tuple."
                )
            object.__setattr__(
                self,
                "occupied_ranks",
                replace(
                    self.occupied_ranks,
                    value=validate_occupied_ranks(
                        self.occupied_ranks.value,
                        context=f"EnemyState {self.actor_id} occupied_ranks",
                    ),
                ),
            )
        if self.enemy_type_id.status is ObservationStatus.OBSERVED:
            require_non_empty_str(
                self.enemy_type_id.value,
                context=f"EnemyState {self.actor_id} enemy_type_id",
            )
        validate_current_maximum(
            self.current_hp,
            self.maximum_hp,
            resource="HP",
            context=f"EnemyState {self.actor_id}",
        )
        object.__setattr__(self, "status_effects", _canonicalize_observed_effects(self.status_effects))

    def to_dict(self) -> dict[str, object]:
        return {
            "actor_id": self.actor_id,
            "occupied_ranks": self.occupied_ranks.to_dict(),
            "enemy_type_id": self.enemy_type_id.to_dict(),
            "current_hp": self.current_hp.to_dict(),
            "maximum_hp": self.maximum_hp.to_dict(),
            "status_effects": self.status_effects.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EnemyState:
        if not isinstance(payload, Mapping):
            raise InvalidCombatStateError("EnemyState payload must be a mapping.")
        return cls(
            actor_id=str(payload["actor_id"]),
            occupied_ranks=ObservedValue.from_dict(
                payload["occupied_ranks"],  # type: ignore[arg-type]
                decode_value=_decode_occupied_ranks,
            ),
            enemy_type_id=ObservedValue.from_dict(
                payload["enemy_type_id"],  # type: ignore[arg-type]
                decode_value=_decode_enemy_type_id,
            ),
            current_hp=ObservedValue.from_dict(payload["current_hp"], decode_value=_decode_int),  # type: ignore[arg-type]
            maximum_hp=ObservedValue.from_dict(payload["maximum_hp"], decode_value=_decode_int),  # type: ignore[arg-type]
            status_effects=ObservedValue.from_dict(
                payload["status_effects"],  # type: ignore[arg-type]
                decode_value=decode_status_effects,
            ),
        )
