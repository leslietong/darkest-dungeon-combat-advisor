"""Aggregated combat snapshot and party-level invariants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ddca.combat.actions import SLOT_ORDER, ActionState
from ddca.combat.actors import EnemyState, HeroState
from ddca.combat.enums import ActionSlot, ObservationStatus
from ddca.combat.errors import InvalidCombatStateError
from ddca.combat.evidence import FrameReference
from ddca.combat.observed import ObservedValue
from ddca.combat.resources import require_int, validate_non_negative_resource
from ddca.combat.schema import COMBAT_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS


def _decode_round(raw: object) -> int:
    value = require_int(raw, context="round_number")
    if value < 1:
        raise InvalidCombatStateError(
            f"round_number must be >= 1 when observed (got {value})."
        )
    return value


def _decode_actor_id(raw: object) -> str:
    from ddca.combat.resources import require_actor_id

    return require_actor_id(raw, context="active_actor_id")


def _hero_sort_key(hero: HeroState) -> tuple[int, str]:
    rank = hero.rank.value if hero.rank.status is ObservationStatus.OBSERVED and hero.rank.value is not None else 99
    return (rank, hero.actor_id)


def _enemy_sort_key(enemy: EnemyState) -> tuple[int, str]:
    if (
        enemy.occupied_ranks.status is ObservationStatus.OBSERVED
        and enemy.occupied_ranks.value
    ):
        return (min(enemy.occupied_ranks.value), enemy.actor_id)
    return (99, enemy.actor_id)


def _action_sort_key(action: ActionState) -> int:
    try:
        return SLOT_ORDER.index(action.slot)
    except ValueError:
        return 99


@dataclass(frozen=True)
class CombatState:
    """Validated combat snapshot. Per-field confidence lives on each ObservedValue."""

    schema_version: int
    frame: FrameReference
    round_number: ObservedValue[int]
    heroes: tuple[HeroState, ...]
    enemies: tuple[EnemyState, ...]
    actions: tuple[ActionState, ...]
    active_actor_id: ObservedValue[str]

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise InvalidCombatStateError(
                f"schema_version must be an integer (got {self.schema_version!r})."
            )
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise InvalidCombatStateError(
                f"Unsupported combat schema_version {self.schema_version}. "
                f"Supported: {SUPPORTED_SCHEMA_VERSIONS}. "
                "This number is independent of the ddca package version."
            )
        if not isinstance(self.frame, FrameReference):
            raise InvalidCombatStateError("CombatState.frame must be a FrameReference.")
        validate_non_negative_resource(self.round_number, context="CombatState.round_number")
        if self.round_number.status is ObservationStatus.OBSERVED:
            _decode_round(self.round_number.value)

        heroes = tuple(sorted(self.heroes, key=_hero_sort_key))
        enemies = tuple(sorted(self.enemies, key=_enemy_sort_key))
        actions = tuple(sorted(self.actions, key=_action_sort_key))
        object.__setattr__(self, "heroes", heroes)
        object.__setattr__(self, "enemies", enemies)
        object.__setattr__(self, "actions", actions)

        _validate_unique_actor_ids(heroes, enemies)
        _validate_unique_hero_ranks(heroes)
        _validate_enemy_rank_coverage(enemies)
        _validate_unique_action_slots(actions)
        _validate_active_actor(self.active_actor_id, heroes, enemies)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "frame": self.frame.to_dict(),
            "round_number": self.round_number.to_dict(),
            "heroes": [hero.to_dict() for hero in self.heroes],
            "enemies": [enemy.to_dict() for enemy in self.enemies],
            "actions": [action.to_dict() for action in self.actions],
            "active_actor_id": self.active_actor_id.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CombatState:
        if not isinstance(payload, Mapping):
            raise InvalidCombatStateError("CombatState payload must be a mapping.")
        heroes_raw = payload.get("heroes", [])
        enemies_raw = payload.get("enemies", [])
        actions_raw = payload.get("actions", [])
        if not isinstance(heroes_raw, list) or not isinstance(enemies_raw, list) or not isinstance(actions_raw, list):
            raise InvalidCombatStateError("heroes, enemies, and actions must be lists.")
        return cls(
            schema_version=require_int(payload.get("schema_version"), context="schema_version"),
            frame=FrameReference.from_dict(payload["frame"]),  # type: ignore[arg-type]
            round_number=ObservedValue.from_dict(payload["round_number"], decode_value=_decode_round),  # type: ignore[arg-type]
            heroes=tuple(HeroState.from_dict(item) for item in heroes_raw),
            enemies=tuple(EnemyState.from_dict(item) for item in enemies_raw),
            actions=tuple(ActionState.from_dict(item) for item in actions_raw),
            active_actor_id=ObservedValue.from_dict(
                payload["active_actor_id"],  # type: ignore[arg-type]
                decode_value=_decode_actor_id,
            ),
        )

    @classmethod
    def from_observations(
        cls,
        *,
        frame: FrameReference,
        round_number: ObservedValue[int],
        heroes: Sequence[object],
        enemies: Sequence[object],
        actions: Sequence[object],
        active_actor_id: ObservedValue[str],
        schema_version: int = COMBAT_SCHEMA_VERSION,
    ) -> CombatState:
        """Build a CombatState from observation objects or already-built states."""

        return cls(
            schema_version=schema_version,
            frame=frame,
            round_number=round_number,
            heroes=tuple(_as_hero_state(item) for item in heroes),
            enemies=tuple(_as_enemy_state(item) for item in enemies),
            actions=tuple(_as_action_state(item) for item in actions),
            active_actor_id=active_actor_id,
        )


def _as_hero_state(item: object) -> HeroState:
    to_state = getattr(item, "to_state", None)
    if callable(to_state):
        state = to_state()
        if isinstance(state, HeroState):
            return state
    if isinstance(item, HeroState):
        return item
    raise InvalidCombatStateError("heroes must be HeroObservation or HeroState instances.")


def _as_enemy_state(item: object) -> EnemyState:
    to_state = getattr(item, "to_state", None)
    if callable(to_state):
        state = to_state()
        if isinstance(state, EnemyState):
            return state
    if isinstance(item, EnemyState):
        return item
    raise InvalidCombatStateError("enemies must be EnemyObservation or EnemyState instances.")


def _as_action_state(item: object) -> ActionState:
    to_state = getattr(item, "to_state", None)
    if callable(to_state):
        state = to_state()
        if isinstance(state, ActionState):
            return state
    if isinstance(item, ActionState):
        return item
    raise InvalidCombatStateError("actions must be ActionObservation or ActionState instances.")


def _validate_unique_actor_ids(heroes: Sequence[HeroState], enemies: Sequence[EnemyState]) -> None:
    seen: dict[str, str] = {}
    for hero in heroes:
        if hero.actor_id in seen:
            raise InvalidCombatStateError(
                f"Duplicate actor_id {hero.actor_id!r} ({seen[hero.actor_id]} and hero). "
                "Each actor in a snapshot needs a unique instance id."
            )
        seen[hero.actor_id] = "hero"
    for enemy in enemies:
        if enemy.actor_id in seen:
            raise InvalidCombatStateError(
                f"Duplicate actor_id {enemy.actor_id!r} ({seen[enemy.actor_id]} and enemy). "
                "Each actor in a snapshot needs a unique instance id."
            )
        seen[enemy.actor_id] = "enemy"


def _validate_unique_hero_ranks(heroes: Sequence[HeroState]) -> None:
    claimed: dict[int, str] = {}
    for hero in heroes:
        if hero.rank.status is not ObservationStatus.OBSERVED or hero.rank.value is None:
            continue
        rank = hero.rank.value
        if rank in claimed:
            raise InvalidCombatStateError(
                f"Heroes {claimed[rank]!r} and {hero.actor_id!r} both occupy rank {rank}. "
                "Each hero rank 1-4 can hold at most one hero."
            )
        claimed[rank] = hero.actor_id


def _validate_enemy_rank_coverage(enemies: Sequence[EnemyState]) -> None:
    claimed: dict[int, str] = {}
    for enemy in enemies:
        if (
            enemy.occupied_ranks.status is not ObservationStatus.OBSERVED
            or enemy.occupied_ranks.value is None
        ):
            continue
        for rank in enemy.occupied_ranks.value:
            if rank in claimed:
                raise InvalidCombatStateError(
                    f"Enemies {claimed[rank]!r} and {enemy.actor_id!r} both occupy rank {rank}. "
                    "Enemy occupied_ranks must not overlap."
                )
            claimed[rank] = enemy.actor_id


def _validate_unique_action_slots(actions: Sequence[ActionState]) -> None:
    claimed: dict[ActionSlot, str] = {}
    for action in actions:
        if action.slot in claimed:
            raise InvalidCombatStateError(
                f"Duplicate action slot {action.slot.value}. "
                "The action bar has four skill slots plus move_action_slot."
            )
        claimed[action.slot] = action.slot.value


def _validate_active_actor(
    active_actor_id: ObservedValue[str],
    heroes: Sequence[HeroState],
    enemies: Sequence[EnemyState],
) -> None:
    if active_actor_id.status is not ObservationStatus.OBSERVED:
        return
    actor_id = active_actor_id.value
    known = {hero.actor_id for hero in heroes} | {enemy.actor_id for enemy in enemies}
    if actor_id not in known:
        raise InvalidCombatStateError(
            f"active_actor_id {actor_id!r} does not match any hero or enemy in this snapshot. "
            "Unknown turn ownership should use status 'unknown' instead of a dangling id."
        )
