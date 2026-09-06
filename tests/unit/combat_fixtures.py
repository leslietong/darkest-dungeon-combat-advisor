"""Synthetic combat fixtures. These never open the game or read screenshots."""

from __future__ import annotations

from ddca.combat.actions import ActionState
from ddca.combat.actors import EnemyState, HeroState
from ddca.combat.effects import StatusEffect
from ddca.combat.enums import ActionKind, ActionSlot, EvidenceSource
from ddca.combat.evidence import FrameReference, RegionEvidence
from ddca.combat.observations import ActionObservation, EnemyObservation, HeroObservation
from ddca.combat.observed import ObservedValue, not_applicable, observed, unknown
from ddca.combat.schema import COMBAT_SCHEMA_VERSION
from ddca.combat.state import CombatState


def synthetic_frame() -> FrameReference:
    return FrameReference(
        capture_id="synthetic_capture_001",
        captured_at="2026-08-27T00:00:00Z",
        image_path=None,
        image_width=1111,
        image_height=654,
    )


def ov(value: object, *, score: float = 1.0, source: EvidenceSource = EvidenceSource.MANUAL) -> ObservedValue:
    return observed(value, score=score, source=source)


def bleed(*, stacks: int = 3, duration: int = 3, score: float = 0.8) -> StatusEffect:
    return StatusEffect(
        effect_id="bleed",
        stacks=ov(stacks, score=score),
        duration_rounds=ov(duration, score=score),
    )


def make_hero(
    *,
    actor_id: str = "hero_rank_1",
    rank: int = 1,
    class_id: str = "highwayman",
    current_hp: int = 23,
    maximum_hp: int = 23,
    current_stress: int = 12,
    maximum_stress: int = 200,
    effects: tuple[StatusEffect, ...] | None = None,
    hp_score: float = 0.95,
    class_score: float = 0.7,
) -> HeroState:
    return HeroState(
        actor_id=actor_id,
        rank=ov(rank, score=1.0),
        class_id=ov(class_id, score=class_score),
        current_hp=ov(current_hp, score=hp_score),
        maximum_hp=ov(maximum_hp, score=hp_score),
        current_stress=ov(current_stress, score=0.6),
        maximum_stress=ov(maximum_stress, score=1.0),
        status_effects=ov(effects if effects is not None else (), score=0.5),
    )


def make_enemy(
    *,
    actor_id: str = "enemy_front",
    occupied_ranks: tuple[int, ...] = (1,),
    enemy_type_id: str = "bone_soldier",
    current_hp: int = 8,
    maximum_hp: int = 10,
    effects: tuple[StatusEffect, ...] | None = None,
) -> EnemyState:
    return EnemyState(
        actor_id=actor_id,
        occupied_ranks=ov(occupied_ranks, score=0.9),
        enemy_type_id=ov(enemy_type_id, score=0.65),
        current_hp=ov(current_hp, score=0.85),
        maximum_hp=ov(maximum_hp, score=0.85),
        status_effects=ov(effects if effects is not None else (), score=0.4),
    )


def make_skill(
    slot: ActionSlot,
    skill_id: str,
    *,
    available: bool = True,
    targets: tuple[int, ...] = (1, 2),
) -> ActionState:
    return ActionState(
        slot=slot,
        kind=ActionKind.SKILL,
        skill_id=ov(skill_id, score=0.75),
        is_available=ov(available, score=0.9),
        legal_target_ranks=ov(targets, score=0.55),
    )


def make_move(*, available: bool = True) -> ActionState:
    return ActionState(
        slot=ActionSlot.MOVE,
        kind=ActionKind.MOVE,
        skill_id=not_applicable(),
        is_available=ov(available, score=0.9),
        legal_target_ranks=unknown(score=0.0),
    )


def default_actions() -> tuple[ActionState, ...]:
    return (
        make_skill(ActionSlot.SKILL_1, "opened_vein"),
        make_skill(ActionSlot.SKILL_2, "pistol_shot", targets=(2, 3, 4)),
        make_skill(ActionSlot.SKILL_3, "grapeshot_blast", available=False, targets=(1, 2, 3)),
        make_skill(ActionSlot.SKILL_4, "duelists_advance", targets=(1,)),
        make_move(),
    )


def default_heroes() -> tuple[HeroState, ...]:
    return (
        make_hero(actor_id="hero_rank_4", rank=4, class_id="vestal", current_hp=18, maximum_hp=24, current_stress=40),
        make_hero(actor_id="hero_rank_3", rank=3, class_id="plague_doctor", current_hp=15, maximum_hp=22, current_stress=25),
        make_hero(actor_id="hero_rank_2", rank=2, class_id="jester", current_hp=19, maximum_hp=19, current_stress=55),
        make_hero(
            actor_id="hero_rank_1",
            rank=1,
            class_id="highwayman",
            current_hp=20,
            maximum_hp=23,
            current_stress=12,
            effects=(bleed(),),
        ),
    )


def default_enemies() -> tuple[EnemyState, ...]:
    return (
        make_enemy(
            actor_id="enemy_large",
            occupied_ranks=(1, 2),
            enemy_type_id="unholy_giant",
            current_hp=42,
            maximum_hp=55,
        ),
        make_enemy(
            actor_id="enemy_rank_3",
            occupied_ranks=(3,),
            enemy_type_id="bone_courtier",
            current_hp=6,
            maximum_hp=8,
        ),
        make_enemy(
            actor_id="enemy_rank_4",
            occupied_ranks=(4,),
            enemy_type_id="cultist_brawler",
            current_hp=12,
            maximum_hp=12,
        ),
    )


def make_combat_state(**overrides: object) -> CombatState:
    payload = {
        "schema_version": COMBAT_SCHEMA_VERSION,
        "frame": synthetic_frame(),
        "round_number": ov(3, score=0.8),
        "heroes": default_heroes(),
        "enemies": default_enemies(),
        "actions": default_actions(),
        "active_actor_id": ov("hero_rank_1", score=0.92),
    }
    payload.update(overrides)
    return CombatState(**payload)  # type: ignore[arg-type]


def make_hero_observation(**overrides: object) -> HeroObservation:
    hero = make_hero()
    payload = {
        "frame": synthetic_frame(),
        "actor_id": hero.actor_id,
        "rank": hero.rank,
        "class_id": hero.class_id,
        "current_hp": hero.current_hp,
        "maximum_hp": hero.maximum_hp,
        "current_stress": hero.current_stress,
        "maximum_stress": hero.maximum_stress,
        "status_effects": hero.status_effects,
        "evidence": (
            RegionEvidence(region_name="hero_rank_1", x=413, y=202, width=120, height=259),
        ),
    }
    payload.update(overrides)
    return HeroObservation(**payload)  # type: ignore[arg-type]


def make_enemy_observation(**overrides: object) -> EnemyObservation:
    enemy = make_enemy(occupied_ranks=(1, 2), enemy_type_id="unholy_giant")
    payload = {
        "frame": synthetic_frame(),
        "actor_id": enemy.actor_id,
        "occupied_ranks": enemy.occupied_ranks,
        "enemy_type_id": enemy.enemy_type_id,
        "current_hp": enemy.current_hp,
        "maximum_hp": enemy.maximum_hp,
        "status_effects": enemy.status_effects,
        "evidence": (
            RegionEvidence(region_name="enemy_rank_1", x=577, y=202, width=120, height=259),
        ),
    }
    payload.update(overrides)
    return EnemyObservation(**payload)  # type: ignore[arg-type]


def make_action_observation(**overrides: object) -> ActionObservation:
    action = make_skill(ActionSlot.SKILL_1, "opened_vein")
    payload = {
        "frame": synthetic_frame(),
        "slot": action.slot,
        "kind": action.kind,
        "skill_id": action.skill_id,
        "is_available": action.is_available,
        "legal_target_ranks": action.legal_target_ranks,
        "evidence": (
            RegionEvidence(region_name="skill_slot_1", x=304, y=461, width=42, height=42),
        ),
    }
    payload.update(overrides)
    return ActionObservation(**payload)  # type: ignore[arg-type]
