"""Invariant tests for hero, enemy, action, and combat snapshots."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ddca import __version__
from ddca.combat.actions import ActionState
from ddca.combat.enums import ActionKind, ActionSlot, EvidenceSource, ObservationStatus
from ddca.combat.errors import InvalidCombatStateError
from ddca.combat.observations import Observation
from ddca.combat.observed import observed, unknown
from ddca.combat.schema import COMBAT_SCHEMA_VERSION
from ddca.combat.state import CombatState
from ddca.vision.calibration import REQUIRED_REGIONS

from tests.unit.combat_fixtures import (
    default_actions,
    default_enemies,
    default_heroes,
    make_action_observation,
    make_combat_state,
    make_enemy,
    make_enemy_observation,
    make_hero,
    make_hero_observation,
    make_move,
    make_skill,
    ov,
    synthetic_frame,
)


def test_hero_rank_must_be_between_one_and_four() -> None:
    with pytest.raises(InvalidCombatStateError, match="outside 1-4"):
        make_hero(rank=5)


def test_hero_rank_rejects_bool() -> None:
    with pytest.raises(InvalidCombatStateError, match="rank"):
        make_hero().__class__(
            actor_id="hero_rank_1",
            rank=ov(True),  # type: ignore[arg-type]
            class_id=ov("highwayman"),
            current_hp=ov(10),
            maximum_hp=ov(10),
            current_stress=ov(0),
            maximum_stress=ov(200),
            status_effects=ov(()),
        )


def test_duplicate_hero_ranks_are_rejected() -> None:
    heroes = (make_hero(actor_id="a", rank=1), make_hero(actor_id="b", rank=1))
    with pytest.raises(InvalidCombatStateError, match="both occupy rank 1"):
        make_combat_state(heroes=heroes)


def test_duplicate_actor_ids_are_rejected() -> None:
    heroes = (make_hero(actor_id="shared", rank=1), make_hero(actor_id="hero_rank_2", rank=2))
    enemies = (make_enemy(actor_id="shared", occupied_ranks=(1,)),)
    with pytest.raises(InvalidCombatStateError, match="Duplicate actor_id"):
        make_combat_state(heroes=heroes, enemies=enemies)


def test_large_enemy_occupies_contiguous_ranks() -> None:
    enemy = make_enemy(occupied_ranks=(1, 2), enemy_type_id="unholy_giant")
    assert enemy.occupied_ranks.value == (1, 2)


def test_enemy_ranks_must_be_contiguous() -> None:
    with pytest.raises(InvalidCombatStateError, match="not contiguous"):
        make_enemy(occupied_ranks=(1, 3))


def test_enemy_occupied_ranks_cannot_be_empty() -> None:
    with pytest.raises(InvalidCombatStateError, match="at least one rank"):
        make_enemy(occupied_ranks=())


def test_enemy_rank_overlap_is_rejected() -> None:
    enemies = (
        make_enemy(actor_id="left", occupied_ranks=(1, 2)),
        make_enemy(actor_id="right", occupied_ranks=(2, 3)),
    )
    with pytest.raises(InvalidCombatStateError, match="both occupy rank 2"):
        make_combat_state(enemies=enemies, heroes=default_heroes())


def test_hp_current_cannot_exceed_maximum() -> None:
    with pytest.raises(InvalidCombatStateError, match="exceeds maximum"):
        make_hero(current_hp=24, maximum_hp=23)


def test_stress_cannot_be_negative() -> None:
    with pytest.raises(InvalidCombatStateError, match=">= 0"):
        make_hero(current_stress=-1)


def test_unknown_hp_does_not_store_a_guess() -> None:
    hero = make_hero()
    unknown_hp = replace(hero, current_hp=unknown())
    assert unknown_hp.current_hp.value is None
    make_combat_state(heroes=(unknown_hp, make_hero(actor_id="hero_rank_2", rank=2)))


def test_duplicate_action_slots_are_rejected() -> None:
    actions = (
        make_skill(ActionSlot.SKILL_1, "opened_vein"),
        make_skill(ActionSlot.SKILL_1, "pistol_shot"),
    )
    with pytest.raises(InvalidCombatStateError, match="Duplicate action slot"):
        make_combat_state(actions=actions)


def test_move_slot_cannot_use_skill_kind() -> None:
    with pytest.raises(InvalidCombatStateError, match="requires kind move"):
        ActionState(
            slot=ActionSlot.MOVE,
            kind=ActionKind.SKILL,
            skill_id=unknown(),
            is_available=ov(True),
            legal_target_ranks=unknown(),
        )


def test_move_cannot_observe_a_skill_id() -> None:
    with pytest.raises(InvalidCombatStateError, match="cannot observe a skill_id"):
        ActionState(
            slot=ActionSlot.MOVE,
            kind=ActionKind.MOVE,
            skill_id=ov("walk"),
            is_available=ov(True),
            legal_target_ranks=unknown(),
        )


def test_active_actor_must_exist_when_observed() -> None:
    with pytest.raises(InvalidCombatStateError, match="does not match"):
        make_combat_state(active_actor_id=ov("missing_actor"))


def test_unknown_active_actor_is_allowed() -> None:
    state = make_combat_state(active_actor_id=unknown())
    assert state.active_actor_id.status is ObservationStatus.UNKNOWN


def test_schema_version_is_independent_of_package_version() -> None:
    assert COMBAT_SCHEMA_VERSION == 1
    assert __version__ == "0.1.0"
    assert str(COMBAT_SCHEMA_VERSION) != __version__
    with pytest.raises(InvalidCombatStateError, match="Unsupported combat schema_version"):
        make_combat_state(schema_version=99)


def test_action_slot_names_match_phase_2_calibration_keys() -> None:
    for slot in ActionSlot:
        assert slot.value in REQUIRED_REGIONS


def test_per_field_confidence_is_not_a_single_global_score() -> None:
    hero = make_hero(hp_score=0.95, class_score=0.4)
    assert hero.current_hp.confidence.score != hero.class_id.confidence.score
    state = make_combat_state()
    scores = {
        state.round_number.confidence.score,
        state.active_actor_id.confidence.score,
        state.heroes[0].current_hp.confidence.score,
        state.enemies[0].enemy_type_id.confidence.score,
        state.actions[0].skill_id.confidence.score,
    }
    assert len(scores) > 1


def test_observations_convert_into_combat_state() -> None:
    hero_obs = make_hero_observation()
    enemy_obs = make_enemy_observation()
    action_obs = make_action_observation()
    state = CombatState.from_observations(
        frame=synthetic_frame(),
        round_number=ov(1),
        heroes=(hero_obs,),
        enemies=(enemy_obs,),
        actions=(action_obs, make_move()),
        active_actor_id=ov(hero_obs.actor_id),
    )
    assert state.heroes[0] == hero_obs.to_state()
    assert state.enemies[0].occupied_ranks.value == (1, 2)


def test_generic_observation_envelope_does_not_hold_pixels() -> None:
    envelope = Observation(frame=synthetic_frame(), payload=make_hero(), evidence=())
    encoded = envelope.to_dict()
    assert "payload" in encoded
    assert b"raw" not in str(encoded).encode("utf-8")


def test_stable_string_ids_allow_unknown_dlc_classes() -> None:
    hero = make_hero(class_id="flagellant")
    enemy = make_enemy(enemy_type_id="farmstead_harvest_child")
    assert hero.class_id.value == "flagellant"
    assert enemy.enemy_type_id.value == "farmstead_harvest_child"


def test_default_fixture_is_valid() -> None:
    state = make_combat_state()
    assert len(state.heroes) == 4
    assert state.enemies[0].occupied_ranks.value == (1, 2)
    assert [action.slot for action in state.actions] == list(ActionSlot)
    assert default_actions()[-1].kind is ActionKind.MOVE
    assert default_heroes()[0].actor_id == "hero_rank_4"
    assert default_enemies()[0].actor_id == "enemy_large"
    assert make_hero_observation().evidence[0].region_name == "hero_rank_1"
    assert EvidenceSource.MANUAL.value == "manual"
