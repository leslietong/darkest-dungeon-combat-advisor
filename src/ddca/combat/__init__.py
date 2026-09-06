"""Structured combat observations and state. Standard library only."""

from __future__ import annotations

from ddca.combat.actions import ActionState
from ddca.combat.actors import EnemyState, HeroState
from ddca.combat.effects import StatusEffect
from ddca.combat.enums import (
    ActionKind,
    ActionSlot,
    ActorSide,
    EvidenceSource,
    ObservationStatus,
    action_kind_for_slot,
)
from ddca.combat.errors import CombatModelError, InvalidCombatStateError
from ddca.combat.evidence import FrameReference, RegionEvidence
from ddca.combat.observations import (
    ActionObservation,
    EnemyObservation,
    HeroObservation,
    Observation,
)
from ddca.combat.observed import (
    Confidence,
    ObservedValue,
    not_applicable,
    not_visible,
    observed,
    unknown,
)
from ddca.combat.schema import COMBAT_SCHEMA_VERSION
from ddca.combat.serialize import (
    combat_state_from_dict,
    combat_state_from_json,
    combat_state_to_dict,
    combat_state_to_json,
)
from ddca.combat.state import CombatState

__all__ = [
    "COMBAT_SCHEMA_VERSION",
    "ActionKind",
    "ActionObservation",
    "ActionSlot",
    "ActionState",
    "ActorSide",
    "CombatModelError",
    "CombatState",
    "Confidence",
    "EnemyObservation",
    "EnemyState",
    "EvidenceSource",
    "FrameReference",
    "HeroObservation",
    "HeroState",
    "InvalidCombatStateError",
    "Observation",
    "ObservationStatus",
    "ObservedValue",
    "RegionEvidence",
    "StatusEffect",
    "action_kind_for_slot",
    "combat_state_from_dict",
    "combat_state_from_json",
    "combat_state_to_dict",
    "combat_state_to_json",
    "not_applicable",
    "not_visible",
    "observed",
    "unknown",
]
