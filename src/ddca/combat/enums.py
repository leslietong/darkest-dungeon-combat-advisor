"""Stable string enumerations for combat observations and actions."""

from __future__ import annotations

from enum import StrEnum


class ObservationStatus(StrEnum):
    """How a field was resolved. Unknown values stay explicit."""

    OBSERVED = "observed"
    UNKNOWN = "unknown"
    NOT_VISIBLE = "not_visible"
    NOT_APPLICABLE = "not_applicable"


class EvidenceSource(StrEnum):
    """Where an observed value came from. Phase 3A does not run these sources."""

    MANUAL = "manual"
    TEMPLATE_MATCHING = "template_matching"
    OCR = "ocr"
    CLASSIFIER = "classifier"
    GAME_DATA = "game_data"
    RULE_ENGINE = "rule_engine"
    UNKNOWN = "unknown"


class ActionKind(StrEnum):
    """Kind of a legal player combat action on the action bar."""

    SKILL = "skill"
    MOVE = "move"


class ActionSlot(StrEnum):
    """Action-bar slot IDs matching the Phase 2 calibration region names."""

    SKILL_1 = "skill_slot_1"
    SKILL_2 = "skill_slot_2"
    SKILL_3 = "skill_slot_3"
    SKILL_4 = "skill_slot_4"
    MOVE = "move_action_slot"


class ActorSide(StrEnum):
    """Which party an actor belongs to."""

    HERO = "hero"
    ENEMY = "enemy"


def action_kind_for_slot(slot: ActionSlot) -> ActionKind:
    """Return the action kind implied by a calibrated slot."""

    if slot is ActionSlot.MOVE:
        return ActionKind.MOVE
    return ActionKind.SKILL
