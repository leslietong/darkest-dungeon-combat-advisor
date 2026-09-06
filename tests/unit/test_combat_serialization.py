"""JSON round-trip tests for CombatState. Fixtures are synthetic only."""

from __future__ import annotations

import ast
from datetime import timezone
from pathlib import Path

from ddca.combat.effects import StatusEffect
from ddca.combat.enums import ActionKind, ActionSlot
from ddca.combat.observed import Confidence
from ddca.combat.serialize import combat_state_from_json, combat_state_to_json
from ddca.combat.state import CombatState

from tests.unit.combat_fixtures import (
    SYNTHETIC_CAPTURED_AT,
    action_by_slot,
    enemy_by_id,
    hero_by_id,
    make_combat_state,
    make_hero_observation,
)

COMBAT_DIR = Path(__file__).resolve().parents[2] / "src" / "ddca" / "combat"
ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "collections",
    "dataclasses",
    "datetime",
    "enum",
    "json",
    "math",
    "typing",
    "ddca",
}


def test_combat_state_json_round_trip_preserves_equality() -> None:
    original = make_combat_state()
    restored = combat_state_from_json(combat_state_to_json(original))
    assert restored == original
    assert CombatState.from_dict(original.to_dict()) == original


def test_json_encoding_is_deterministic() -> None:
    state = make_combat_state()
    first = combat_state_to_json(state)
    second = combat_state_to_json(state)
    assert first == second
    assert first == combat_state_to_json(combat_state_from_json(first))


def test_nested_values_survive_json_round_trip_exactly() -> None:
    original = make_combat_state()
    restored = combat_state_from_json(combat_state_to_json(original))
    assert restored == original

    highwayman = hero_by_id(restored.heroes, "hero_rank_1")
    giant = enemy_by_id(restored.enemies, "enemy_large")
    opened_vein = action_by_slot(restored.actions, ActionSlot.SKILL_1)
    move = action_by_slot(restored.actions, ActionSlot.MOVE)
    effects = highwayman.status_effects.value

    assert restored.frame.captured_at == SYNTHETIC_CAPTURED_AT
    assert restored.frame.captured_at is not None
    assert restored.frame.captured_at.tzinfo is timezone.utc
    assert giant.occupied_ranks.value == (1, 2)
    assert isinstance(giant.occupied_ranks.value, tuple)
    assert opened_vein.legal_target_ranks.value == (1, 2)
    assert isinstance(opened_vein.legal_target_ranks.value, tuple)
    assert opened_vein.slot is ActionSlot.SKILL_1
    assert opened_vein.kind is ActionKind.SKILL
    assert move.kind is ActionKind.MOVE
    assert isinstance(highwayman.current_hp.confidence, Confidence)
    assert effects is not None
    bleed = next(effect for effect in effects if effect.effect_id == "bleed")
    assert isinstance(bleed, StatusEffect)
    assert bleed.stacks.value == 3


def test_hero_observation_json_includes_frame_not_pixels() -> None:
    encoded = make_hero_observation().to_dict()
    assert encoded["frame"]["capture_id"] == "synthetic_capture_001"
    assert encoded["frame"]["image_path"] is None
    assert "ndarray" not in str(encoded)
    assert "pixels" not in str(encoded)


def test_combat_package_uses_only_stdlib_and_ddca() -> None:
    forbidden = {
        "cv2",
        "numpy",
        "pydantic",
        "mss",
        "torch",
        "PIL",
        "sklearn",
        "win32gui",
        "yaml",
    }
    for path in COMBAT_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module.split(".")[0])
            for root in names:
                assert root not in forbidden, f"{path.name} imports {root}"
                assert root in ALLOWED_IMPORT_ROOTS, f"{path.name} imports {root}"
