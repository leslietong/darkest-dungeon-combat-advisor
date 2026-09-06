"""JSON round-trip tests for CombatState. Fixtures are synthetic only."""

from __future__ import annotations

import ast
from pathlib import Path

from ddca.combat.serialize import combat_state_from_json, combat_state_to_json
from ddca.combat.state import CombatState

from tests.unit.combat_fixtures import make_combat_state, make_hero_observation

COMBAT_DIR = Path(__file__).resolve().parents[2] / "src" / "ddca" / "combat"
ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "collections",
    "dataclasses",
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


def test_tuples_survive_list_encoding_in_json() -> None:
    state = make_combat_state()
    restored = combat_state_from_json(combat_state_to_json(state))
    assert restored.enemies[0].occupied_ranks.value == (1, 2)
    assert restored.actions[0].legal_target_ranks.value == (1, 2)
    assert restored.heroes[0].status_effects.value[0].effect_id == "bleed"


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
