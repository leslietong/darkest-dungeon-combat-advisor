"""Deterministic JSON serialization for CombatState."""

from __future__ import annotations

import json
from collections.abc import Mapping

from ddca.combat.errors import InvalidCombatStateError
from ddca.combat.state import CombatState


def combat_state_to_dict(state: CombatState) -> dict[str, object]:
    """Return a JSON-ready mapping."""

    return state.to_dict()


def combat_state_from_dict(payload: Mapping[str, object]) -> CombatState:
    """Rebuild a CombatState from a mapping."""

    return CombatState.from_dict(payload)


def combat_state_to_json(state: CombatState) -> str:
    """Serialize with sorted keys so identical states produce identical JSON."""

    return json.dumps(state.to_dict(), sort_keys=True, ensure_ascii=True, separators=(", ", ": "))


def combat_state_from_json(text: str) -> CombatState:
    """Deserialize JSON produced by ``combat_state_to_json``."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidCombatStateError("CombatState JSON is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise InvalidCombatStateError("CombatState JSON must be an object.")
    return CombatState.from_dict(payload)
