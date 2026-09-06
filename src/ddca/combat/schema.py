"""Combat schema version. Independent of the ``ddca`` package version."""

from __future__ import annotations

# Bump only when the CombatState JSON shape or invariants change.
COMBAT_SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (COMBAT_SCHEMA_VERSION,)
