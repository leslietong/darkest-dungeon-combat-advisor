"""Helpers for non-negative current/maximum resources such as HP and stress."""

from __future__ import annotations

from ddca.combat.enums import ObservationStatus
from ddca.combat.errors import InvalidCombatStateError
from ddca.combat.observed import ObservedValue


def require_int(value: object, *, context: str) -> int:
    """Return a real integer. ``bool`` is rejected because it subclasses ``int``."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidCombatStateError(
            f"{context} must be an integer (got {value!r})."
        )
    return value


def require_non_empty_str(value: object, *, context: str) -> str:
    """Return a non-empty string without silently rewriting catalog IDs."""

    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise InvalidCombatStateError(
            f"{context} must be a non-empty string without leading or trailing whitespace "
            f"(got {value!r}). Use stable catalog IDs so DLC and mods can extend the set."
        )
    return value


def require_actor_id(value: object, *, context: str) -> str:
    """Return a trimmed instance id used only within one combat snapshot."""

    if not isinstance(value, str) or not value.strip():
        raise InvalidCombatStateError(
            f"{context} must be a non-empty actor id (got {value!r})."
        )
    return value.strip()


def validate_observed_int(field: ObservedValue[int], *, context: str) -> None:
    """When observed, the value must be an integer."""

    if field.status is ObservationStatus.OBSERVED:
        require_int(field.value, context=context)


def validate_non_negative_resource(field: ObservedValue[int], *, context: str) -> None:
    """When observed, HP/stress values must be integers >= 0."""

    if field.status is not ObservationStatus.OBSERVED:
        return
    amount = require_int(field.value, context=context)
    if amount < 0:
        raise InvalidCombatStateError(
            f"{context} must be >= 0 (got {amount}). Negative HP or stress is not valid."
        )


def validate_current_maximum(
    current: ObservedValue[int],
    maximum: ObservedValue[int],
    *,
    resource: str,
    context: str,
) -> None:
    """Validate a current/maximum pair independently, then current <= maximum."""

    validate_non_negative_resource(current, context=f"{context} {resource} current")
    validate_non_negative_resource(maximum, context=f"{context} {resource} maximum")
    if (
        current.status is ObservationStatus.OBSERVED
        and maximum.status is ObservationStatus.OBSERVED
        and current.value is not None
        and maximum.value is not None
        and current.value > maximum.value
    ):
        raise InvalidCombatStateError(
            f"{context}: {resource} current {current.value} exceeds maximum {maximum.value}."
        )
