"""Errors for structured combat observations and state."""

from __future__ import annotations


class CombatModelError(Exception):
    """Base error for combat observation and state models."""


class InvalidCombatStateError(CombatModelError):
    """A combat model invariant was violated."""
