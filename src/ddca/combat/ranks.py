"""Hero and enemy rank helpers. Ranks are 1 (front) through 4 (back)."""

from __future__ import annotations

from collections.abc import Sequence

from ddca.combat.errors import InvalidCombatStateError

MIN_RANK = 1
MAX_RANK = 4
VALID_RANKS: tuple[int, ...] = (1, 2, 3, 4)


def validate_rank(rank: int, *, context: str) -> int:
    """Return `rank` if it is an integer in 1..4."""

    if isinstance(rank, bool) or not isinstance(rank, int):
        raise InvalidCombatStateError(
            f"{context}: rank must be an integer from {MIN_RANK} to {MAX_RANK} "
            f"(got {rank!r}). Use 1 for the front rank and 4 for the back rank."
        )
    if rank < MIN_RANK or rank > MAX_RANK:
        raise InvalidCombatStateError(
            f"{context}: rank {rank} is outside {MIN_RANK}-{MAX_RANK}. "
            "Darkest Dungeon combat ranks are 1 (front) through 4 (back)."
        )
    return rank


def validate_occupied_ranks(ranks: Sequence[int], *, context: str) -> tuple[int, ...]:
    """Return sorted unique contiguous ranks in 1..4."""

    validated = [validate_rank(rank, context=context) for rank in ranks]
    if not validated:
        raise InvalidCombatStateError(
            f"{context}: occupied_ranks must contain at least one rank. "
            "Assign the enemy to the rank slots it covers."
        )
    unique = tuple(sorted(set(validated)))
    if len(unique) != len(validated):
        raise InvalidCombatStateError(
            f"{context}: occupied_ranks contains duplicates {tuple(validated)}. "
            "List each covered rank once."
        )
    expected = tuple(range(unique[0], unique[-1] + 1))
    if unique != expected:
        raise InvalidCombatStateError(
            f"{context}: occupied_ranks {unique} are not contiguous. "
            "A large enemy must occupy a consecutive block such as (1, 2)."
        )
    return unique
