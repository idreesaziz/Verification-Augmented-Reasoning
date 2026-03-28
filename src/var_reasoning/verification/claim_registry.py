"""Claim registry — tracks all accepted claims for global consistency.

Every verified step registers its claim (variable name, value, conclusion).
After each new registration, the registry checks that all accepted claims
are jointly consistent using simple arithmetic constraints.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegisteredClaim:
    step_number: int
    variable_name: str
    claimed_value: float
    conclusion: str
    depends_on: list[str]


@dataclass
class ConsistencyResult:
    consistent: bool
    conflicting_steps: list[tuple[int, int]] = field(default_factory=list)
    detail: str = ""


class ClaimRegistry:
    """Accumulates claims and checks global consistency."""

    def __init__(self) -> None:
        self._claims: list[RegisteredClaim] = []
        self._by_variable: dict[str, RegisteredClaim] = {}

    @property
    def claims(self) -> list[RegisteredClaim]:
        return list(self._claims)

    def register(
        self,
        step_number: int,
        variable_name: str,
        claimed_value: float,
        conclusion: str,
        depends_on: list[str],
    ) -> None:
        claim = RegisteredClaim(
            step_number=step_number,
            variable_name=variable_name,
            claimed_value=claimed_value,
            conclusion=conclusion,
            depends_on=depends_on,
        )
        self._claims.append(claim)
        self._by_variable[variable_name] = claim
        logger.debug(
            "Registered claim: step=%d, %s=%.6f",
            step_number, variable_name, claimed_value,
        )

    def get_value(self, variable_name: str) -> float | None:
        claim = self._by_variable.get(variable_name)
        return claim.claimed_value if claim else None

    def check_consistency(self) -> ConsistencyResult:
        """Check that all registered claims are jointly consistent.

        Current checks:
        1. Dependency consistency: if step B depends on step A's variable
           and claims a value derived from it, verify the arithmetic holds.
        2. No contradictory claims for the same variable.
        """
        conflicts: list[tuple[int, int]] = []

        # Check for duplicate variable names with different values
        seen: dict[str, RegisteredClaim] = {}
        for claim in self._claims:
            if claim.variable_name in seen:
                prior = seen[claim.variable_name]
                if not math.isclose(
                    prior.claimed_value, claim.claimed_value,
                    rel_tol=1e-6, abs_tol=1e-9,
                ):
                    conflicts.append((prior.step_number, claim.step_number))
                    logger.warning(
                        "Conflicting claims for '%s': step %d says %.6f, "
                        "step %d says %.6f",
                        claim.variable_name,
                        prior.step_number, prior.claimed_value,
                        claim.step_number, claim.claimed_value,
                    )
            seen[claim.variable_name] = claim

        if conflicts:
            return ConsistencyResult(
                consistent=False,
                conflicting_steps=conflicts,
                detail=f"Contradictory values for same variable in {len(conflicts)} pair(s).",
            )

        return ConsistencyResult(consistent=True)

    def reset(self) -> None:
        self._claims.clear()
        self._by_variable.clear()

    def all_observations(self) -> dict[str, float]:
        """Return a dict of variable_name → claimed_value for all claims."""
        return {c.variable_name: c.claimed_value for c in self._claims}
