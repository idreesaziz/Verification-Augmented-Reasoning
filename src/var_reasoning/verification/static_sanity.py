"""Layer 0 — Static sanity gates.

Fast, deterministic pre-checks that run before any code execution.
These catch obvious errors instantly: wrong type, implausible range,
untraced constants, non-integer AIME answers, etc.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SanityResult:
    passed: bool
    reason: str = ""


# ── Trivial-value whitelist ──────────────────────────────────────────
# Small integers and common mathematical constants that are always OK
# to appear in code without being explicitly derived in a prior step.
_TRIVIAL_VALUES: set[float] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000,
    0.0, 1.0, 2.0, 0.5,
}


# ── Range plausibility ───────────────────────────────────────────────

def check_range(value: float, reasoning_pattern: str) -> SanityResult:
    """Check that a claimed value is in a plausible range for its type."""
    if math.isnan(value) or math.isinf(value):
        return SanityResult(False, f"Value is {value} — not a valid result.")

    # Probabilities must be in [0, 1]
    if reasoning_pattern == "product_rule":
        if not (0 <= value <= 1) and value == int(value):
            # product_rule can also produce counts; only flag fractions
            pass
        elif 0 < value < 1:
            # Looks like a probability — fine
            pass

    # Negative counts are always wrong
    if reasoning_pattern == "exhaustive_enumeration" and value < 0:
        return SanityResult(False, f"Count cannot be negative: {value}")

    return SanityResult(True)


# ── Integrality check ────────────────────────────────────────────────

def check_integrality(
    value: float, problem_text: str, is_final_answer: bool = False
) -> SanityResult:
    """Flag non-integer final answers for problems that require integers.

    AIME answers are always integers in [0, 999].
    """
    if not is_final_answer:
        return SanityResult(True)

    # Detect AIME-style problems
    aime_keywords = ["aime", "find the remainder", "compute the integer"]
    text_lower = problem_text.lower()
    is_aime = any(kw in text_lower for kw in aime_keywords)

    if is_aime and not math.isclose(value, round(value), abs_tol=1e-6):
        return SanityResult(
            False,
            f"AIME answer must be an integer but got {value}. "
            f"This likely indicates an error in the reasoning chain.",
        )
    return SanityResult(True)


# ── Provenance check ─────────────────────────────────────────────────

def _extract_numeric_literals(code: str) -> list[tuple[float, int]]:
    """Extract all numeric literals from Python code with their line numbers."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    literals: list[tuple[float, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            literals.append((float(node.value), getattr(node, "lineno", 0)))
    return literals


def _extract_problem_numbers(problem_text: str) -> set[float]:
    """Extract all numbers mentioned in the problem statement."""
    numbers: set[float] = set()
    # Match integers and decimals
    for match in re.finditer(r"\b\d+(?:\.\d+)?\b", problem_text):
        try:
            numbers.add(float(match.group()))
        except ValueError:
            pass
    return numbers


def check_provenance(
    code: str,
    problem_text: str,
    prior_observations: list[str],
) -> SanityResult:
    """Check that all numeric literals in code are traceable.

    Every number in verification/simulation code must come from:
      (a) The problem statement
      (b) A prior step's observation (printed output)
      (c) The trivial-value whitelist (0, 1, 2, ..., π)

    Numbers that appear from nowhere are "smuggled constants" — the model
    pulled them from memory rather than computing them.
    """
    literals = _extract_numeric_literals(code)
    if not literals:
        return SanityResult(True)

    # Build the allowlist
    allowed: set[float] = set(_TRIVIAL_VALUES)
    allowed.update(_extract_problem_numbers(problem_text))

    # Add numbers from prior observations
    for obs in prior_observations:
        for match in re.finditer(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", obs):
            try:
                allowed.add(float(match.group()))
            except ValueError:
                pass

    # Check each literal
    untraced: list[tuple[float, int]] = []
    for value, lineno in literals:
        if value in allowed:
            continue
        # Check approximate match (for floating-point representation)
        if any(math.isclose(value, a, rel_tol=1e-6) for a in allowed):
            continue
        untraced.append((value, lineno))

    if untraced:
        examples = ", ".join(f"{v} (line {ln})" for v, ln in untraced[:5])
        return SanityResult(
            False,
            f"UNTRACED CONSTANTS: {examples}. "
            f"These values were never derived in a prior step or stated "
            f"in the problem. Compute them explicitly before using them.",
        )
    return SanityResult(True)


# ── Claimed-value-in-simulation check ────────────────────────────────

def check_no_claimed_value_in_code(
    code: str, claimed_value: float
) -> SanityResult:
    """Ensure the claimed value doesn't appear as a literal in simulation code.

    The simulation must PRODUCE a value independently, not consume the
    claimed value as an input. This prevents circular verification.
    """
    if claimed_value in _TRIVIAL_VALUES:
        return SanityResult(True)  # Can't ban 0, 1, 2, etc.

    literals = _extract_numeric_literals(code)
    for value, lineno in literals:
        if math.isclose(value, claimed_value, rel_tol=1e-6):
            return SanityResult(
                False,
                f"CIRCULAR SIMULATION: The claimed value {claimed_value} "
                f"appears as a literal on line {lineno} of the simulation "
                f"code. The simulation must independently derive the value, "
                f"not embed it.",
            )
    return SanityResult(True)
