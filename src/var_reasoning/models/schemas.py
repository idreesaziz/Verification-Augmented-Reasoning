"""Pydantic schemas for LLM structured output.

VAR v2 architecture — premise provenance + adversarial falsification.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel


# ── Enums ────────────────────────────────────────────────────────────


class FactType(str, Enum):
    """How a fact entered the reasoning chain."""

    GIVEN = "given"  # Verbatim from problem statement
    COMPUTED = "computed"  # Printed by executed code
    DERIVED = "derived"  # Logical deduction — must be stress-tested


class ClaimType(str, Enum):
    """What kind of claim a derivation makes."""

    UNIVERSAL = "universal"  # "for all X, P holds"
    EXISTENTIAL = "existential"  # "there exists X such that..."
    EXPECTED_VALUE = "expected_value"  # "E[X] = v"
    PROBABILITY = "probability"  # "P(event) = p"
    DETERMINISTIC_COUNT = "deterministic_count"  # "the number of X is N"
    IDENTITY = "identity"  # "expression A = expression B"


class AttackTool(str, Enum):
    """Which tool the adversary uses to falsify a claim."""

    Z3 = "z3"  # Constraint solver — find counterexample
    MONTE_CARLO = "monte_carlo"  # Simulation — estimate empirically
    BRUTE_FORCE = "brute_force"  # Enumerate all cases


class Verdict(str, Enum):
    """Outcome of adversarial falsification."""

    REJECT = "reject"  # Strong evidence claim is wrong
    SURVIVE = "survive"  # No evidence against — claim stands
    INCONCLUSIVE = "inconclusive"  # Couldn't run or interpret


# ── Reasoner schemas ─────────────────────────────────────────────────


class Derivation(BaseModel):
    """A single logical claim made by the reasoner."""

    premise: str  # Formal: "P(two cross-quadrant chords intersect) = 1"
    justification: str  # WHY: "Since both chords must cross the center region..."
    depends_on: list[str]  # Fact IDs this derives from: ["given_1", "computed_3"]
    claimed_value: Optional[float] = None  # Numeric value if applicable


class ReasoningStep(BaseModel):
    """What the reasoner produces each step."""

    objective: str  # The ONE question this step answers
    facts_used: list[str]  # Fact IDs from the pool this step reads
    thought: str  # Planning — not graded
    action: str  # Python code to execute
    result_variable: str  # Variable printed by the code
    derivations: list[Derivation] = []  # Logical claims (can be empty for pure computation)


class FinalAnswer(BaseModel):
    """End of reasoning."""

    answer: str
    fact_chain: list[str]  # Ordered fact IDs tracing from givens to answer


class StepOutput(BaseModel):
    """LLM returns exactly one of these."""

    reasoning: Optional[ReasoningStep] = None
    final_answer: Optional[FinalAnswer] = None


class CodeFix(BaseModel):
    """Code repair output."""

    fixed_code: str
    explanation: str


# ── Adversary schemas ────────────────────────────────────────────────


class FalsificationAttempt(BaseModel):
    """What the adversarial verifier produces."""

    claim_type: ClaimType
    hidden_assumptions: list[str]  # What unstated assumptions does this rest on?
    attack_tool: AttackTool
    attack_rationale: str  # One sentence: why this tool, what it targets
    code: str  # Executable falsification code


class StepRevision(BaseModel):
    """What the reasoner produces when a derivation is rejected."""

    choice: Literal["revise", "investigate"]
    # For revise: new derivations for the same step
    revised_derivations: Optional[list[Derivation]] = None
    # For investigate: run new code
    thought: Optional[str] = None
    action: Optional[str] = None
