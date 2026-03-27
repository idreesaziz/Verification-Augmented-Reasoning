"""Pydantic schemas for LLM structured output."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel


class VerificationType(str, Enum):
    Z3 = "z3"
    SYMPY = "sympy"
    PYTHON_ASSERT = "python_assert"
    INFORMAL = "informal"


class ReasoningPattern(str, Enum):
    EXHAUSTIVE_ENUMERATION = "exhaustive_enumeration"
    PRODUCT_RULE = "product_rule"
    UNIVERSAL_CLAIM = "universal_claim"
    EXISTENTIAL = "existential"
    ALGEBRAIC = "algebraic"
    CASE_ANALYSIS = "case_analysis"


class VerificationTarget(BaseModel):
    type: VerificationType
    statement: str  # The actual Z3/SymPy/assert code to execute
    # Required when type == informal; soft-enforced (missing increments counter)
    informal_reason: Optional[str] = None


class ReasoningStep(BaseModel):
    objective: str  # The ONE question this step answers
    depends_on: list[str]  # result_variable names from prior steps this reads
    thought: str  # Why this is the right next thing to do
    action: str  # Python code that computes the answer
    result_variable: str  # The ONE variable assigned and printed by the action


class InferenceStep(BaseModel):
    premises: list[str]  # Discrete factual claims from the observation
    conclusion: str  # The single claim being drawn from the premises
    reasoning_pattern: ReasoningPattern  # Strict: enum enforced by schema mode
    verification_target: VerificationTarget


class FinalAnswer(BaseModel):
    answer: str
    justification: str  # Must reference specific step numbers


class StepOutput(BaseModel):
    """LLM returns exactly one of these three."""

    reasoning: Optional[ReasoningStep] = None
    final_answer: Optional[FinalAnswer] = None


class CodeFix(BaseModel):
    fixed_code: str
    explanation: str


class InferenceRevision(BaseModel):
    choice: Literal["revise", "investigate"]
    # For choice == "revise": provide all three structured fields
    revised_premises: Optional[list[str]] = None
    revised_conclusion: Optional[str] = None
    revised_reasoning_pattern: Optional[ReasoningPattern] = None
    revised_verification_target: Optional[VerificationTarget] = None
    # For choice == "investigate": provide thought + action
    thought: Optional[str] = None
    action: Optional[str] = None
