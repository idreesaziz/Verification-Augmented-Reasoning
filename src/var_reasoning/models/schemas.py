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


class VerificationTarget(BaseModel):
    type: VerificationType
    statement: str  # The actual Z3/SymPy/assert code to execute
    premises: Optional[list[str]] = None  # References to prior steps


class ReasoningStep(BaseModel):
    thought: str  # What to investigate and why
    action: str  # Python code to execute


class InferenceStep(BaseModel):
    inference: str  # Conclusion drawn from observation
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
    revised_inference: Optional[str] = None
    revised_verification_target: Optional[VerificationTarget] = None
    thought: Optional[str] = None
    action: Optional[str] = None
