"""Internal state models for the reasoning session."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from var_reasoning.models.schemas import (
    ReasoningPattern,
    VerificationTarget,
    VerificationType,
)


def build_inference_string(premises: list[str], conclusion: str) -> str:
    """Derive a human-readable inference string from structured premises."""
    if premises:
        return "Given: " + " | ".join(premises) + " — I conclude: " + conclusion
    return conclusion


class VerificationResult(BaseModel):
    passed: bool
    verification_type: VerificationType
    error_message: Optional[str] = None
    counterexample: Optional[str] = None


class CompletedStep(BaseModel):
    step_number: int
    objective: str  # The ONE question this step answered
    depends_on: list[str]  # result_variable names from prior steps
    thought: str
    action: str  # Python code
    observation: str  # stdout from execution
    result_variable: str  # The variable this step produced
    premises: list[str]
    conclusion: str
    reasoning_pattern: ReasoningPattern
    inference: str  # derived: build_inference_string(premises, conclusion)
    verification_target: VerificationTarget
    verification_result: VerificationResult


class Session(BaseModel):
    problem_id: str
    problem_text: str
    steps: list[CompletedStep] = Field(default_factory=list)
    final_answer: Optional[str] = None
    justification: Optional[str] = None
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_backtracks: int = 0
    total_code_retries: int = 0
    total_inference_retries: int = 0
    informal_skip_count: int = 0
    informal_without_reason_count: int = 0
    verification_type_counts: dict[str, int] = Field(default_factory=dict)
    reasoning_pattern_counts: dict[str, int] = Field(default_factory=dict)
