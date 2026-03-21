"""Internal state models for the reasoning session."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from var_reasoning.models.schemas import VerificationTarget, VerificationType


class VerificationResult(BaseModel):
    passed: bool
    verification_type: VerificationType
    error_message: Optional[str] = None
    counterexample: Optional[str] = None


class CompletedStep(BaseModel):
    step_number: int
    thought: str
    action: str  # Python code
    observation: str  # stdout from execution
    inference: str
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
    verification_type_counts: dict[str, int] = Field(default_factory=dict)
