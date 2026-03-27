"""Feedback builders for the engine's retry loops."""

from __future__ import annotations

from var_reasoning.models.state import CompletedStep, Session
from var_reasoning.prompts.code_repair_prompt import build_code_repair_prompt
from var_reasoning.prompts.inference_retry_prompt import build_inference_retry_prompt


def build_conversation_history(session: Session) -> list[str]:
    """Build the conversation history for the reasoning LLM call."""
    parts: list[str] = [f"Problem:\n{session.problem_text}"]
    for step in session.steps:
        premises_str = "; ".join(step.premises) if step.premises else "(none)"
        depends_str = ", ".join(step.depends_on) if step.depends_on else "(none)"
        parts.append(
            f"Step {step.step_number}:\n"
            f"OBJECTIVE: {step.objective}\n"
            f"DEPENDS_ON: {depends_str}\n"
            f"RESULT: {step.result_variable}\n"
            f"OBSERVATION:\n{step.observation}\n"
            f"PREMISES: {premises_str}\n"
            f"CONCLUSION: {step.conclusion}\n"
            f"PATTERN: {step.reasoning_pattern.value}\n"
            f"VERIFICATION: PASSED"
        )
    if session.steps:
        parts.append(
            "All steps above have been verified and accepted. "
            "Continue investigating or provide your FINAL_ANSWER."
        )
    return parts


def build_inference_context(
    session: Session,
    thought: str,
    action: str,
    observation: str,
    objective: str = "",
    result_variable: str = "",
) -> str:
    """Build context for the inference generation LLM call."""
    parts: list[str] = [f"Problem:\n{session.problem_text}"]
    for step in session.steps:
        parts.append(
            f"Step {step.step_number}:\n"
            f"OBJECTIVE: {step.objective}\n"
            f"RESULT: {step.result_variable}\n"
            f"OBSERVATION:\n{step.observation}\n"
            f"INFERENCE: {step.inference}\n"
            f"VERIFICATION: PASSED"
        )
    # Current step (not yet completed) -- full context needed for inference
    step_num = len(session.steps) + 1
    parts.append(
        f"Step {step_num} (current):\n"
        f"OBJECTIVE: {objective}\n"
        f"RESULT_VARIABLE: {result_variable}\n"
        f"THOUGHT: {thought}\n"
        f"ACTION:\n{action}\n"
        f"OBSERVATION:\n{observation}\n"
        f"\nNow provide your PREMISES, CONCLUSION, REASONING_PATTERN, "
        f"and VERIFICATION_TARGET for this step."
    )
    return "\n\n".join(parts)


def build_step_history_summary(session: Session) -> str:
    """Build a concise step history for retry prompts."""
    if not session.steps:
        return "(No prior steps)"
    parts: list[str] = []
    for step in session.steps:
        parts.append(
            f"Step {step.step_number}: {step.conclusion} "
            f"[pattern: {step.reasoning_pattern.value}, "
            f"verified: {step.verification_result.passed}]"
        )
    return "\n".join(parts)


def make_code_repair_prompt(
    thought: str,
    code: str,
    error: str,
    prior_attempts: list[tuple[str, str]] | None = None,
) -> str:
    return build_code_repair_prompt(thought, code, error, prior_attempts)


def make_inference_retry_prompt(
    session: Session,
    observation: str,
    premises: list[str],
    conclusion: str,
    reasoning_pattern: str,
    verification_type: str,
    failure_details: str,
    prior_attempts: list[tuple[str, str, str]] | None = None,
) -> str:
    return build_inference_retry_prompt(
        problem_text=session.problem_text,
        step_history=build_step_history_summary(session),
        observation=observation,
        premises=premises,
        conclusion=conclusion,
        reasoning_pattern=reasoning_pattern,
        verification_type=verification_type,
        failure_details=failure_details,
        prior_attempts=prior_attempts,
    )
