"""Feedback builders for the engine's retry loops."""

from __future__ import annotations

from var_reasoning.models.state import CompletedStep, Session
from var_reasoning.prompts.code_repair_prompt import build_code_repair_prompt
from var_reasoning.prompts.inference_retry_prompt import build_inference_retry_prompt


def build_conversation_history(session: Session) -> list[str]:
    """Build the conversation history for the reasoning LLM call."""
    parts: list[str] = [f"Problem:\n{session.problem_text}"]
    for step in session.steps:
        parts.append(
            f"Step {step.step_number}:\n"
            f"THOUGHT: {step.thought}\n"
            f"ACTION:\n```python\n{step.action}\n```\n"
            f"OBSERVATION:\n{step.observation}\n"
            f"INFERENCE: {step.inference}"
        )
    return parts


def build_inference_context(
    session: Session,
    thought: str,
    action: str,
    observation: str,
) -> str:
    """Build context for the inference generation LLM call."""
    parts: list[str] = [f"Problem:\n{session.problem_text}"]
    for step in session.steps:
        parts.append(
            f"Step {step.step_number}:\n"
            f"THOUGHT: {step.thought}\n"
            f"ACTION:\n```python\n{step.action}\n```\n"
            f"OBSERVATION:\n{step.observation}\n"
            f"INFERENCE: {step.inference}"
        )
    # Current step (not yet completed)
    step_num = len(session.steps) + 1
    parts.append(
        f"Step {step_num} (current):\n"
        f"THOUGHT: {thought}\n"
        f"ACTION:\n```python\n{action}\n```\n"
        f"OBSERVATION:\n{observation}\n"
        f"\nNow provide your INFERENCE and VERIFICATION_TARGET for this step."
    )
    return "\n\n".join(parts)


def build_step_history_summary(session: Session) -> str:
    """Build a concise step history for retry prompts."""
    if not session.steps:
        return "(No prior steps)"
    parts: list[str] = []
    for step in session.steps:
        parts.append(
            f"Step {step.step_number}: {step.inference} "
            f"[verified: {step.verification_result.passed}]"
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
    inference: str,
    verification_type: str,
    failure_details: str,
    prior_attempts: list[tuple[str, str, str]] | None = None,
) -> str:
    return build_inference_retry_prompt(
        problem_text=session.problem_text,
        step_history=build_step_history_summary(session),
        observation=observation,
        inference=inference,
        verification_type=verification_type,
        failure_details=failure_details,
        prior_attempts=prior_attempts,
    )
