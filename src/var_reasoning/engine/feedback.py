"""Feedback builders for the v2 engine.

Builds conversation history from the fact pool and completed steps.
"""

from __future__ import annotations

from var_reasoning.models.schemas import Derivation, Verdict
from var_reasoning.models.state import CompletedStep, FalsificationResult, Session
from var_reasoning.prompts.code_repair_prompt import build_code_repair_prompt


def build_conversation_history(session: Session) -> list[str]:
    """Build conversation for the reasoning LLM call.

    Includes the problem, the current fact pool, and a summary of
    each completed step (objective, observation, accepted derivations).
    """
    parts: list[str] = [f"Problem:\n{session.problem_text}"]

    # Fact pool
    pool_str = session.fact_pool.render_for_reasoner()
    parts.append(f"FACT POOL:\n{pool_str}")

    # Prior steps
    for step in session.steps:
        derivation_strs = []
        for d in step.derivations:
            deps = ", ".join(d.depends_on) if d.depends_on else "(none)"
            val = f" = {d.claimed_value}" if d.claimed_value is not None else ""
            derivation_strs.append(f"    - {d.premise}{val}  [depends: {deps}]")
        derivation_block = "\n".join(derivation_strs) if derivation_strs else "    (none)"

        computed = ", ".join(step.computed_fact_ids) if step.computed_fact_ids else "(none)"
        derived = ", ".join(step.derived_fact_ids) if step.derived_fact_ids else "(none)"

        parts.append(
            f"Step {step.step_number}:\n"
            f"  OBJECTIVE: {step.objective}\n"
            f"  RESULT: {step.result_variable}\n"
            f"  OBSERVATION: {step.observation}\n"
            f"  DERIVATIONS:\n{derivation_block}\n"
            f"  NEW COMPUTED FACTS: {computed}\n"
            f"  NEW DERIVED FACTS: {derived}\n"
            f"  STATUS: ACCEPTED"
        )

    if session.steps:
        parts.append(
            "All steps above are verified and accepted. "
            "Continue investigating or provide your FINAL_ANSWER."
        )
    return parts


def build_rejection_feedback(
    derivation: Derivation,
    result: FalsificationResult,
) -> str:
    """Build feedback for the reasoner when a derivation is rejected."""
    parts = [
        f"DERIVATION REJECTED: {derivation.premise}",
        f"Feedback: {result.feedback}",
    ]
    if result.hidden_assumptions:
        parts.append(f"Hidden assumptions found: {'; '.join(result.hidden_assumptions)}")
    if result.empirical_value is not None:
        parts.append(f"Empirical value: {result.empirical_value}")
    if derivation.claimed_value is not None:
        parts.append(f"Your claimed value: {derivation.claimed_value}")
    parts.append(
        "Please revise your reasoning. The above derivation was independently "
        "tested and found to be inconsistent with empirical evidence."
    )
    return "\n".join(parts)


def make_code_repair_prompt(
    thought: str,
    code: str,
    error: str,
    prior_attempts: list[tuple[str, str]] | None = None,
) -> str:
    return build_code_repair_prompt(thought, code, error, prior_attempts)
