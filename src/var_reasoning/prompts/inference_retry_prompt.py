"""Prompt template for inference revision requests."""

INFERENCE_RETRY_PROMPT_TEMPLATE = """\
Your inference failed verification. Read the error carefully, then choose:

1. "revise" — fix the inference. Provide all four fields: revised_premises,
   revised_conclusion, revised_reasoning_pattern, revised_verification_target.
   - If the error says TAUTOLOGICAL: your verification did no real work.
     Write code that independently recomputes or cross-checks the claim.
   - If the error says pattern/vtype mismatch: use the required type
     (algebraic→sympy, universal_claim→z3).
2. "investigate" — the observation is insufficient. Provide a thought
   and action (Python code) to gather new evidence.

## Problem
{problem_text}

## Prior Steps
{step_history}

## Observation
```
{observation}
```

## Failed Inference
Premises: {premises}
Conclusion: {conclusion}
Pattern: {reasoning_pattern}

## Error ({verification_type})
{failure_details}

{prior_attempts_section}
"""


def build_inference_retry_prompt(
    problem_text: str,
    step_history: str,
    observation: str,
    premises: list[str],
    conclusion: str,
    reasoning_pattern: str,
    verification_type: str,
    failure_details: str,
    prior_attempts: list[tuple[str, str, str]] | None = None,
) -> str:
    prior_section = ""
    if prior_attempts:
        parts = []
        for i, (attempt_inf, attempt_vt, attempt_err) in enumerate(
            prior_attempts, 1
        ):
            parts.append(
                f"### Prior Revision Attempt {i}\n"
                f"Inference: {attempt_inf}\n"
                f"Verification: {attempt_vt}\n"
                f"Result: {attempt_err}"
            )
        prior_section = (
            "## Prior Failed Revision Attempts\n"
            "The following revisions were already tried and also failed:\n\n"
            + "\n\n".join(parts)
        )

    return INFERENCE_RETRY_PROMPT_TEMPLATE.format(
        problem_text=problem_text,
        step_history=step_history,
        observation=observation,
        premises="; ".join(premises) if premises else "(none)",
        conclusion=conclusion,
        reasoning_pattern=reasoning_pattern,
        verification_type=verification_type,
        failure_details=failure_details,
        prior_attempts_section=prior_section,
    )
