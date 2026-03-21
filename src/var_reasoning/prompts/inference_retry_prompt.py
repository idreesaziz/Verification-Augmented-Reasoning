"""Prompt template for inference revision requests."""

INFERENCE_RETRY_PROMPT_TEMPLATE = """\
Your inference failed formal verification. You must either revise the
inference with a new verification target, or request a new investigation.

## Problem Context
{problem_text}

## Step History
{step_history}

## Current Observation
The code executed successfully and produced:
```
{observation}
```

## Failed Inference
{inference}

## Verification Failure
Type: {verification_type}
{failure_details}

{prior_attempts_section}

Choose one:
1. "revise" — Provide a corrected inference and updated verification
   target that will pass verification.
2. "investigate" — The observation is insufficient. Provide a new
   thought and action (Python code) to gather more information.
"""


def build_inference_retry_prompt(
    problem_text: str,
    step_history: str,
    observation: str,
    inference: str,
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
        inference=inference,
        verification_type=verification_type,
        failure_details=failure_details,
        prior_attempts_section=prior_section,
    )
