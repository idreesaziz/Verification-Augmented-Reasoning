"""Prompt template for silent code repair requests."""

CODE_REPAIR_PROMPT_TEMPLATE = """\
The following Python code was written to accomplish a specific goal but
failed to execute. Fix the code so it runs correctly.

## Intent
{thought}

## Failed Code
```python
{code}
```

## Error
```
{error}
```

{prior_attempts_section}

Provide the fixed code and a brief explanation of what you changed.
The fixed code must be complete and self-contained (not a diff).
"""


def build_code_repair_prompt(
    thought: str,
    code: str,
    error: str,
    prior_attempts: list[tuple[str, str]] | None = None,
) -> str:
    prior_section = ""
    if prior_attempts:
        parts = []
        for i, (attempt_code, attempt_error) in enumerate(prior_attempts, 1):
            parts.append(
                f"### Prior Attempt {i}\n"
                f"```python\n{attempt_code}\n```\n"
                f"Error:\n```\n{attempt_error}\n```"
            )
        prior_section = (
            "## Prior Failed Attempts\n"
            "The following fixes were already tried and also failed:\n\n"
            + "\n\n".join(parts)
        )

    return CODE_REPAIR_PROMPT_TEMPLATE.format(
        thought=thought,
        code=code,
        error=error,
        prior_attempts_section=prior_section,
    )
