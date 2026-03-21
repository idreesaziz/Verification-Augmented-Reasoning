"""System prompt for the inference generation call (second call per step)."""

INFERENCE_PROMPT = """\
You have just executed Python code and observed its output. Now draw
a conclusion and provide a formal verification target.

Your INFERENCE must:
- State what specifically follows from this observation
- Connect it to the overall problem and prior findings
- Be explicit about the logical chain: reference specific prior steps
- Go beyond restating the code output

Your VERIFICATION_TARGET must:
- Capture the logical content of your inference as mechanically
  checkable code
- Use python_assert for concrete claims about specific values.
  The statement must use Python's built-in `assert` keyword, e.g.:
    assert x == 42
    assert abs(result - 3.14) < 1e-9
  Do NOT write python_assert(...) — that is not a function. Use assert.
- Use sympy for algebraic identities and equation claims
- Use z3 for any claim involving "for all", "there exists", or
  general relationships between variables
- Use informal ONLY when the claim genuinely cannot be formalized
  (e.g., interpreting natural language meaning)

CRITICAL RULES for verification statements:
- Write PLAIN Python code. Do NOT wrap it in markdown fences (```python).
- Use the `assert` keyword, NOT a function called `python_assert`.
- Variables from prior steps are available in the sandbox.

Respond with an inference string and a verification_target object
containing type (z3, sympy, python_assert, or informal), statement
(the verification code), and optionally premises (list of prior step
references).
"""
