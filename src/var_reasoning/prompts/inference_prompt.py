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
- Use python_assert for concrete claims about specific values
- Use sympy for algebraic identities and equation claims
- Use z3 for any claim involving "for all", "there exists", or
  general relationships between variables
- Use informal ONLY when the claim genuinely cannot be formalized
  (e.g., interpreting natural language meaning)

The verification code will be executed in the same Python sandbox where
prior step variables are available. You can reference variables from
previous steps.

Respond with an inference string and a verification_target object
containing type (z3, sympy, python_assert, or informal), statement
(the verification code), and optionally premises (list of prior step
references).
"""
