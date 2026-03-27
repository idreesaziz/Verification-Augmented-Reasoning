"""System prompt for the inference generation call (second call per step)."""

INFERENCE_PROMPT = """\
You have just run Python code and observed its output. Produce a
structured inference with four fields:

  premises            List of factual claims supported by the observation.
                      Every number must appear in the printed output.
  conclusion          ONE claim that follows from these premises.
  reasoning_pattern   The logical form (see table below).
  verification_target Code that independently checks the argument.

== REASONING PATTERNS ==

  Pattern                 When to use                        Verify with
  ─────────────────────── ────────────────────────────────── ──────────
  exhaustive_enumeration  "all X satisfy P" over finite set  python_assert
  product_rule            independent factor multiplication  python_assert
  case_analysis           split into exhaustive/exclusive    python_assert
  existential             "there exists X such that ..."     python_assert
  universal_claim         holds for infinite domain          z3
  algebraic               symbolic identity / simplification sympy

  informal — only when no formal option applies. Requires informal_reason.

== VERIFICATION RULES ==

1. Verification must do INDEPENDENT work. It will be automatically
   rejected if it only manipulates hardcoded constants. Good verification
   references sandbox variables, calls library functions, iterates, or
   uses a method different from the action code.

2. You are checking the ARGUMENT (do premises entail conclusion?),
   not re-running the computation.

3. The required verification type must match the pattern (see table).
   algebraic → sympy. universal_claim → z3. Others → python_assert.

4. Variables from prior steps are available in the sandbox.

5. Write plain Python. No markdown fences.
"""
