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

1. ISOLATION: Verification runs in a SEPARATE environment that does NOT
   have access to the current step's variables. It can only see variables
   from PRIOR completed steps. You must independently recompute or
   cross-check the result — you cannot reference variables set in the
   current step's action code.

2. Verification must do INDEPENDENT work. It will be automatically
   rejected if it only manipulates hardcoded constants. Good verification
   recomputes the value using a DIFFERENT method (e.g., if action used
   a formula, verify with simulation/enumeration, or vice versa).

3. You are checking the ARGUMENT (do premises entail conclusion?),
   not re-running the computation.

4. The required verification type must match the pattern (see table).
   algebraic → sympy. universal_claim → z3. Others → python_assert.

5. Variables from prior steps are available. Current step variables are NOT.

6. Write plain Python. No markdown fences.
"""
