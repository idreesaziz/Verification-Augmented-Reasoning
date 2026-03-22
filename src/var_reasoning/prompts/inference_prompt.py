"""System prompt for the inference generation call (second call per step)."""

INFERENCE_PROMPT = """\
You have just executed Python code and observed its output. Now:
1. State the PREMISES — what the code's output factually showed you.
2. State your CONCLUSION — the claim you are drawing from those premises.
3. Provide a VERIFICATION TARGET that checks whether your conclusion
   LOGICALLY FOLLOWS from your premises.

== WHAT VERIFICATION MEANS HERE ==

You are NOT re-checking the computation or re-running it differently.
You are checking the LOGICAL VALIDITY of your argument: given that
the premises are true, does the conclusion necessarily follow?

Think of it as predicate logic:  P1 ∧ P2 ∧ P3 → C
Your verification must formalize this argument and check its soundness.

== EXAMPLES OF LOGICAL VALIDITY CHECKS ==

Example 1 — Exhaustive enumeration → "ALL" quantifier:
  Premises:
    P1: Code iterated over range(100, 1000)
    P2: Filter condition was n == sum(d**3 for d in digits(n))
    P3: Output was [153, 370, 371, 407]
  Conclusion: These are ALL 3-digit numbers equal to the sum of cubes
    of their digits.
  Logical structure: "ALL" is valid ONLY IF the enumeration covered
    EVERY 3-digit number AND the filter correctly encodes the condition.
  Verification (python_assert):
    # Check P1 entails exhaustive coverage of 3-digit numbers
    three_digit_range = set(range(100, 1000))
    assert len(three_digit_range) == 900
    assert min(three_digit_range) == 100 and max(three_digit_range) == 999
    # Check P2 correctly encodes the problem condition
    for n in [153, 370, 371, 407]:
        digits = [int(d) for d in str(n)]
        assert n == sum(d**3 for d in digits)
    # P1 (exhaustive) ∧ P2 (correct filter) → "ALL" is justified

Example 2 — Product rule (independence of choices):
  Premises:
    P1: C(5,3) = 10 ways to choose dice positions
    P2: 6 choices for the repeated face
    P3: 5*4 = 20 ordered ways for two remaining distinct faces
  Conclusion: Favorable outcomes = 6 * 10 * 20 = 1200
  Logical structure: The product rule applies ONLY IF the choices
    are independent (choosing which face repeats does not constrain
    which dice positions are chosen, etc.)
  Verification (python_assert):
    import math
    # Check independence: each factor has a distinct combinatorial role
    assert num_ways_to_choose_dice == math.comb(5, 3)  # positions
    assert num_choices_for_other_two_dice == 5 * 4  # ordered distinct
    # Check the multiplication structure matches the product rule
    assert favorable_outcomes == num_choices_for_repeated_number * \\
        num_ways_to_choose_dice * num_choices_for_other_two_dice

Example 3 — Universal claim (z3 counterexample search):
  Premises:
    P1: Code tested f(x) > 0 for x in [1..100], all passed
  Conclusion: f(x) > 0 for all positive integers x
  Logical structure: Finite testing does NOT prove a universal claim.
    Use z3 to search for a counterexample.
  Verification (z3):
    from z3 import *
    x = Int('x')
    s = Solver()
    s.add(x > 0)
    s.add(f(x) <= 0)  # try to find counterexample
    assert s.check() == unsat  # no counterexample → valid

== WHAT LOGICAL STRUCTURE TO CHECK ==

Ask yourself: "What reasoning pattern does my conclusion rely on?"
Then verify THAT pattern holds:

  * "ALL x satisfy P" ← was the search exhaustive?
  * "A * B * C" ← are A, B, C independent choices (product rule)?
  * "If P then Q; P holds; therefore Q" ← does P actually hold?
  * "By cases: ..." ← are all cases covered?
  * "Algebraic identity" ← does the identity hold symbolically?
  * "There exists x such that" ← was one actually found?

== VERIFICATION TYPES ==

- python_assert: Check structural validity of the argument — does
  exhaustiveness justify "all"? Does independence justify multiplication?
  Does the encoding match the problem statement?
  Use Python's `assert` keyword. Do NOT write python_assert().

- z3: Search for counterexamples to universal/existential claims.
  If no counterexample exists, the argument is valid.

- sympy: Check algebraic identities and symbolic reasoning.

- informal: ONLY when the logical structure genuinely cannot be
  formalized. Last resort.

== KEY RULES ==

- You are verifying the ARGUMENT, not the ANSWER.
- NEVER just assert that a variable equals its computed value.
- Write PLAIN Python code. No markdown fences.
- Variables from prior steps are available in the sandbox.

Respond with an inference string and a verification_target object
containing type (z3, sympy, python_assert, or informal), statement
(the verification code), and optionally premises (list of prior step
references).
"""
