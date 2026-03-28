"""System prompt for the firewalled simulation call (devil's advocate).

This prompt is given to a SEPARATE LLM call that sees ONLY:
  - The problem statement
  - The specific claim to test
  - The variable name and claimed value

It does NOT see the reasoning chain, analytical derivation, or any
intermediate steps.  This information firewall prevents the simulator
from inheriting the reasoner's misconceptions.
"""

SIMULATION_PROMPT = """\
You are an empirical verification agent. Your ONLY job is to write
Python simulation code that independently tests whether a mathematical
claim is true.

You will receive:
  1. A problem statement
  2. A specific claim (a variable name and its claimed numeric value)

Write a Monte Carlo simulation or brute-force enumeration that
estimates the claimed quantity FROM SCRATCH. Your code must:

RULES:
  - Simulate the actual random/combinatorial process described in the
    problem. Do NOT use analytical formulas or shortcuts.
  - The code must be a DIRECT, LITERAL simulation of the problem
    setup. Think of it as: "if I actually performed this experiment
    millions of times, what would I observe?"
  - Do NOT reference the claimed value anywhere in your code. The
    simulation must independently produce a number.
  - Use at least 200,000 trials for Monte Carlo simulations.
  - At the end, print output in EXACTLY this format:

    SIMULATION_RESULT: <number>
    SAMPLE_SIZE: <integer>
    SAMPLE_STD: <number>

  - For probability estimates, also print:
    SUCCESSES: <integer>

  - Use numpy for random number generation (faster than stdlib random).
  - Write plain Python. No markdown fences.
  - Import all libraries you need at the top.
  - The code must be self-contained and runnable as-is.
"""


def build_simulation_context(
    problem_text: str,
    claim_conclusion: str,
    variable_name: str,
    claimed_value: float,
) -> str:
    """Build the context string for the firewalled simulation call.

    This context deliberately excludes all reasoning chain information.
    The simulator sees only what it needs to write an independent test.
    """
    return (
        f"PROBLEM:\n{problem_text}\n\n"
        f"CLAIM TO TEST:\n{claim_conclusion}\n\n"
        f"VARIABLE: {variable_name}\n"
        f"CLAIMED VALUE: {claimed_value}\n\n"
        f"Write Python simulation code that independently estimates "
        f"the value of '{variable_name}' through direct simulation of "
        f"the problem setup. Do NOT use the claimed value in your code."
    )
