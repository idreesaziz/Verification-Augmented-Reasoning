"""Adversarial falsification orchestrator.

Receives a DERIVED premise, generates an independent falsification
attempt behind an information firewall, executes it, and returns a
verdict with e-value evidence.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from var_reasoning.models.gemini_provider import GeminiProvider, TokenUsage
from var_reasoning.models.schemas import (
    AttackTool,
    ClaimType,
    Derivation,
    FalsificationAttempt,
    Verdict,
)
from var_reasoning.models.state import FactPool, FalsificationResult
from var_reasoning.prompts.adversary_prompt import (
    ADVERSARY_PROMPT,
    build_adversary_context,
)
from var_reasoning.verification.e_value import (
    EValueResult,
    SimulationStats,
    exact_e_value,
    mean_e_value,
    test_claim,
)

# ── Constants ────────────────────────────────────────────────────────

EXECUTION_TIMEOUT = 45  # seconds
E_REJECT = 1000.0
E_SUSPECT = 20.0


# ── Output parsers ───────────────────────────────────────────────────


def _parse_simulation(stdout: str) -> dict | None:
    """Parse SIMULATION_RESULT / SAMPLE_SIZE / SAMPLE_STD from stdout."""
    result_m = re.search(r"SIMULATION_RESULT:\s*([\d.eE+\-]+)", stdout)
    size_m = re.search(r"SAMPLE_SIZE:\s*(\d+)", stdout)
    std_m = re.search(r"SAMPLE_STD:\s*([\d.eE+\-]+)", stdout)
    if result_m and size_m:
        return {
            "empirical_value": float(result_m.group(1)),
            "sample_size": int(size_m.group(1)),
            "sample_std": float(std_m.group(1)) if std_m else 0.0,
        }
    return None


def _parse_z3(stdout: str) -> dict | None:
    """Parse Z3_RESULT from stdout."""
    result_m = re.search(r"Z3_RESULT:\s*(SAT|UNSAT)", stdout, re.IGNORECASE)
    if result_m:
        sat = result_m.group(1).upper() == "SAT"
        cx_m = re.search(r"COUNTEREXAMPLE:\s*(.+)", stdout)
        return {
            "sat": sat,
            "counterexample": cx_m.group(1).strip() if cx_m else None,
        }
    return None


def _parse_brute_force(stdout: str) -> dict | None:
    """Parse BRUTEFORCE_RESULT / TOTAL_CASES from stdout."""
    result_m = re.search(r"BRUTEFORCE_RESULT:\s*([\d.eE+\-]+)", stdout)
    total_m = re.search(r"TOTAL_CASES:\s*(\d+)", stdout)
    if result_m:
        return {
            "empirical_value": float(result_m.group(1)),
            "total_cases": int(total_m.group(1)) if total_m else 0,
        }
    return None


# ── Subprocess executor (isolated, no state) ────────────────────────


def _execute_adversary_code(code: str, timeout: int = EXECUTION_TIMEOUT) -> tuple[bool, str]:
    """Run adversary code in an isolated subprocess."""
    # Strip markdown fences
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout}s"
    finally:
        Path(script_path).unlink(missing_ok=True)


# ── Core orchestrator ────────────────────────────────────────────────


def falsify_derivation(
    derivation: Derivation,
    problem_text: str,
    fact_pool: FactPool,
    provider: GeminiProvider,
) -> tuple[FalsificationResult, TokenUsage]:
    """Attempt to falsify a single DERIVED premise.

    Pipeline:
      1. Build firewalled context (problem + claim + cited facts only)
      2. LLM generates FalsificationAttempt (attack tool + code)
      3. Execute code in isolated subprocess
      4. Parse output and run e-value test
      5. Return verdict

    Returns (FalsificationResult, token_usage).
    """
    # 1) Build context for adversary
    depends_rendered = fact_pool.render_for_adversary(derivation.depends_on)
    context = build_adversary_context(
        problem_text=problem_text,
        premise=derivation.premise,
        claimed_value=derivation.claimed_value,
        depends_on_rendered=depends_rendered,
    )

    # 2) Generate falsification attempt
    try:
        attempt, usage = provider.generate_falsification(
            system_prompt=ADVERSARY_PROMPT,
            context=context,
        )
    except Exception as e:
        return FalsificationResult(
            executed=False,
            verdict=Verdict.INCONCLUSIVE,
            feedback=f"Adversary LLM call failed: {e}",
        ), TokenUsage()

    # 3) Execute the adversary's code
    success, stdout = _execute_adversary_code(attempt.code)

    if not success:
        return FalsificationResult(
            executed=False,
            verdict=Verdict.INCONCLUSIVE,
            hidden_assumptions=attempt.hidden_assumptions,
            feedback=f"Adversary code failed: {stdout[:500]}",
        ), usage

    # 4) Parse output based on attack tool
    result = _evaluate_output(
        attempt=attempt,
        stdout=stdout,
        claimed_value=derivation.claimed_value,
    )
    result.hidden_assumptions = attempt.hidden_assumptions
    return result, usage


def _evaluate_output(
    attempt: FalsificationAttempt,
    stdout: str,
    claimed_value: float | None,
) -> FalsificationResult:
    """Parse execution output and compute e-value verdict."""

    if attempt.attack_tool == AttackTool.Z3:
        return _evaluate_z3(stdout, claimed_value)
    elif attempt.attack_tool == AttackTool.MONTE_CARLO:
        return _evaluate_simulation(stdout, claimed_value)
    elif attempt.attack_tool == AttackTool.BRUTE_FORCE:
        return _evaluate_brute_force(stdout, claimed_value)

    return FalsificationResult(
        executed=True,
        verdict=Verdict.INCONCLUSIVE,
        feedback=f"Unknown attack tool: {attempt.attack_tool}",
    )


def _evaluate_z3(stdout: str, claimed_value: float | None) -> FalsificationResult:
    """Evaluate Z3 solver output."""
    parsed = _parse_z3(stdout)
    if parsed is None:
        return FalsificationResult(
            executed=True,
            verdict=Verdict.INCONCLUSIVE,
            feedback=f"Could not parse Z3 output: {stdout[:300]}",
        )

    if parsed["sat"]:
        # Found counterexample — claim is false
        cx = parsed.get("counterexample", "")
        return FalsificationResult(
            executed=True,
            verdict=Verdict.REJECT,
            e_value=float("inf"),
            claimed_value=claimed_value,
            feedback=f"Z3 found counterexample: {cx}",
        )
    else:
        # UNSAT — no counterexample found
        return FalsificationResult(
            executed=True,
            verdict=Verdict.SURVIVE,
            e_value=1.0,
            claimed_value=claimed_value,
            feedback="Z3: no counterexample found (UNSAT).",
        )


def _evaluate_simulation(
    stdout: str, claimed_value: float | None
) -> FalsificationResult:
    """Evaluate Monte Carlo simulation output."""
    parsed = _parse_simulation(stdout)
    if parsed is None:
        return FalsificationResult(
            executed=True,
            verdict=Verdict.INCONCLUSIVE,
            feedback=f"Could not parse simulation output: {stdout[:300]}",
        )

    empirical = parsed["empirical_value"]
    sample_size = parsed["sample_size"]
    sample_std = parsed["sample_std"]

    if claimed_value is None:
        # No numeric claim to test — just report
        return FalsificationResult(
            executed=True,
            verdict=Verdict.INCONCLUSIVE,
            empirical_value=empirical,
            sample_size=sample_size,
            feedback=f"Simulation result: {empirical} (n={sample_size}), but no claimed value to compare.",
        )

    # Run e-value test
    stats = SimulationStats(
        sample_mean=empirical,
        sample_size=sample_size,
        sample_std=sample_std,
    )
    ev_result = test_claim(claimed_value, stats)

    verdict = Verdict.REJECT if ev_result.verdict.value == "reject" else Verdict.SURVIVE

    feedback_parts = [
        f"Claimed: {claimed_value}, Empirical: {empirical:.6f}",
        f"n={sample_size}, std={sample_std:.6f}",
        f"e-value={ev_result.e_value:.2f}",
        f"CI=({ev_result.confidence_interval[0]:.6f}, {ev_result.confidence_interval[1]:.6f})",
    ]

    return FalsificationResult(
        executed=True,
        verdict=verdict,
        e_value=ev_result.e_value,
        claimed_value=claimed_value,
        empirical_value=empirical,
        sample_size=sample_size,
        feedback="; ".join(feedback_parts),
    )


def _evaluate_brute_force(
    stdout: str, claimed_value: float | None
) -> FalsificationResult:
    """Evaluate brute-force enumeration output."""
    parsed = _parse_brute_force(stdout)
    if parsed is None:
        return FalsificationResult(
            executed=True,
            verdict=Verdict.INCONCLUSIVE,
            feedback=f"Could not parse brute-force output: {stdout[:300]}",
        )

    empirical = parsed["empirical_value"]
    total_cases = parsed["total_cases"]

    if claimed_value is None:
        return FalsificationResult(
            executed=True,
            verdict=Verdict.INCONCLUSIVE,
            empirical_value=empirical,
            sample_size=total_cases,
            feedback=f"Brute-force result: {empirical} (cases={total_cases}), no claimed value.",
        )

    # Exact comparison for deterministic counts
    e = exact_e_value(claimed_value, empirical)
    if e == float("inf"):
        verdict = Verdict.REJECT
        feedback = f"MISMATCH: claimed {claimed_value}, brute-force found {empirical} ({total_cases} cases)"
    else:
        verdict = Verdict.SURVIVE
        feedback = f"MATCH: claimed {claimed_value} == brute-force {empirical} ({total_cases} cases)"

    return FalsificationResult(
        executed=True,
        verdict=verdict,
        e_value=e,
        claimed_value=claimed_value,
        empirical_value=empirical,
        sample_size=total_cases,
        feedback=feedback,
    )
