"""Firewalled simulation verifier.

Orchestrates the devil's-advocate pipeline:
  1. Firewalled LLM call → simulation code (no reasoning chain context)
  2. Static sanity gate → reject circular simulations
  3. Execute simulation in an isolated namespace
  4. Parse output → feed to e-value engine
  5. Return verdict
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from var_reasoning.models.gemini_provider import GeminiProvider
from var_reasoning.prompts.simulation_prompt import (
    SIMULATION_PROMPT,
    build_simulation_context,
)
from var_reasoning.verification.e_value import (
    EValueResult,
    SimulationStats,
    Verdict,
    test_claim,
)
from var_reasoning.verification.static_sanity import check_no_claimed_value_in_code

logger = logging.getLogger(__name__)

# Number of times to retry simulation generation if code fails
_SIM_RETRIES = 2


@dataclass(frozen=True)
class SimulationResult:
    """Full result of the simulation verification layer."""

    ran: bool  # Whether a simulation was actually executed
    verdict: Verdict | None = None  # None if simulation was skipped
    e_value: float = 1.0
    claimed_value: float | None = None
    empirical_value: float | None = None
    sample_size: int = 0
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    detail: str = ""
    simulation_code: str = ""

    @property
    def passed(self) -> bool:
        """True if the simulation didn't reject the claim."""
        if not self.ran:
            return True  # No simulation → no evidence against
        return self.verdict != Verdict.REJECT


# ── Output parsing ───────────────────────────────────────────────────

_RESULT_RE = re.compile(r"SIMULATION_RESULT:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.I)
_SIZE_RE = re.compile(r"SAMPLE_SIZE:\s*(\d+)", re.I)
_STD_RE = re.compile(r"SAMPLE_STD:\s*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", re.I)
_SUCCESS_RE = re.compile(r"SUCCESSES:\s*(\d+)", re.I)


def _parse_simulation_output(output: str) -> SimulationStats | None:
    """Parse the structured output from a simulation run."""
    m_result = _RESULT_RE.search(output)
    m_size = _SIZE_RE.search(output)
    if not m_result or not m_size:
        return None

    sample_mean = float(m_result.group(1))
    sample_size = int(m_size.group(1))
    if sample_size <= 0:
        return None

    m_std = _STD_RE.search(output)
    sample_std = float(m_std.group(1)) if m_std else 0.0

    m_success = _SUCCESS_RE.search(output)
    successes = int(m_success.group(1)) if m_success else None

    return SimulationStats(
        sample_mean=sample_mean,
        sample_size=sample_size,
        sample_std=sample_std,
        successes=successes,
    )


def try_parse_numeric(observation: str) -> float | None:
    """Try to extract a single numeric value from an observation string."""
    obs = observation.strip()
    # Direct parse
    try:
        return float(obs)
    except ValueError:
        pass
    # Last line
    for line in reversed(obs.splitlines()):
        line = line.strip()
        try:
            return float(line)
        except ValueError:
            pass
    # Single number anywhere
    matches = re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", obs)
    if len(matches) == 1:
        return float(matches[0])
    return None


# ── Main verifier ────────────────────────────────────────────────────


class SimulationVerifier:
    """Runs the firewalled simulation pipeline for a single claim."""

    def __init__(
        self,
        gemini: GeminiProvider,
        executor_factory,
    ) -> None:
        self._gemini = gemini
        self._executor_factory = executor_factory

    def verify(
        self,
        problem_text: str,
        claim_conclusion: str,
        variable_name: str,
        claimed_value: float,
        prior_observations: list[str] | None = None,
    ) -> SimulationResult:
        """Run the full simulation verification pipeline.

        Returns a SimulationResult with the verdict.  The caller can
        inspect ``result.passed`` for a quick pass/fail check.
        """
        context = build_simulation_context(
            problem_text, claim_conclusion, variable_name, claimed_value,
        )

        for attempt in range(_SIM_RETRIES + 1):
            # Step 1: Firewalled LLM call
            try:
                sim_code_obj, usage = self._gemini.generate_simulation(
                    SIMULATION_PROMPT, context,
                )
            except Exception as exc:
                logger.warning("Simulation LLM call failed: %s", exc)
                continue

            sim_code = sim_code_obj.code

            # Step 2: Static sanity — reject circular simulations
            sanity = check_no_claimed_value_in_code(sim_code, claimed_value)
            if not sanity.passed:
                logger.info(
                    "Simulation code rejected (circular): %s", sanity.reason,
                )
                # Append a hint to the context for the retry
                context += (
                    "\n\nIMPORTANT: Your previous simulation code contained "
                    f"the claimed value {claimed_value} as a literal. "
                    "The simulation must INDEPENDENTLY derive the value. "
                    "Do NOT embed the claimed value in your code."
                )
                continue

            # Step 3: Execute in an isolated namespace
            executor = self._executor_factory()
            executor.reset_namespace()
            success, output = executor.execute(sim_code, timeout=45)
            if not success:
                logger.info(
                    "Simulation execution failed (attempt %d): %s",
                    attempt + 1, output[:200],
                )
                continue

            # Step 4: Parse output
            stats = _parse_simulation_output(output)
            if stats is None:
                logger.info(
                    "Failed to parse simulation output (attempt %d): %s",
                    attempt + 1, output[:200],
                )
                continue

            # Step 5: e-value test
            e_result: EValueResult = test_claim(claimed_value, stats)
            logger.info(
                "Simulation verdict: %s (e=%.2f, claimed=%.6f, empirical=%.6f, n=%d)",
                e_result.verdict.value,
                e_result.e_value,
                claimed_value,
                e_result.empirical_value,
                stats.sample_size,
            )
            return SimulationResult(
                ran=True,
                verdict=e_result.verdict,
                e_value=e_result.e_value,
                claimed_value=claimed_value,
                empirical_value=e_result.empirical_value,
                sample_size=stats.sample_size,
                confidence_interval=e_result.confidence_interval,
                detail=e_result.detail,
                simulation_code=sim_code,
            )

        # All retries exhausted — simulation inconclusive
        return SimulationResult(
            ran=False,
            detail="Simulation generation/execution failed after all retries.",
        )
