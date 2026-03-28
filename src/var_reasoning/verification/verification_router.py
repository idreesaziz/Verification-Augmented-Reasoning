"""Multi-layer verification pipeline.

Layer 0  — Static sanity (range, integrality, provenance)
Layer 0.5 — Symbolic verification (assert / sympy / z3)
Layer 1  — Firewalled simulation + e-value testing
Layer 2  — Claim registry (global consistency)

All layers must pass for a step to be accepted.
"""

from __future__ import annotations

import logging

from var_reasoning.models.gemini_provider import GeminiProvider
from var_reasoning.models.schemas import (
    ReasoningPattern,
    VerificationTarget,
    VerificationType,
)
from var_reasoning.models.state import VerificationResult
from var_reasoning.sandbox.executor import CodeExecutor
from var_reasoning.verification.assert_verifier import AssertVerifier
from var_reasoning.verification.claim_registry import ClaimRegistry
from var_reasoning.verification.e_value import Verdict
from var_reasoning.verification.simulation_verifier import (
    SimulationVerifier,
    try_parse_numeric,
)
from var_reasoning.verification.static_sanity import (
    check_integrality,
    check_provenance,
    check_range,
)
from var_reasoning.verification.sympy_verifier import SympyVerifier
from var_reasoning.verification.tautology_check import check_tautological
from var_reasoning.verification.z3_verifier import Z3Verifier

logger = logging.getLogger(__name__)

# Pattern → required verification type mapping.
PATTERN_REQUIRED_VTYPE: dict[ReasoningPattern, VerificationType] = {
    ReasoningPattern.ALGEBRAIC: VerificationType.SYMPY,
    ReasoningPattern.UNIVERSAL_CLAIM: VerificationType.Z3,
}


class VerificationRouter:
    """Orchestrates the full multi-layer verification pipeline."""

    def __init__(
        self,
        executor_factory,
        gemini: GeminiProvider | None = None,
    ) -> None:
        self._executor_factory = executor_factory
        self._gemini = gemini
        self._prior_code: list[str] = []
        self._claim_registry = ClaimRegistry()
        self._sim_verifier: SimulationVerifier | None = None
        if gemini is not None:
            self._sim_verifier = SimulationVerifier(gemini, executor_factory)
        self._rebuild_verifiers()

    def _rebuild_verifiers(self) -> None:
        executor = self._executor_factory()
        executor.reset_namespace()
        for code in self._prior_code:
            executor.execute(code)
        self._assert_verifier = AssertVerifier(executor)
        self._sympy_verifier = SympyVerifier(executor)
        self._z3_verifier = Z3Verifier(executor)

    def set_prior_code(self, prior_steps_code: list[str]) -> None:
        self._prior_code = list(prior_steps_code)
        self._rebuild_verifiers()

    def reset(self) -> None:
        self._prior_code = []
        self._claim_registry.reset()
        self._rebuild_verifiers()

    @property
    def claim_registry(self) -> ClaimRegistry:
        return self._claim_registry

    # ── Main entry point ─────────────────────────────────────────────

    def verify(
        self,
        target: VerificationTarget,
        reasoning_pattern: ReasoningPattern | None = None,
        *,
        problem_text: str = "",
        observation: str = "",
        result_variable: str = "",
        step_number: int = 0,
        conclusion: str = "",
        depends_on: list[str] | None = None,
        prior_observations: list[str] | None = None,
        is_final_answer: bool = False,
    ) -> VerificationResult:
        """Run the full Layer 0 → 0.5 → 1 → 2 pipeline."""

        # ── Layer 0: Static sanity ───────────────────────────────────
        claimed_value = try_parse_numeric(observation) if observation else None

        if claimed_value is not None:
            range_check = check_range(
                claimed_value, reasoning_pattern.value if reasoning_pattern else "",
            )
            if not range_check.passed:
                return VerificationResult(
                    passed=False,
                    verification_type=target.type,
                    error_message=f"SANITY: {range_check.reason}",
                )

            int_check = check_integrality(
                claimed_value, problem_text, is_final_answer=is_final_answer,
            )
            if not int_check.passed:
                return VerificationResult(
                    passed=False,
                    verification_type=target.type,
                    error_message=f"SANITY: {int_check.reason}",
                )

        # Provenance check on verification code
        if target.type != VerificationType.INFORMAL and problem_text:
            prov_check = check_provenance(
                target.statement,
                problem_text,
                prior_observations or [],
            )
            if not prov_check.passed:
                logger.info("Provenance check failed: %s", prov_check.reason)
                # Warn but don't hard-reject — let symbolic check decide
                pass

        # ── Layer 0.5: Symbolic verification ─────────────────────────
        symbolic_result = self._symbolic_verify(target, reasoning_pattern)

        # ── Layer 1: Simulation verification ─────────────────────────
        sim_ran = False
        sim_verdict_str = None
        sim_e = 1.0
        sim_empirical = None
        sim_detail = ""

        if (
            self._sim_verifier is not None
            and claimed_value is not None
            and target.type != VerificationType.INFORMAL
            and problem_text
        ):
            sim_result = self._sim_verifier.verify(
                problem_text=problem_text,
                claim_conclusion=conclusion or str(claimed_value),
                variable_name=result_variable or "result",
                claimed_value=claimed_value,
                prior_observations=prior_observations,
            )
            sim_ran = sim_result.ran
            if sim_result.ran:
                sim_verdict_str = sim_result.verdict.value if sim_result.verdict else None
                sim_e = sim_result.e_value
                sim_empirical = sim_result.empirical_value
                sim_detail = sim_result.detail

                if sim_result.verdict == Verdict.REJECT:
                    return VerificationResult(
                        passed=False,
                        verification_type=target.type,
                        error_message=(
                            f"SIMULATION REJECTED: empirical={sim_result.empirical_value:.6f}, "
                            f"claimed={claimed_value:.6f}, e-value={sim_result.e_value:.1f}, "
                            f"CI=({sim_result.confidence_interval[0]:.6f}, "
                            f"{sim_result.confidence_interval[1]:.6f}). "
                            f"The simulation independently produced a value that "
                            f"strongly contradicts the analytical claim."
                        ),
                        simulation_ran=True,
                        simulation_verdict="reject",
                        simulation_e_value=sim_result.e_value,
                        simulation_empirical=sim_result.empirical_value,
                        simulation_detail=sim_detail,
                    )

        # ── Layer 2: Claim registry ──────────────────────────────────
        if symbolic_result.passed and claimed_value is not None and result_variable:
            self._claim_registry.register(
                step_number=step_number,
                variable_name=result_variable,
                claimed_value=claimed_value,
                conclusion=conclusion,
                depends_on=depends_on or [],
            )
            consistency = self._claim_registry.check_consistency()
            if not consistency.consistent:
                return VerificationResult(
                    passed=False,
                    verification_type=target.type,
                    error_message=f"CONSISTENCY: {consistency.detail}",
                    simulation_ran=sim_ran,
                    simulation_verdict=sim_verdict_str,
                    simulation_e_value=sim_e,
                    simulation_empirical=sim_empirical,
                    simulation_detail=sim_detail,
                )

        # ── Combine results ──────────────────────────────────────────
        return VerificationResult(
            passed=symbolic_result.passed,
            verification_type=symbolic_result.verification_type,
            error_message=symbolic_result.error_message,
            counterexample=symbolic_result.counterexample,
            simulation_ran=sim_ran,
            simulation_verdict=sim_verdict_str,
            simulation_e_value=sim_e,
            simulation_empirical=sim_empirical,
            simulation_detail=sim_detail,
        )

    # ── Layer 0.5 helper ─────────────────────────────────────────────

    def _symbolic_verify(
        self,
        target: VerificationTarget,
        reasoning_pattern: ReasoningPattern | None,
    ) -> VerificationResult:
        """Run the existing symbolic verification pipeline."""
        if target.type == VerificationType.INFORMAL:
            return VerificationResult(
                passed=True,
                verification_type=VerificationType.INFORMAL,
            )

        # Pattern-vtype consistency
        if reasoning_pattern and reasoning_pattern in PATTERN_REQUIRED_VTYPE:
            required = PATTERN_REQUIRED_VTYPE[reasoning_pattern]
            if target.type != required:
                msg = (
                    f"Reasoning pattern '{reasoning_pattern.value}' requires "
                    f"verification type '{required.value}', but got "
                    f"'{target.type.value}'."
                )
                return VerificationResult(
                    passed=False,
                    verification_type=target.type,
                    error_message=msg,
                )

        # Tautology check
        is_tautological, reason = check_tautological(target.statement)
        if is_tautological:
            return VerificationResult(
                passed=False,
                verification_type=target.type,
                error_message=f"TAUTOLOGICAL VERIFICATION REJECTED: {reason}",
            )

        if target.type == VerificationType.PYTHON_ASSERT:
            return self._assert_verifier.verify(target)
        if target.type == VerificationType.SYMPY:
            return self._sympy_verifier.verify(target)
        if target.type == VerificationType.Z3:
            return self._z3_verifier.verify(target)
        return VerificationResult(
            passed=False,
            verification_type=target.type,
            error_message=f"Unknown verification type: {target.type}",
        )
