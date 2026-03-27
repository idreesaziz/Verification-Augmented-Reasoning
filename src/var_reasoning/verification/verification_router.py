"""Routes verification targets to the appropriate verifier."""

from __future__ import annotations

import logging

from var_reasoning.models.schemas import (
    ReasoningPattern,
    VerificationTarget,
    VerificationType,
)
from var_reasoning.models.state import VerificationResult
from var_reasoning.sandbox.executor import CodeExecutor
from var_reasoning.verification.assert_verifier import AssertVerifier
from var_reasoning.verification.sympy_verifier import SympyVerifier
from var_reasoning.verification.tautology_check import check_tautological
from var_reasoning.verification.z3_verifier import Z3Verifier

logger = logging.getLogger(__name__)

# Fix 4: Pattern → required verification type mapping.
# If a reasoning pattern is listed here, it MUST use the specified vtype.
# If the model picks a mismatched vtype, verification is rejected.
PATTERN_REQUIRED_VTYPE: dict[ReasoningPattern, VerificationType] = {
    ReasoningPattern.ALGEBRAIC: VerificationType.SYMPY,
    ReasoningPattern.UNIVERSAL_CLAIM: VerificationType.Z3,
}


class VerificationRouter:
    def __init__(self, executor: CodeExecutor) -> None:
        self._assert_verifier = AssertVerifier(executor)
        self._sympy_verifier = SympyVerifier(executor)
        self._z3_verifier = Z3Verifier(executor)

    def verify(
        self,
        target: VerificationTarget,
        reasoning_pattern: ReasoningPattern | None = None,
    ) -> VerificationResult:
        if target.type == VerificationType.INFORMAL:
            return VerificationResult(
                passed=True,
                verification_type=VerificationType.INFORMAL,
            )

        # Fix 4: Enforce pattern-vtype consistency
        if reasoning_pattern and reasoning_pattern in PATTERN_REQUIRED_VTYPE:
            required = PATTERN_REQUIRED_VTYPE[reasoning_pattern]
            if target.type != required:
                msg = (
                    f"Reasoning pattern '{reasoning_pattern.value}' requires "
                    f"verification type '{required.value}', but got "
                    f"'{target.type.value}'. Change the verification to use "
                    f"{required.value}, or change the reasoning pattern."
                )
                logger.info("Pattern-vtype mismatch: %s", msg)
                return VerificationResult(
                    passed=False,
                    verification_type=target.type,
                    error_message=msg,
                )

        # Fix 3: Tautology check — reject verification code that does no real work
        is_tautological, reason = check_tautological(target.statement)
        if is_tautological:
            logger.info("Tautological verification detected: %s", reason)
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
