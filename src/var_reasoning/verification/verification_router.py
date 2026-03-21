"""Routes verification targets to the appropriate verifier."""

from __future__ import annotations

from var_reasoning.models.schemas import VerificationTarget, VerificationType
from var_reasoning.models.state import VerificationResult
from var_reasoning.sandbox.executor import CodeExecutor
from var_reasoning.verification.assert_verifier import AssertVerifier
from var_reasoning.verification.sympy_verifier import SympyVerifier
from var_reasoning.verification.z3_verifier import Z3Verifier


class VerificationRouter:
    def __init__(self, executor: CodeExecutor) -> None:
        self._assert_verifier = AssertVerifier(executor)
        self._sympy_verifier = SympyVerifier(executor)
        self._z3_verifier = Z3Verifier(executor)

    def verify(self, target: VerificationTarget) -> VerificationResult:
        if target.type == VerificationType.INFORMAL:
            return VerificationResult(
                passed=True,
                verification_type=VerificationType.INFORMAL,
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
