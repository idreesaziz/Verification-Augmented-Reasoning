"""Assert-based verifier: runs Python assert statements in the sandbox."""

from __future__ import annotations

from var_reasoning.models.schemas import VerificationTarget, VerificationType
from var_reasoning.models.state import VerificationResult
from var_reasoning.sandbox.executor import CodeExecutor


class AssertVerifier:
    def __init__(self, executor: CodeExecutor) -> None:
        self._executor = executor

    def verify(self, target: VerificationTarget) -> VerificationResult:
        success, output = self._executor.execute(target.statement)
        if success:
            return VerificationResult(
                passed=True,
                verification_type=VerificationType.PYTHON_ASSERT,
            )
        return VerificationResult(
            passed=False,
            verification_type=VerificationType.PYTHON_ASSERT,
            error_message=output,
        )
