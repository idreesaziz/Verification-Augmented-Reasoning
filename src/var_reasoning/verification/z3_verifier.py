"""Z3-based verifier: runs Z3 verification code in the sandbox."""

from __future__ import annotations

from var_reasoning.models.schemas import VerificationTarget, VerificationType
from var_reasoning.models.state import VerificationResult
from var_reasoning.sandbox.executor import CodeExecutor


class Z3Verifier:
    def __init__(self, executor: CodeExecutor) -> None:
        self._executor = executor

    def verify(self, target: VerificationTarget) -> VerificationResult:
        # Z3 verification code is expected to:
        # - Check satisfiability of the NEGATION of a universal claim
        # - If sat => claim is false, model is counterexample
        # - If unsat => claim is true
        # - The code should raise/assert on failure and print counterexample
        success, output = self._executor.execute(target.statement)
        if success:
            return VerificationResult(
                passed=True,
                verification_type=VerificationType.Z3,
            )
        # Extract counterexample from output if present
        counterexample = None
        if output:
            for line in output.splitlines():
                if "counterexample" in line.lower() or line.startswith("["):
                    counterexample = line
                    break
        return VerificationResult(
            passed=False,
            verification_type=VerificationType.Z3,
            error_message=output,
            counterexample=counterexample,
        )
