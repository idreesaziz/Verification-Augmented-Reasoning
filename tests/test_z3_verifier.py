"""Tests for the Z3 verifier."""

from unittest.mock import MagicMock

import pytest

from var_reasoning.models.schemas import VerificationTarget, VerificationType
from var_reasoning.verification.z3_verifier import Z3Verifier


@pytest.fixture
def mock_executor():
    return MagicMock()


@pytest.fixture
def verifier(mock_executor):
    return Z3Verifier(mock_executor)


class TestZ3Verifier:
    def test_valid_universal_claim(self, verifier, mock_executor):
        """Universal claim verified (negation is unsat)."""
        mock_executor.execute.return_value = (True, "")
        target = VerificationTarget(
            type=VerificationType.Z3,
            statement=(
                "from z3 import *\n"
                "x = Int('x')\n"
                "s = Solver()\n"
                "s.add(Not(x + 0 == x))\n"
                "assert s.check() == unsat, 'Claim falsified'"
            ),
        )
        result = verifier.verify(target)
        assert result.passed is True
        assert result.verification_type == VerificationType.Z3

    def test_invalid_universal_claim_with_counterexample(
        self, verifier, mock_executor
    ):
        """Universal claim falsified (negation is sat, counterexample found)."""
        mock_executor.execute.return_value = (
            False,
            "AssertionError: Claim falsified\n[x = 5]",
        )
        target = VerificationTarget(
            type=VerificationType.Z3,
            statement=(
                "from z3 import *\n"
                "x = Int('x')\n"
                "s = Solver()\n"
                "s.add(Not(x * x < 10))\n"
                "result = s.check()\n"
                "if result == sat:\n"
                "    print(f'[x = {s.model()[x]}]')\n"
                "assert result == unsat, 'Claim falsified'"
            ),
        )
        result = verifier.verify(target)
        assert result.passed is False
        assert result.counterexample is not None
        assert "x = 5" in result.counterexample

    def test_z3_syntax_error(self, verifier, mock_executor):
        mock_executor.execute.return_value = (
            False,
            "NameError: name 'Int' is not defined",
        )
        target = VerificationTarget(
            type=VerificationType.Z3,
            statement="x = Int('x')",  # missing import
        )
        result = verifier.verify(target)
        assert result.passed is False
        assert "NameError" in result.error_message

    def test_no_counterexample_in_output(self, verifier, mock_executor):
        """Error without counterexample pattern."""
        mock_executor.execute.return_value = (
            False,
            "AssertionError: check failed",
        )
        target = VerificationTarget(
            type=VerificationType.Z3,
            statement="assert False",
        )
        result = verifier.verify(target)
        assert result.passed is False
        assert result.counterexample is None
