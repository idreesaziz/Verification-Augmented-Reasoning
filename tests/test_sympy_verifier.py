"""Tests for the SymPy verifier."""

from unittest.mock import MagicMock

import pytest

from var_reasoning.models.schemas import VerificationTarget, VerificationType
from var_reasoning.verification.sympy_verifier import SympyVerifier


@pytest.fixture
def mock_executor():
    return MagicMock()


@pytest.fixture
def verifier(mock_executor):
    return SympyVerifier(mock_executor)


class TestSympyVerifier:
    def test_correct_sympy_check(self, verifier, mock_executor):
        mock_executor.execute.return_value = (True, "")
        target = VerificationTarget(
            type=VerificationType.SYMPY,
            statement=(
                "from sympy import symbols, simplify\n"
                "x = symbols('x')\n"
                "assert simplify((x+1)**2 - (x**2 + 2*x + 1)) == 0"
            ),
        )
        result = verifier.verify(target)
        assert result.passed is True
        assert result.verification_type == VerificationType.SYMPY

    def test_incorrect_sympy_check(self, verifier, mock_executor):
        mock_executor.execute.return_value = (
            False,
            "AssertionError: simplify result was not zero",
        )
        target = VerificationTarget(
            type=VerificationType.SYMPY,
            statement=(
                "from sympy import symbols, simplify\n"
                "x = symbols('x')\n"
                "assert simplify((x+1)**2 - (x**2 + 3*x + 1)) == 0"
            ),
        )
        result = verifier.verify(target)
        assert result.passed is False
        assert result.error_message is not None

    def test_sympy_import_error(self, verifier, mock_executor):
        mock_executor.execute.return_value = (
            False,
            "ModuleNotFoundError: No module named 'nonexistent'",
        )
        target = VerificationTarget(
            type=VerificationType.SYMPY,
            statement="import nonexistent",
        )
        result = verifier.verify(target)
        assert result.passed is False

    def test_sympy_solve_verification(self, verifier, mock_executor):
        mock_executor.execute.return_value = (True, "")
        target = VerificationTarget(
            type=VerificationType.SYMPY,
            statement=(
                "from sympy import symbols, solve\n"
                "x = symbols('x')\n"
                "solutions = solve(x**2 - 4, x)\n"
                "assert set(solutions) == {-2, 2}"
            ),
        )
        result = verifier.verify(target)
        assert result.passed is True
