"""Tests for the verification router."""

from unittest.mock import MagicMock

import pytest

from var_reasoning.models.schemas import (
    ReasoningPattern,
    VerificationTarget,
    VerificationType,
)
from var_reasoning.models.state import VerificationResult
from var_reasoning.verification.verification_router import VerificationRouter


@pytest.fixture
def mock_executor():
    return MagicMock()


@pytest.fixture
def router(mock_executor):
    return VerificationRouter(mock_executor)


class TestVerificationRouter:
    def test_routes_to_assert_verifier(self, router, mock_executor):
        mock_executor.execute.return_value = (True, "")
        target = VerificationTarget(
            type=VerificationType.PYTHON_ASSERT,
            statement="assert total == 42",
        )
        result = router.verify(target)
        assert result.passed is True
        assert result.verification_type == VerificationType.PYTHON_ASSERT
        mock_executor.execute.assert_called_once_with("assert total == 42")

    def test_routes_to_sympy_verifier(self, router, mock_executor):
        mock_executor.execute.return_value = (True, "")
        target = VerificationTarget(
            type=VerificationType.SYMPY,
            statement="from sympy import *; assert simplify(x**2 - x**2) == 0",
        )
        result = router.verify(target)
        assert result.passed is True
        assert result.verification_type == VerificationType.SYMPY

    def test_routes_to_z3_verifier(self, router, mock_executor):
        mock_executor.execute.return_value = (True, "")
        target = VerificationTarget(
            type=VerificationType.Z3,
            statement="from z3 import *; x = Int('x')",
        )
        result = router.verify(target)
        assert result.passed is True
        assert result.verification_type == VerificationType.Z3

    def test_informal_always_passes(self, router, mock_executor):
        target = VerificationTarget(
            type=VerificationType.INFORMAL,
            statement="This is a qualitative claim",
        )
        result = router.verify(target)
        assert result.passed is True
        assert result.verification_type == VerificationType.INFORMAL
        # Executor should NOT be called for informal
        mock_executor.execute.assert_not_called()

    def test_failed_assert(self, router, mock_executor):
        mock_executor.execute.return_value = (
            False,
            "AssertionError: assert total == 2",
        )
        target = VerificationTarget(
            type=VerificationType.PYTHON_ASSERT,
            statement="assert total == 2",
        )
        result = router.verify(target)
        assert result.passed is False
        assert "AssertionError" in result.error_message

    def test_tautological_verification_rejected(self, router):
        """Verification code with only hardcoded literals is rejected."""
        target = VerificationTarget(
            type=VerificationType.PYTHON_ASSERT,
            statement="assert 6 * 10 * 20 == 1200",
        )
        result = router.verify(target)
        assert result.passed is False
        assert "TAUTOLOGICAL" in result.error_message

    def test_pattern_vtype_mismatch_algebraic(self, router):
        """algebraic pattern must use sympy, not python_assert."""
        target = VerificationTarget(
            type=VerificationType.PYTHON_ASSERT,
            statement="assert total == 42",
        )
        result = router.verify(target, ReasoningPattern.ALGEBRAIC)
        assert result.passed is False
        assert "requires" in result.error_message
        assert "sympy" in result.error_message

    def test_pattern_vtype_mismatch_universal(self, router):
        """universal_claim pattern must use z3, not python_assert."""
        target = VerificationTarget(
            type=VerificationType.PYTHON_ASSERT,
            statement="assert total == 42",
        )
        result = router.verify(target, ReasoningPattern.UNIVERSAL_CLAIM)
        assert result.passed is False
        assert "z3" in result.error_message

    def test_pattern_vtype_match_passes_through(self, router, mock_executor):
        """algebraic + sympy is allowed — passes through to verifier."""
        mock_executor.execute.return_value = (True, "")
        target = VerificationTarget(
            type=VerificationType.SYMPY,
            statement="from sympy import *; assert simplify(x - x) == 0",
        )
        result = router.verify(target, ReasoningPattern.ALGEBRAIC)
        assert result.passed is True

    def test_no_pattern_skips_enforcement(self, router, mock_executor):
        """Without reasoning_pattern, no enforcement happens."""
        mock_executor.execute.return_value = (True, "")
        target = VerificationTarget(
            type=VerificationType.PYTHON_ASSERT,
            statement="assert total == 42",
        )
        result = router.verify(target)
        assert result.passed is True
