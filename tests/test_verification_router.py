"""Tests for the verification router."""

from unittest.mock import MagicMock

import pytest

from var_reasoning.models.schemas import VerificationTarget, VerificationType
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
            statement="assert 1 == 1",
        )
        result = router.verify(target)
        assert result.passed is True
        assert result.verification_type == VerificationType.PYTHON_ASSERT
        mock_executor.execute.assert_called_once_with("assert 1 == 1")

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
            statement="from z3 import *; ...",
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
            "AssertionError: assert 1 == 2",
        )
        target = VerificationTarget(
            type=VerificationType.PYTHON_ASSERT,
            statement="assert 1 == 2",
        )
        result = router.verify(target)
        assert result.passed is False
        assert "AssertionError" in result.error_message
