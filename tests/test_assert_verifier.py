"""Tests for the assert verifier."""

from unittest.mock import MagicMock

import pytest

from var_reasoning.models.schemas import VerificationTarget, VerificationType
from var_reasoning.verification.assert_verifier import AssertVerifier


@pytest.fixture
def mock_executor():
    return MagicMock()


@pytest.fixture
def verifier(mock_executor):
    return AssertVerifier(mock_executor)


class TestAssertVerifier:
    def test_passing_assert(self, verifier, mock_executor):
        mock_executor.execute.return_value = (True, "")
        target = VerificationTarget(
            type=VerificationType.PYTHON_ASSERT,
            statement="assert 2 + 2 == 4",
        )
        result = verifier.verify(target)
        assert result.passed is True
        assert result.verification_type == VerificationType.PYTHON_ASSERT
        assert result.error_message is None

    def test_failing_assert(self, verifier, mock_executor):
        mock_executor.execute.return_value = (
            False,
            "Traceback (most recent call last):\n  File ...\nAssertionError",
        )
        target = VerificationTarget(
            type=VerificationType.PYTHON_ASSERT,
            statement="assert 2 + 2 == 5",
        )
        result = verifier.verify(target)
        assert result.passed is False
        assert "AssertionError" in result.error_message

    def test_failing_assert_with_message(self, verifier, mock_executor):
        mock_executor.execute.return_value = (
            False,
            "AssertionError: Expected 5 but got 4",
        )
        target = VerificationTarget(
            type=VerificationType.PYTHON_ASSERT,
            statement='assert 2 + 2 == 5, "Expected 5 but got 4"',
        )
        result = verifier.verify(target)
        assert result.passed is False
        assert "Expected 5 but got 4" in result.error_message

    def test_runtime_error(self, verifier, mock_executor):
        mock_executor.execute.return_value = (
            False,
            "NameError: name 'x' is not defined",
        )
        target = VerificationTarget(
            type=VerificationType.PYTHON_ASSERT,
            statement="assert x == 5",
        )
        result = verifier.verify(target)
        assert result.passed is False
        assert "NameError" in result.error_message

    def test_multiple_asserts_all_pass(self, verifier, mock_executor):
        mock_executor.execute.return_value = (True, "")
        target = VerificationTarget(
            type=VerificationType.PYTHON_ASSERT,
            statement="assert 1 == 1\nassert 2 == 2\nassert 3 == 3",
        )
        result = verifier.verify(target)
        assert result.passed is True
