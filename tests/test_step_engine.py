"""Tests for the step engine — mocked LLM and executor."""

from unittest.mock import MagicMock, patch

import pytest

from var_reasoning.engine.backtracking import BacktrackManager
from var_reasoning.engine.step_engine import ConditionBEngine, StepEngine
from var_reasoning.models.gemini_provider import GeminiProvider, TokenUsage
from var_reasoning.models.schemas import (
    FinalAnswer,
    InferenceStep,
    ReasoningPattern,
    ReasoningStep,
    StepOutput,
    VerificationTarget,
    VerificationType,
)
from var_reasoning.models.state import VerificationResult
from var_reasoning.sandbox.executor import CodeExecutor
from var_reasoning.verification.verification_router import VerificationRouter


def _usage():
    return TokenUsage(input_tokens=100, output_tokens=50)


@pytest.fixture
def mock_gemini():
    g = MagicMock(spec=GeminiProvider)
    g.reset_usage = MagicMock()
    return g


@pytest.fixture
def mock_executor():
    e = MagicMock(spec=CodeExecutor)
    e.reset_namespace = MagicMock()
    return e


@pytest.fixture
def mock_router():
    return MagicMock(spec=VerificationRouter)


class TestStepEngine:
    def test_immediate_final_answer(self, mock_gemini, mock_executor, mock_router):
        """LLM immediately returns a final answer on the first call."""
        mock_gemini.generate_reasoning_step.return_value = (
            StepOutput(
                final_answer=FinalAnswer(answer="42", justification="obvious")
            ),
            _usage(),
        )
        engine = StepEngine(mock_gemini, mock_executor, mock_router)
        session = engine.solve("p1", "What is 6*7?")

        assert session.final_answer == "42"
        assert session.justification == "obvious"
        assert session.total_llm_calls == 1
        assert len(session.steps) == 0

    def test_one_step_then_answer(self, mock_gemini, mock_executor, mock_router):
        """One reasoning step, then final answer."""
        # First call: reasoning step
        # Second call: final answer
        mock_gemini.generate_reasoning_step.side_effect = [
            (
                StepOutput(
                    reasoning=ReasoningStep(
                        objective="Compute 6*7",
                        depends_on=[],
                        thought="Compute 6*7",
                        action="print(6*7)",
                        result_variable="product",
                    )
                ),
                _usage(),
            ),
            (
                StepOutput(
                    final_answer=FinalAnswer(answer="42", justification="step 1")
                ),
                _usage(),
            ),
        ]
        mock_executor.execute.return_value = (True, "42")
        mock_gemini.generate_inference.return_value = (
            InferenceStep(
                premises=["P1: 6*7 computed", "P2: result was 42"],
                conclusion="6*7=42",
                reasoning_pattern=ReasoningPattern.ALGEBRAIC,
                verification_target=VerificationTarget(
                    type=VerificationType.PYTHON_ASSERT,
                    statement="assert 6*7 == 42",
                ),
            ),
            _usage(),
        )
        mock_router.verify.return_value = VerificationResult(
            passed=True, verification_type=VerificationType.PYTHON_ASSERT
        )

        engine = StepEngine(mock_gemini, mock_executor, mock_router)
        session = engine.solve("p1", "What is 6*7?")

        assert session.final_answer == "42"
        assert len(session.steps) == 1
        assert session.steps[0].observation == "42"
        assert session.steps[0].conclusion == "6*7=42"

    def test_code_failure_triggers_repair(
        self, mock_gemini, mock_executor, mock_router
    ):
        """Code fails, repair loop fires, then succeeds."""
        mock_gemini.generate_reasoning_step.side_effect = [
            (
                StepOutput(
                    reasoning=ReasoningStep(
                        objective="Compute",
                        depends_on=[],
                        thought="Compute",
                        action="prnt(1)",
                        result_variable="r",
                    )
                ),
                _usage(),
            ),
            (
                StepOutput(
                    final_answer=FinalAnswer(answer="1", justification="done")
                ),
                _usage(),
            ),
        ]
        # First execute fails, repair succeeds
        from var_reasoning.models.schemas import CodeFix

        mock_executor.execute.side_effect = [
            (False, "NameError: prnt"),
            (True, "1"),
        ]
        mock_gemini.generate_code_fix.return_value = (
            CodeFix(fixed_code="print(1)", explanation="typo"),
            _usage(),
        )
        mock_gemini.generate_inference.return_value = (
            InferenceStep(
                premises=["P1: print(1) returned 1"],
                conclusion="result is 1",
                reasoning_pattern=ReasoningPattern.ALGEBRAIC,
                verification_target=VerificationTarget(
                    type=VerificationType.PYTHON_ASSERT,
                    statement="assert True",
                ),
            ),
            _usage(),
        )
        mock_router.verify.return_value = VerificationResult(
            passed=True, verification_type=VerificationType.PYTHON_ASSERT
        )

        engine = StepEngine(mock_gemini, mock_executor, mock_router)
        session = engine.solve("p1", "test")

        assert session.final_answer == "1"
        assert session.total_code_retries == 1

    def test_verification_failure_triggers_backtrack(
        self, mock_gemini, mock_executor, mock_router
    ):
        """Verification fails, inference retries exhaust, then backtrack."""
        # First step: reasoning
        mock_gemini.generate_reasoning_step.side_effect = [
            (
                StepOutput(
                    reasoning=ReasoningStep(
                        objective="t",
                        depends_on=[],
                        thought="t",
                        action="print(1)",
                        result_variable="r",
                    )
                ),
                _usage(),
            ),
            (
                StepOutput(
                    final_answer=FinalAnswer(answer="1", justification="done")
                ),
                _usage(),
            ),
        ]
        mock_executor.execute.return_value = (True, "1")
        mock_gemini.generate_inference.return_value = (
            InferenceStep(
                premises=["P1: observed 1"],
                conclusion="wrong",
                reasoning_pattern=ReasoningPattern.ALGEBRAIC,
                verification_target=VerificationTarget(
                    type=VerificationType.PYTHON_ASSERT,
                    statement="assert False",
                ),
            ),
            _usage(),
        )
        mock_router.verify.return_value = VerificationResult(
            passed=False,
            verification_type=VerificationType.PYTHON_ASSERT,
            error_message="AssertionError",
        )
        # Inference revision always fails
        from var_reasoning.models.schemas import InferenceRevision

        mock_gemini.generate_inference_revision.return_value = (
            InferenceRevision(
                choice="revise",
                revised_conclusion="still wrong",
                revised_verification_target=VerificationTarget(
                    type=VerificationType.PYTHON_ASSERT,
                    statement="assert False",
                ),
            ),
            _usage(),
        )

        engine = StepEngine(
            mock_gemini,
            mock_executor,
            mock_router,
            backtrack_manager=BacktrackManager(
                inference_retries=2, total_backtrack_limit=1
            ),
        )
        session = engine.solve("p1", "test")

        assert session.total_backtracks >= 1
        assert session.total_inference_retries >= 1


class TestConditionBEngine:
    def test_basic_solve(self, mock_gemini, mock_executor):
        mock_gemini.generate_reasoning_step.side_effect = [
            (
                StepOutput(
                    reasoning=ReasoningStep(
                        objective="t",
                        depends_on=[],
                        thought="t",
                        action="print(1)",
                        result_variable="r",
                    )
                ),
                _usage(),
            ),
            (
                StepOutput(
                    final_answer=FinalAnswer(answer="1", justification="done")
                ),
                _usage(),
            ),
        ]
        mock_executor.execute.return_value = (True, "1")
        mock_gemini.generate_inference.return_value = (
            InferenceStep(
                premises=["P1: observed 1"],
                conclusion="result is 1",
                reasoning_pattern=ReasoningPattern.ALGEBRAIC,
                verification_target=VerificationTarget(
                    type=VerificationType.PYTHON_ASSERT,
                    statement="assert True",
                ),
            ),
            _usage(),
        )

        engine = ConditionBEngine(mock_gemini, mock_executor)
        session = engine.solve("p1", "test")

        assert session.final_answer == "1"
        assert len(session.steps) == 1
        # No verification was actually run (accepted directly)
        assert session.steps[0].verification_result.passed is True
