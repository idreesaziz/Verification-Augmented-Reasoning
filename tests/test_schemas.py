"""Tests for Pydantic schema models."""

import pytest
from pydantic import ValidationError

from var_reasoning.models.schemas import (
    CodeFix,
    FinalAnswer,
    InferenceRevision,
    InferenceStep,
    ReasoningStep,
    StepOutput,
    VerificationTarget,
    VerificationType,
)


class TestVerificationType:
    def test_enum_values(self):
        assert VerificationType.Z3 == "z3"
        assert VerificationType.SYMPY == "sympy"
        assert VerificationType.PYTHON_ASSERT == "python_assert"
        assert VerificationType.INFORMAL == "informal"


class TestVerificationTarget:
    def test_valid_target(self):
        target = VerificationTarget(
            type=VerificationType.PYTHON_ASSERT,
            statement="assert 1 + 1 == 2",
        )
        assert target.type == VerificationType.PYTHON_ASSERT
        assert target.statement == "assert 1 + 1 == 2"
        assert target.premises is None

    def test_with_premises(self):
        target = VerificationTarget(
            type=VerificationType.Z3,
            statement="from z3 import *; ...",
            premises=["step_1", "step_2"],
        )
        assert target.premises == ["step_1", "step_2"]

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            VerificationTarget(type=VerificationType.Z3)  # missing statement

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            VerificationTarget(type="invalid", statement="x")


class TestReasoningStep:
    def test_valid(self):
        step = ReasoningStep(
            thought="I need to compute X",
            action="print(1+1)",
        )
        assert step.thought == "I need to compute X"
        assert step.action == "print(1+1)"

    def test_missing_fields(self):
        with pytest.raises(ValidationError):
            ReasoningStep(thought="test")  # missing action


class TestInferenceStep:
    def test_valid(self):
        step = InferenceStep(
            inference="The result is 2",
            verification_target=VerificationTarget(
                type=VerificationType.PYTHON_ASSERT,
                statement="assert result == 2",
            ),
        )
        assert step.inference == "The result is 2"
        assert step.verification_target.type == VerificationType.PYTHON_ASSERT


class TestFinalAnswer:
    def test_valid(self):
        fa = FinalAnswer(answer="42", justification="See steps 1-3")
        assert fa.answer == "42"
        assert fa.justification == "See steps 1-3"


class TestStepOutput:
    def test_with_reasoning(self):
        output = StepOutput(
            reasoning=ReasoningStep(thought="think", action="code"),
        )
        assert output.reasoning is not None
        assert output.final_answer is None

    def test_with_final_answer(self):
        output = StepOutput(
            final_answer=FinalAnswer(answer="42", justification="done"),
        )
        assert output.reasoning is None
        assert output.final_answer is not None

    def test_empty_is_valid(self):
        output = StepOutput()
        assert output.reasoning is None
        assert output.final_answer is None


class TestCodeFix:
    def test_valid(self):
        fix = CodeFix(fixed_code="print(1)", explanation="Fixed syntax")
        assert fix.fixed_code == "print(1)"


class TestInferenceRevision:
    def test_revise(self):
        rev = InferenceRevision(
            choice="revise",
            revised_inference="Updated conclusion",
            revised_verification_target=VerificationTarget(
                type=VerificationType.PYTHON_ASSERT,
                statement="assert True",
            ),
        )
        assert rev.choice == "revise"

    def test_investigate(self):
        rev = InferenceRevision(
            choice="investigate",
            thought="Need more data",
            action="print(more_data)",
        )
        assert rev.choice == "investigate"

    def test_invalid_choice(self):
        with pytest.raises(ValidationError):
            InferenceRevision(choice="invalid")
