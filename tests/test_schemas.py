"""Tests for Pydantic schema models."""

import pytest
from pydantic import ValidationError

from var_reasoning.models.schemas import (
    CodeFix,
    FinalAnswer,
    InferenceRevision,
    InferenceStep,
    ReasoningPattern,
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
        assert target.informal_reason is None

    def test_with_informal_reason(self):
        target = VerificationTarget(
            type=VerificationType.INFORMAL,
            statement="",
            informal_reason="Cannot formalize semantic similarity",
        )
        assert target.informal_reason == "Cannot formalize semantic similarity"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            VerificationTarget(type=VerificationType.Z3)  # missing statement

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            VerificationTarget(type="invalid", statement="x")


class TestReasoningPattern:
    def test_enum_values(self):
        assert ReasoningPattern.EXHAUSTIVE_ENUMERATION == "exhaustive_enumeration"
        assert ReasoningPattern.PRODUCT_RULE == "product_rule"
        assert ReasoningPattern.UNIVERSAL_CLAIM == "universal_claim"
        assert ReasoningPattern.EXISTENTIAL == "existential"
        assert ReasoningPattern.ALGEBRAIC == "algebraic"
        assert ReasoningPattern.CASE_ANALYSIS == "case_analysis"


class TestReasoningStep:
    def test_valid(self):
        step = ReasoningStep(
            objective="Compute X",
            depends_on=[],
            thought="I need to compute X",
            action="print(1+1)",
            result_variable="x",
        )
        assert step.objective == "Compute X"
        assert step.depends_on == []
        assert step.thought == "I need to compute X"
        assert step.action == "print(1+1)"
        assert step.result_variable == "x"

    def test_missing_fields(self):
        with pytest.raises(ValidationError):
            ReasoningStep(thought="test")  # missing action and others


class TestInferenceStep:
    def test_valid(self):
        step = InferenceStep(
            premises=["P1: computed 6*7", "P2: result was 42"],
            conclusion="6*7 equals 42",
            reasoning_pattern=ReasoningPattern.ALGEBRAIC,
            verification_target=VerificationTarget(
                type=VerificationType.PYTHON_ASSERT,
                statement="assert 6*7 == 42",
            ),
        )
        assert step.premises == ["P1: computed 6*7", "P2: result was 42"]
        assert step.conclusion == "6*7 equals 42"
        assert step.reasoning_pattern == ReasoningPattern.ALGEBRAIC
        assert step.verification_target.type == VerificationType.PYTHON_ASSERT

    def test_missing_premises_fails(self):
        with pytest.raises(ValidationError):
            InferenceStep(
                conclusion="6*7 equals 42",
                reasoning_pattern=ReasoningPattern.ALGEBRAIC,
                verification_target=VerificationTarget(
                    type=VerificationType.PYTHON_ASSERT,
                    statement="assert 6*7 == 42",
                ),
            )

    def test_invalid_pattern_fails(self):
        with pytest.raises(ValidationError):
            InferenceStep(
                premises=["P1"],
                conclusion="test",
                reasoning_pattern="made_up_pattern",
                verification_target=VerificationTarget(
                    type=VerificationType.PYTHON_ASSERT,
                    statement="assert True",
                ),
            )


class TestFinalAnswer:
    def test_valid(self):
        fa = FinalAnswer(answer="42", justification="See steps 1-3")
        assert fa.answer == "42"
        assert fa.justification == "See steps 1-3"


class TestStepOutput:
    def test_with_reasoning(self):
        output = StepOutput(
            reasoning=ReasoningStep(
                objective="think",
                depends_on=[],
                thought="think",
                action="code",
                result_variable="r",
            ),
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
            revised_premises=["P1: updated"],
            revised_conclusion="Updated conclusion",
            revised_reasoning_pattern=ReasoningPattern.ALGEBRAIC,
            revised_verification_target=VerificationTarget(
                type=VerificationType.PYTHON_ASSERT,
                statement="assert True",
            ),
        )
        assert rev.choice == "revise"
        assert rev.revised_premises == ["P1: updated"]
        assert rev.revised_reasoning_pattern == ReasoningPattern.ALGEBRAIC

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
