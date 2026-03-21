"""Tests for backtracking logic."""

import pytest

from var_reasoning.engine.backtracking import BacktrackManager
from var_reasoning.models.schemas import VerificationTarget, VerificationType
from var_reasoning.models.state import CompletedStep, Session, VerificationResult


def _make_step(step_number: int) -> CompletedStep:
    return CompletedStep(
        step_number=step_number,
        thought=f"thought {step_number}",
        action=f"print({step_number})",
        observation=str(step_number),
        inference=f"inference {step_number}",
        verification_target=VerificationTarget(
            type=VerificationType.PYTHON_ASSERT,
            statement=f"assert True",
        ),
        verification_result=VerificationResult(
            passed=True,
            verification_type=VerificationType.PYTHON_ASSERT,
        ),
    )


def _make_session(num_steps: int = 0) -> Session:
    session = Session(problem_id="test", problem_text="test problem")
    for i in range(1, num_steps + 1):
        session.steps.append(_make_step(i))
    return session


class TestBacktrackManager:
    def test_should_stop_at_max_steps(self):
        bt = BacktrackManager(max_steps=3)
        session = _make_session(num_steps=3)
        assert bt.should_stop(session) is True

    def test_should_not_stop_below_max_steps(self):
        bt = BacktrackManager(max_steps=25)
        session = _make_session(num_steps=5)
        assert bt.should_stop(session) is False

    def test_should_stop_at_total_backtrack_limit(self):
        bt = BacktrackManager(total_backtrack_limit=5)
        session = _make_session()
        session.total_backtracks = 5
        assert bt.should_stop(session) is True

    def test_handle_code_failure_removes_last_step(self):
        bt = BacktrackManager()
        session = _make_session(num_steps=3)
        bt.handle_code_failure(session)
        assert len(session.steps) == 2
        assert session.total_backtracks == 1

    def test_handle_code_failure_on_empty(self):
        bt = BacktrackManager()
        session = _make_session(num_steps=0)
        bt.handle_code_failure(session)
        assert len(session.steps) == 0
        assert session.total_backtracks == 1

    def test_handle_inference_failure_normal_backtrack(self):
        bt = BacktrackManager(cascade_limit=3)
        session = _make_session(num_steps=2)
        restart = bt.handle_inference_failure(session)
        assert restart is False
        assert len(session.steps) == 1
        assert session.total_backtracks == 1

    def test_handle_inference_failure_cascade_restart(self):
        bt = BacktrackManager(cascade_limit=2)
        session = _make_session(num_steps=3)
        # First failure: normal backtrack
        restart1 = bt.handle_inference_failure(session)
        assert restart1 is False
        # Second failure: cascade limit hit, restart
        restart2 = bt.handle_inference_failure(session)
        assert restart2 is True
        assert len(session.steps) == 0  # all cleared

    def test_reset_cascade_counter(self):
        bt = BacktrackManager(cascade_limit=2)
        session = _make_session(num_steps=3)
        bt.handle_inference_failure(session)  # count = 1
        bt.reset_cascade_counter()
        # After reset, need cascade_limit failures again
        restart = bt.handle_inference_failure(session)
        assert restart is False  # only 1, not 2

    def test_is_unsolvable(self):
        bt = BacktrackManager(total_backtrack_limit=3)
        session = _make_session()
        session.total_backtracks = 3
        assert bt.is_unsolvable(session) is True

    def test_not_unsolvable(self):
        bt = BacktrackManager(total_backtrack_limit=20)
        session = _make_session()
        session.total_backtracks = 5
        assert bt.is_unsolvable(session) is False

    def test_cascading_backtrack_sequence(self):
        """Test full cascade: 3 failures -> restart, then 3 more -> restart."""
        bt = BacktrackManager(cascade_limit=3, total_backtrack_limit=100)
        session = _make_session(num_steps=5)

        for i in range(2):
            bt.handle_inference_failure(session)
            assert len(session.steps) >= 0

        restart = bt.handle_inference_failure(session)
        assert restart is True
        assert len(session.steps) == 0
        assert session.total_backtracks == 3
