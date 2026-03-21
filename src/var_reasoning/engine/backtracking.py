"""Backtracking logic for the reasoning engine."""

from __future__ import annotations

from var_reasoning.models.state import Session


# Limits
MAX_STEPS = 25
CODE_RETRIES_PER_STEP = 3
INFERENCE_RETRIES_PER_STEP = 5
CASCADE_BACKTRACKS = 3
TOTAL_BACKTRACKS = 20


class BacktrackManager:
    """Manages backtracking state and decisions."""

    def __init__(
        self,
        max_steps: int = MAX_STEPS,
        code_retries: int = CODE_RETRIES_PER_STEP,
        inference_retries: int = INFERENCE_RETRIES_PER_STEP,
        cascade_limit: int = CASCADE_BACKTRACKS,
        total_backtrack_limit: int = TOTAL_BACKTRACKS,
    ) -> None:
        self.max_steps = max_steps
        self.code_retries = code_retries
        self.inference_retries = inference_retries
        self.cascade_limit = cascade_limit
        self.total_backtrack_limit = total_backtrack_limit
        self._consecutive_cascade_count = 0

    def should_stop(self, session: Session) -> bool:
        """Check if we've hit hard limits."""
        if len(session.steps) >= self.max_steps:
            return True
        if session.total_backtracks >= self.total_backtrack_limit:
            return True
        return False

    def handle_code_failure(self, session: Session) -> None:
        """Mode A: code retries exhausted — backtrack by removing last step."""
        session.total_backtracks += 1
        if session.steps:
            session.steps.pop()

    def handle_inference_failure(self, session: Session) -> bool:
        """Mode B/C: inference retries exhausted.

        Returns True if we should restart with a different approach
        (cascade limit reached), False if we just backtrack one step.
        """
        session.total_backtracks += 1
        self._consecutive_cascade_count += 1

        if self._consecutive_cascade_count >= self.cascade_limit:
            # Restart: clear all steps and try a completely different approach
            self._consecutive_cascade_count = 0
            session.steps.clear()
            return True  # signal restart

        # Normal cascade backtrack: remove the current step
        if session.steps:
            session.steps.pop()
        return False

    def reset_cascade_counter(self) -> None:
        """Call when a step succeeds to reset the cascade counter."""
        self._consecutive_cascade_count = 0

    def is_unsolvable(self, session: Session) -> bool:
        """Check if we've exhausted all backtrack attempts."""
        return session.total_backtracks >= self.total_backtrack_limit
