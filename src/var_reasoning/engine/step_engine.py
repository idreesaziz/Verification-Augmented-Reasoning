"""Core reasoning loop: the step engine."""

from __future__ import annotations

import logging

from var_reasoning.engine.backtracking import BacktrackManager
from var_reasoning.engine.feedback import (
    build_conversation_history,
    build_inference_context,
    make_code_repair_prompt,
    make_inference_retry_prompt,
)
from var_reasoning.models.gemini_provider import GeminiProvider
from var_reasoning.models.schemas import VerificationType
from var_reasoning.models.state import CompletedStep, Session, VerificationResult
from var_reasoning.prompts.inference_prompt import INFERENCE_PROMPT
from var_reasoning.prompts.reasoning_prompt import REASONING_PROMPT
from var_reasoning.sandbox.executor import CodeExecutor
from var_reasoning.verification.verification_router import VerificationRouter

logger = logging.getLogger(__name__)


class StepEngine:
    """Runs the VAR reasoning loop on a single problem."""

    def __init__(
        self,
        gemini: GeminiProvider,
        executor: CodeExecutor,
        router: VerificationRouter,
        backtrack_manager: BacktrackManager | None = None,
    ) -> None:
        self._gemini = gemini
        self._executor = executor
        self._router = router
        self._bt = backtrack_manager or BacktrackManager()

    def _track_call(self, session: Session, usage) -> None:
        session.total_llm_calls += 1
        session.total_input_tokens += usage.input_tokens
        session.total_output_tokens += usage.output_tokens

    def _execute_with_repair(
        self,
        code: str,
        thought: str,
        session: Session,
    ) -> tuple[bool, str]:
        """Execute code with silent repair loop (Mode A)."""
        success, output = self._executor.execute(code)
        if success:
            return True, output

        prior_attempts: list[tuple[str, str]] = []
        current_code = code
        current_error = output

        for _ in range(self._bt.code_retries):
            session.total_code_retries += 1
            prompt = make_code_repair_prompt(
                thought, current_code, current_error, prior_attempts
            )
            code_fix, usage = self._gemini.generate_code_fix(prompt)
            self._track_call(session, usage)

            prior_attempts.append((current_code, current_error))
            current_code = code_fix.fixed_code

            success, output = self._executor.execute(current_code)
            if success:
                return True, output
            current_error = output

        return False, current_error

    def _retry_inference(
        self,
        session: Session,
        thought: str,
        action: str,
        observation: str,
        inference: str,
        verification_type: str,
        failure_details: str,
    ) -> CompletedStep | None:
        """Inference retry loop (Mode B). Returns completed step or None."""
        prior_attempts: list[tuple[str, str, str]] = []
        current_inference = inference
        current_vtype = verification_type
        current_failure = failure_details

        for _ in range(self._bt.inference_retries):
            session.total_inference_retries += 1
            prompt = make_inference_retry_prompt(
                session,
                observation,
                current_inference,
                current_vtype,
                current_failure,
                prior_attempts,
            )
            revision, usage = self._gemini.generate_inference_revision(prompt)
            self._track_call(session, usage)

            if revision.choice == "investigate" and revision.thought and revision.action:
                # LLM wants to run new code instead of revising
                success, new_obs = self._execute_with_repair(
                    revision.action, revision.thought, session
                )
                if not success:
                    continue
                # Generate new inference for the new observation
                ctx = build_inference_context(
                    session, revision.thought, revision.action, new_obs
                )
                inf_step, inf_usage = self._gemini.generate_inference(
                    INFERENCE_PROMPT, ctx
                )
                self._track_call(session, inf_usage)
                result = self._router.verify(inf_step.verification_target)
                vtype_key = inf_step.verification_target.type.value
                session.verification_type_counts[vtype_key] = (
                    session.verification_type_counts.get(vtype_key, 0) + 1
                )
                if inf_step.verification_target.type == VerificationType.INFORMAL:
                    session.informal_skip_count += 1
                if result.passed or inf_step.verification_target.type == VerificationType.INFORMAL:
                    self._bt.reset_cascade_counter()
                    return CompletedStep(
                        step_number=len(session.steps) + 1,
                        thought=revision.thought,
                        action=revision.action,
                        observation=new_obs,
                        inference=inf_step.inference,
                        verification_target=inf_step.verification_target,
                        verification_result=result,
                    )
                current_inference = inf_step.inference
                current_vtype = inf_step.verification_target.type.value
                current_failure = result.error_message or "Verification failed"
                prior_attempts.append(
                    (current_inference, current_vtype, current_failure)
                )
            elif revision.choice == "revise" and revision.revised_inference and revision.revised_verification_target:
                prior_attempts.append(
                    (current_inference, current_vtype, current_failure)
                )
                current_inference = revision.revised_inference
                target = revision.revised_verification_target
                result = self._router.verify(target)
                vtype_key = target.type.value
                session.verification_type_counts[vtype_key] = (
                    session.verification_type_counts.get(vtype_key, 0) + 1
                )
                if target.type == VerificationType.INFORMAL:
                    session.informal_skip_count += 1
                if result.passed or target.type == VerificationType.INFORMAL:
                    self._bt.reset_cascade_counter()
                    return CompletedStep(
                        step_number=len(session.steps) + 1,
                        thought=thought,
                        action=action,
                        observation=observation,
                        inference=revision.revised_inference,
                        verification_target=target,
                        verification_result=result,
                    )
                current_vtype = target.type.value
                current_failure = result.error_message or "Verification failed"
            else:
                # Invalid revision — count it and continue
                prior_attempts.append(
                    (current_inference, current_vtype, current_failure)
                )

        return None

    def solve(self, problem_id: str, problem_text: str) -> Session:
        """Run the full VAR loop on a problem."""
        session = Session(problem_id=problem_id, problem_text=problem_text)
        self._executor.reset_namespace()
        self._gemini.reset_usage()

        while not self._bt.should_stop(session):
            # Phase 1: Generate reasoning step
            conversation = build_conversation_history(session)
            step_output, usage = self._gemini.generate_reasoning_step(
                REASONING_PROMPT, conversation
            )
            self._track_call(session, usage)

            # Check for final answer
            if step_output.final_answer:
                session.final_answer = step_output.final_answer.answer
                session.justification = step_output.final_answer.justification
                break

            if not step_output.reasoning:
                # Malformed output — treat as step failure, backtrack
                self._bt.handle_code_failure(session)
                continue

            thought = step_output.reasoning.thought
            action = step_output.reasoning.action

            # Phase 2: Execute code (with silent repair loop)
            success, observation = self._execute_with_repair(
                action, thought, session
            )
            if not success:
                self._bt.handle_code_failure(session)
                continue

            # Phase 3: Generate inference + verification target
            ctx = build_inference_context(session, thought, action, observation)
            inference_step, inf_usage = self._gemini.generate_inference(
                INFERENCE_PROMPT, ctx
            )
            self._track_call(session, inf_usage)

            # Phase 4: Verify
            result = self._router.verify(inference_step.verification_target)
            vtype_key = inference_step.verification_target.type.value
            session.verification_type_counts[vtype_key] = (
                session.verification_type_counts.get(vtype_key, 0) + 1
            )

            if inference_step.verification_target.type == VerificationType.INFORMAL:
                session.informal_skip_count += 1

            if result.passed or inference_step.verification_target.type == VerificationType.INFORMAL:
                # Step accepted
                self._bt.reset_cascade_counter()
                session.steps.append(
                    CompletedStep(
                        step_number=len(session.steps) + 1,
                        thought=thought,
                        action=action,
                        observation=observation,
                        inference=inference_step.inference,
                        verification_target=inference_step.verification_target,
                        verification_result=result,
                    )
                )
            else:
                # Inference retry loop (Mode B)
                failure_details = result.error_message or "Verification failed"
                if result.counterexample:
                    failure_details += f"\nCounterexample: {result.counterexample}"

                revised = self._retry_inference(
                    session,
                    thought,
                    action,
                    observation,
                    inference_step.inference,
                    vtype_key,
                    failure_details,
                )
                if revised:
                    session.steps.append(revised)
                else:
                    # Mode C: cascade backtrack
                    restart = self._bt.handle_inference_failure(session)
                    if restart:
                        self._executor.reset_namespace()
                        logger.info(
                            "Cascade limit reached for %s. Restarting.",
                            problem_id,
                        )

        if self._bt.is_unsolvable(session) and session.final_answer is None:
            session.final_answer = "UNSOLVABLE"
            session.justification = "Exceeded backtrack limits."

        return session


class ConditionBEngine:
    """Condition B: code execution + reasoning loop, NO verification."""

    def __init__(
        self,
        gemini: GeminiProvider,
        executor: CodeExecutor,
    ) -> None:
        self._gemini = gemini
        self._executor = executor
        self._max_steps = 25

    def _track_call(self, session: Session, usage) -> None:
        session.total_llm_calls += 1
        session.total_input_tokens += usage.input_tokens
        session.total_output_tokens += usage.output_tokens

    def solve(self, problem_id: str, problem_text: str) -> Session:
        session = Session(problem_id=problem_id, problem_text=problem_text)
        self._executor.reset_namespace()
        self._gemini.reset_usage()

        for _ in range(self._max_steps):
            conversation = build_conversation_history(session)
            step_output, usage = self._gemini.generate_reasoning_step(
                REASONING_PROMPT, conversation
            )
            self._track_call(session, usage)

            if step_output.final_answer:
                session.final_answer = step_output.final_answer.answer
                session.justification = step_output.final_answer.justification
                break

            if not step_output.reasoning:
                continue

            thought = step_output.reasoning.thought
            action = step_output.reasoning.action

            success, observation = self._executor.execute(action)
            if not success:
                observation = f"[Code error]: {observation}"

            # Generate inference (no verification)
            ctx = build_inference_context(session, thought, action, observation)
            inference_step, inf_usage = self._gemini.generate_inference(
                INFERENCE_PROMPT, ctx
            )
            self._track_call(session, inf_usage)

            # Accept every inference without verification
            session.steps.append(
                CompletedStep(
                    step_number=len(session.steps) + 1,
                    thought=thought,
                    action=action,
                    observation=observation,
                    inference=inference_step.inference,
                    verification_target=inference_step.verification_target,
                    verification_result=VerificationResult(
                        passed=True,
                        verification_type=inference_step.verification_target.type,
                    ),
                )
            )

        return session
