"""Core reasoning loop — VAR v2.

Architecture: fact-pool + adversarial falsification.

Each step:
  1. Reasoner produces code + optional derivations
  2. Code executes → COMPUTED fact added to pool
  3. Each derivation is independently falsified by the adversary
  4. Survived derivations → DERIVED facts added to pool
  5. Rejected derivation → step discarded, backtrack

No inference layer. No model-authored verification.
"""

from __future__ import annotations

import logging
import re

from var_reasoning.engine.backtracking import BacktrackManager
from var_reasoning.engine.feedback import (
    build_conversation_history,
    build_rejection_feedback,
    make_code_repair_prompt,
)
from var_reasoning.models.gemini_provider import GeminiProvider
from var_reasoning.models.schemas import Verdict
from var_reasoning.models.state import CompletedStep, FalsificationResult, Session
from var_reasoning.prompts.reasoning_prompt import REASONING_PROMPT
from var_reasoning.verification.adversary import falsify_derivation

logger = logging.getLogger(__name__)


def _try_parse_float(s: str) -> float | None:
    """Try to extract a float from code output."""
    s = s.strip()
    # Try direct parse
    try:
        return float(s)
    except ValueError:
        pass
    # Try to find a number in the output
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return None


class StepEngine:
    """Runs the VAR v2 reasoning loop on a single problem."""

    def __init__(
        self,
        gemini: GeminiProvider,
        executor,  # anything with .execute(code) -> (bool, str) and .reset_namespace()
        backtrack_manager: BacktrackManager | None = None,
    ) -> None:
        self._gemini = gemini
        self._executor = executor
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
        """Execute code with silent repair loop."""
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

    def solve(self, problem_id: str, problem_text: str) -> Session:
        """Run the full VAR v2 loop on a problem."""
        session = Session(problem_id=problem_id, problem_text=problem_text)
        self._executor.reset_namespace()
        self._gemini.reset_usage()

        # Rejection feedback carried across retries of the same logical point
        rejection_feedback: str | None = None

        while not self._bt.should_stop(session):
            step_number = len(session.steps) + 1

            # ── Phase 1: Generate reasoning step ────────────────────
            step_output = None
            for _attempt in range(3):
                conversation = build_conversation_history(session)
                if rejection_feedback:
                    conversation.append(rejection_feedback)
                raw_output, usage = self._gemini.generate_reasoning_step(
                    REASONING_PROMPT, conversation
                )
                self._track_call(session, usage)
                if raw_output and (raw_output.final_answer or raw_output.reasoning):
                    step_output = raw_output
                    break

            if step_output is None:
                self._bt.handle_code_failure(session)
                rejection_feedback = None
                continue

            # ── Check for final answer ──────────────────────────────
            if step_output.final_answer:
                session.final_answer = step_output.final_answer.answer
                session.fact_chain = step_output.final_answer.fact_chain
                # Compute compound confidence over the fact chain
                session.compound_confidence = session.fact_pool.compound_confidence(
                    step_output.final_answer.fact_chain
                )
                break

            reasoning = step_output.reasoning

            # ── Phase 2: Execute code ───────────────────────────────
            success, observation = self._execute_with_repair(
                reasoning.action, reasoning.thought, session
            )
            if not success:
                logger.warning(
                    "Step %d code failed after retries: %s",
                    step_number,
                    observation[:200],
                )
                self._bt.handle_code_failure(session)
                rejection_feedback = None
                continue

            # ── Phase 3: Add COMPUTED fact ──────────────────────────
            parsed_value = _try_parse_float(observation)
            computed_fact = session.fact_pool.add_computed(
                statement=f"{reasoning.result_variable} = {observation.strip()[:200]}",
                value=parsed_value,
                step=step_number,
            )

            logger.info(
                "Step %d: %s → %s (COMPUTED: %s)",
                step_number,
                reasoning.objective,
                observation.strip()[:80],
                computed_fact.id,
            )

            # ── Phase 4: Falsify derivations ────────────────────────
            all_survived = True
            falsification_results: list[FalsificationResult] = []
            rejected_feedback = None

            for derivation in reasoning.derivations:
                # Auto-fill claimed_value from computed result if missing
                if derivation.claimed_value is None and parsed_value is not None:
                    derivation.claimed_value = parsed_value

                logger.info(
                    "  Falsifying: %s (claimed=%s)",
                    derivation.premise[:80],
                    derivation.claimed_value,
                )

                result, adv_usage = falsify_derivation(
                    derivation=derivation,
                    problem_text=session.problem_text,
                    fact_pool=session.fact_pool,
                    provider=self._gemini,
                )
                self._track_call(session, adv_usage)
                session.total_adversary_calls += 1
                session.total_falsifications += 1
                falsification_results.append(result)

                logger.info(
                    "    Verdict: %s (e=%.2f) %s",
                    result.verdict.value,
                    result.e_value,
                    result.feedback[:100] if result.feedback else "",
                )

                if result.verdict == Verdict.REJECT:
                    all_survived = False
                    session.total_rejections += 1
                    rejected_feedback = build_rejection_feedback(
                        derivation, result
                    )
                    break

                # Auto cross-check: if the derivation's claimed_value
                # equals the step's parsed_value (meaning the derivation
                # IS about the step's main output) and the adversary's
                # empirical disagrees with both, reject.
                # Also check when derivation claimed_value differs from
                # parsed_value — then compare empirical to claimed_value
                # only when verdict was INCONCLUSIVE (adversary couldn't
                # do the e-value test itself).
                if (
                    result.empirical_value is not None
                    and parsed_value is not None
                    and derivation.claimed_value is not None
                ):
                    # Compare empirical to the derivation's claimed value
                    cv = derivation.claimed_value
                    rel_err = abs(result.empirical_value - cv) / max(abs(cv), 1e-12)
                    # Only cross-check reject on large discrepancies
                    # This catches the case where e-value test said SURVIVE
                    # but the simulation clearly disagrees with the claim
                    if rel_err > 0.10 and result.verdict != Verdict.REJECT:
                        logger.warning(
                            "    Auto cross-check REJECT: claimed=%.6f vs empirical=%.6f (%.1f%% off)",
                            cv, result.empirical_value, rel_err * 100,
                        )
                        result.verdict = Verdict.REJECT
                        result.e_value = float("inf")
                        result.feedback = (
                            f"Cross-check: derivation claimed {cv} but simulation found "
                            f"{result.empirical_value:.6f} ({rel_err*100:.1f}% discrepancy)"
                        )
                        all_survived = False
                        session.total_rejections += 1
                        rejected_feedback = build_rejection_feedback(
                            derivation, result
                        )
                        break

            if not all_survived:
                # Derivation rejected — backtrack
                logger.warning(
                    "Step %d: derivation rejected. Backtracking.",
                    step_number,
                )
                # Remove the computed fact we just added (it's from rejected step)
                session.fact_pool.facts.pop(computed_fact.id, None)
                session.fact_pool._computed_counter -= 1

                restart = self._bt.handle_inference_failure(session)
                if restart:
                    self._executor.reset_namespace()
                    rejection_feedback = None
                    logger.info("Cascade limit — restarting from scratch.")
                else:
                    # Carry rejection feedback so the model knows what went wrong
                    rejection_feedback = rejected_feedback
                    session.total_step_retries += 1
                continue

            # ── Phase 5: All derivations survived — add DERIVED facts
            derived_fact_ids: list[str] = []
            for i, derivation in enumerate(reasoning.derivations):
                fr = falsification_results[i]
                # Confidence mapping:
                #   SURVIVE + executed → high confidence (use e_value, min 20)
                #   INCONCLUSIVE      → moderate confidence (e_value=5)
                #   not executed       → moderate (e_value=5)
                if fr.verdict == Verdict.SURVIVE and fr.executed:
                    e_val = max(fr.e_value, 20.0)
                elif fr.verdict == Verdict.INCONCLUSIVE or not fr.executed:
                    e_val = 5.0
                else:
                    e_val = fr.e_value
                fact = session.fact_pool.add_derived(
                    statement=derivation.premise,
                    value=derivation.claimed_value,
                    step=step_number,
                    depends_on=derivation.depends_on,
                    e_value=e_val,
                )
                derived_fact_ids.append(fact.id)
                logger.info("  Derived fact added: %s (conf=%.4f)", fact.id, fact.confidence)

            # ── Phase 6: Record completed step ──────────────────────
            session.steps.append(
                CompletedStep(
                    step_number=step_number,
                    objective=reasoning.objective,
                    facts_used=reasoning.facts_used,
                    thought=reasoning.thought,
                    action=reasoning.action,
                    observation=observation,
                    result_variable=reasoning.result_variable,
                    derivations=reasoning.derivations,
                    falsification_results=falsification_results,
                    computed_fact_ids=[computed_fact.id],
                    derived_fact_ids=derived_fact_ids,
                )
            )
            self._bt.reset_cascade_counter()
            rejection_feedback = None  # Clear any prior rejection

        if self._bt.is_unsolvable(session) and session.final_answer is None:
            session.final_answer = "UNSOLVABLE"

        return session
