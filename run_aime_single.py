"""Focused test: AIME 2025 I Problem 13 (expected answer: 204).

Uses the FULL VAR pipeline with:
- Fix 3: Tautological verification rejection (AST-based)
- Fix 4: Pattern-vtype enforcement (algebraic→sympy, universal→z3)
- Fix 5: Strengthened prompts (one claim per step, cite observed values)
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: Set GEMINI_API_KEY in .env or environment first")
    sys.exit(1)

import subprocess
import tempfile

from var_reasoning.engine.backtracking import BacktrackManager
from var_reasoning.engine.feedback import (
    build_conversation_history,
    build_inference_context,
    make_code_repair_prompt,
    make_inference_retry_prompt,
)
from var_reasoning.models.gemini_provider import GeminiProvider
from var_reasoning.models.schemas import ReasoningPattern, VerificationType
from var_reasoning.models.state import (
    CompletedStep,
    Session,
    VerificationResult,
    build_inference_string,
)
from var_reasoning.prompts.inference_prompt import INFERENCE_PROMPT
from var_reasoning.prompts.reasoning_prompt import REASONING_PROMPT
from var_reasoning.verification.verification_router import VerificationRouter


# ── Local executor ──────────────────────────────────────────────────
class LocalExecutor:
    def __init__(self):
        self._cumulative_code: list[str] = []

    def reset_namespace(self):
        self._cumulative_code = []

    @staticmethod
    def _strip_fences(code: str) -> str:
        code = code.strip()
        if code.startswith("```"):
            lines = code.splitlines()
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)
        return code

    def execute(self, code: str, timeout: int = 30) -> tuple[bool, str]:
        code = self._strip_fences(code)
        prior = "\n".join(self._cumulative_code)
        if prior:
            suppressed_prior = (
                "import sys as _sys, io as _io\n"
                "_old_stdout = _sys.stdout\n"
                "_sys.stdout = _io.StringIO()\n"
                + prior + "\n"
                "_sys.stdout = _old_stdout\n"
            )
        else:
            suppressed_prior = ""
        full_script = suppressed_prior + code
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(full_script)
                f.flush()
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            if result.returncode == 0:
                self._cumulative_code.append(code)
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "Timeout: code took too long"
        except Exception as e:
            return False, str(e)


PROBLEM = {
    "id": "2025_AIME_I_P13",
    "text": (
        "Alex divides a disk into four quadrants with two perpendicular "
        "diameters intersecting at the center of the disk. He draws 25 "
        "more line segments through the disk, drawing each segment by "
        "selecting two points at random on the perimeter of the disk in "
        "different quadrants and connecting these two points. Find the "
        "expected number of regions into which these 27 line segments "
        "divide the disk."
    ),
    "expected": "204",
}


def _normalize(s: str | None) -> str:
    if s is None:
        return ""
    s = s.strip().lower()
    for prefix in ["$", "\\boxed{", "the answer is "]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    for suffix in ["$", "}", ".", "%"]:
        if s.endswith(suffix):
            s = s[:-1]
    return s.strip()


def run():
    output_lines: list[str] = []

    def log(msg=""):
        print(msg, flush=True)
        output_lines.append(msg)

    gemini = GeminiProvider()
    executor = LocalExecutor()
    router = VerificationRouter(LocalExecutor, gemini=gemini)
    bt = BacktrackManager(
        code_retries=3,
        inference_retries=5,
        cascade_limit=3,
        total_backtrack_limit=20,
        max_steps=25,
    )

    pid = PROBLEM["id"]
    session = Session(problem_id=pid, problem_text=PROBLEM["text"])
    executor.reset_namespace()

    log(f"{'='*70}")
    log(f"PROBLEM: {pid}")
    log(f"Expected: {PROBLEM['expected']}")
    log(f"{'='*70}")
    log(f"Q: {PROBLEM['text']}")
    log("")

    events = []
    total_start = time.time()

    def event(etype, **kwargs):
        t = round(time.time() - total_start, 1)
        e = {"t": t, "type": etype, **kwargs}
        events.append(e)
        return e

    while not bt.should_stop(session):
        step_num = len(session.steps) + 1
        log(f"\n--- Step {step_num} ---")

        # Phase 1: Reasoning
        step_output = None
        for _attempt in range(3):
            conversation = build_conversation_history(session)
            t0 = time.time()
            raw_output, usage = gemini.generate_reasoning_step(
                REASONING_PROMPT, conversation
            )
            elapsed = time.time() - t0
            session.total_llm_calls += 1
            session.total_input_tokens += usage.input_tokens
            session.total_output_tokens += usage.output_tokens
            log(f"  [reasoning] {elapsed:.1f}s, {usage.input_tokens}+{usage.output_tokens} tok")
            if raw_output and (raw_output.final_answer or raw_output.reasoning):
                step_output = raw_output
                break
            log(f"  [WARN] malformed, retrying...")

        if step_output is None:
            event("backtrack_reasoning_fail")
            log(f"  [BACKTRACK] reasoning failed")
            bt.handle_code_failure(session)
            continue

        if step_output.final_answer:
            session.final_answer = step_output.final_answer.answer
            session.justification = step_output.final_answer.justification
            event("final_answer", answer=session.final_answer)
            log(f"  [FINAL ANSWER] {session.final_answer}")
            break

        thought = step_output.reasoning.thought
        action = step_output.reasoning.action
        objective = step_output.reasoning.objective
        depends_on = step_output.reasoning.depends_on
        result_variable = step_output.reasoning.result_variable
        log(f"  [thought] {thought[:200]}...")

        # Phase 2: Execute with repair (Mode A)
        success, observation = executor.execute(action)
        if not success:
            prior_attempts = []
            current_code, current_error = action, observation
            repaired = False
            for retry in range(bt.code_retries):
                session.total_code_retries += 1
                event("code_retry", attempt=retry + 1, error=current_error[:100])
                log(f"  [code retry {retry+1}] {current_error[:100]}...")
                prompt = make_code_repair_prompt(
                    thought, current_code, current_error, prior_attempts
                )
                code_fix, fix_usage = gemini.generate_code_fix(prompt)
                session.total_llm_calls += 1
                session.total_input_tokens += fix_usage.input_tokens
                session.total_output_tokens += fix_usage.output_tokens
                prior_attempts.append((current_code, current_error))
                current_code = code_fix.fixed_code
                success, observation = executor.execute(current_code)
                if success:
                    action = current_code
                    repaired = True
                    event("code_repair_success", attempt=retry + 1)
                    log(f"  [repair OK] attempt {retry+1}")
                    break
                current_error = observation

            if not repaired:
                event("backtrack_code_fail")
                log(f"  [BACKTRACK] code repair exhausted")
                bt.handle_code_failure(session)
                continue

        obs_preview = observation[:300].replace("\n", " | ")
        log(f"  [observation] {obs_preview}")

        # Phase 3: Inference
        ctx = build_inference_context(
            session, thought, action, observation,
            objective=objective, result_variable=result_variable,
        )
        t0 = time.time()
        inference_step, inf_usage = gemini.generate_inference(
            INFERENCE_PROMPT, ctx
        )
        inf_elapsed = time.time() - t0
        session.total_llm_calls += 1
        session.total_input_tokens += inf_usage.input_tokens
        session.total_output_tokens += inf_usage.output_tokens
        log(f"  [inference] {inf_elapsed:.1f}s")
        log(f"  [premises] {inference_step.premises}")
        log(f"  [conclusion] {inference_step.conclusion}")
        log(f"  [pattern] {inference_step.reasoning_pattern.value}")
        log(f"  [vtype] {inference_step.verification_target.type.value}")
        log(f"  [verification_code] {inference_step.verification_target.statement[:200]}")

        # Phase 4: Verify (multi-layer pipeline)
        router.set_prior_code([step.action for step in session.steps])
        t0 = time.time()
        prior_obs = [step.observation for step in session.steps]
        result = router.verify(
            inference_step.verification_target,
            inference_step.reasoning_pattern,
            problem_text=session.problem_text,
            observation=observation,
            result_variable=result_variable,
            step_number=len(session.steps) + 1,
            conclusion=inference_step.conclusion,
            depends_on=depends_on,
            prior_observations=prior_obs,
        )
        ver_elapsed = time.time() - t0

        vtype_key = inference_step.verification_target.type.value
        session.verification_type_counts[vtype_key] = (
            session.verification_type_counts.get(vtype_key, 0) + 1
        )
        pat_key = inference_step.reasoning_pattern.value
        session.reasoning_pattern_counts[pat_key] = (
            session.reasoning_pattern_counts.get(pat_key, 0) + 1
        )
        if inference_step.verification_target.type == VerificationType.INFORMAL:
            session.informal_skip_count += 1
            if not inference_step.verification_target.informal_reason:
                session.informal_without_reason_count += 1

        if result.simulation_ran:
            emp = f"{result.simulation_empirical:.6g}" if result.simulation_empirical is not None else "n/a"
            log(f"  [sim] verdict={result.simulation_verdict}, empirical={emp}, e={result.simulation_e_value:.2f}")

        if result.passed or inference_step.verification_target.type == VerificationType.INFORMAL:
            status = "PASSED" if result.passed else "INFORMAL_SKIP"
            event("verify_pass", pattern=pat_key, vtype=vtype_key)
            log(f"  [verify] {status} ({ver_elapsed:.1f}s)")
            bt.reset_cascade_counter()
            completed = CompletedStep(
                step_number=len(session.steps) + 1,
                objective=objective,
                depends_on=depends_on,
                thought=thought,
                action=action,
                observation=observation,
                result_variable=result_variable,
                premises=inference_step.premises,
                conclusion=inference_step.conclusion,
                reasoning_pattern=inference_step.reasoning_pattern,
                inference=build_inference_string(
                    inference_step.premises, inference_step.conclusion
                ),
                verification_target=inference_step.verification_target,
                verification_result=result,
            )
            session.steps.append(completed)
        else:
            # ── FULL INFERENCE RETRY LOOP (Mode B) ──────────────────
            fail_msg = result.error_message or "Verification failed"
            if result.counterexample:
                fail_msg += f"\nCounterexample: {result.counterexample}"
            event("verify_fail", pattern=pat_key, vtype=vtype_key,
                  error=fail_msg[:200])
            log(f"  [VERIFY FAILED] {fail_msg[:200]}")

            current_premises = inference_step.premises
            current_conclusion = inference_step.conclusion
            current_pattern = inference_step.reasoning_pattern.value
            current_vtype = vtype_key
            current_failure = fail_msg
            prior_attempts: list[tuple[str, str, str]] = []
            retry_succeeded = False

            for retry_num in range(bt.inference_retries):
                session.total_inference_retries += 1
                event("inference_retry", attempt=retry_num + 1)
                log(f"\n  >> Inference retry {retry_num+1}/{bt.inference_retries}")

                prompt = make_inference_retry_prompt(
                    session,
                    observation,
                    current_premises,
                    current_conclusion,
                    current_pattern,
                    current_vtype,
                    current_failure,
                    prior_attempts,
                )
                revision, rev_usage = gemini.generate_inference_revision(prompt)
                session.total_llm_calls += 1
                session.total_input_tokens += rev_usage.input_tokens
                session.total_output_tokens += rev_usage.output_tokens

                if revision.choice == "investigate" and revision.thought and revision.action:
                    log(f"  >> [investigate] {revision.thought[:120]}...")
                    event("retry_investigate")
                    success, new_obs = executor.execute(revision.action)
                    if not success:
                        log(f"  >> [investigate code failed] {new_obs[:100]}")
                        continue
                    new_obs_preview = new_obs[:200].replace("\n", " | ")
                    log(f"  >> [new observation] {new_obs_preview}")
                    ctx2 = build_inference_context(
                        session, revision.thought, revision.action, new_obs
                    )
                    inf2, inf2_usage = gemini.generate_inference(
                        INFERENCE_PROMPT, ctx2
                    )
                    session.total_llm_calls += 1
                    session.total_input_tokens += inf2_usage.input_tokens
                    session.total_output_tokens += inf2_usage.output_tokens

                    router.set_prior_code([step.action for step in session.steps])
                    res2 = router.verify(
                        inf2.verification_target, inf2.reasoning_pattern,
                        problem_text=session.problem_text,
                        observation=new_obs,
                        result_variable=result_variable,
                        step_number=len(session.steps) + 1,
                        conclusion=inf2.conclusion,
                        depends_on=depends_on,
                        prior_observations=[s.observation for s in session.steps],
                    )
                    vk2 = inf2.verification_target.type.value
                    session.verification_type_counts[vk2] = (
                        session.verification_type_counts.get(vk2, 0) + 1
                    )
                    pk2 = inf2.reasoning_pattern.value
                    session.reasoning_pattern_counts[pk2] = (
                        session.reasoning_pattern_counts.get(pk2, 0) + 1
                    )
                    log(f"  >> [re-verify] {'PASS' if res2.passed else 'FAIL'}")
                    if res2.error_message:
                        log(f"  >> [error] {res2.error_message[:150]}")

                    if res2.passed or inf2.verification_target.type == VerificationType.INFORMAL:
                        event("retry_investigate_success", attempt=retry_num + 1)
                        bt.reset_cascade_counter()
                        completed = CompletedStep(
                            step_number=len(session.steps) + 1,
                            objective=objective,
                            depends_on=depends_on,
                            thought=revision.thought,
                            action=revision.action,
                            observation=new_obs,
                            result_variable=result_variable,
                            premises=inf2.premises,
                            conclusion=inf2.conclusion,
                            reasoning_pattern=inf2.reasoning_pattern,
                            inference=build_inference_string(
                                inf2.premises, inf2.conclusion
                            ),
                            verification_target=inf2.verification_target,
                            verification_result=res2,
                        )
                        session.steps.append(completed)
                        retry_succeeded = True
                        break

                    inf_str = build_inference_string(inf2.premises, inf2.conclusion)
                    current_premises = inf2.premises
                    current_conclusion = inf2.conclusion
                    current_pattern = inf2.reasoning_pattern.value
                    current_vtype = inf2.verification_target.type.value
                    current_failure = res2.error_message or "Verification failed"
                    prior_attempts.append(
                        (inf_str, current_vtype, current_failure)
                    )

                elif (
                    revision.choice == "revise"
                    and revision.revised_conclusion
                    and revision.revised_verification_target
                ):
                    rev_premises = revision.revised_premises or current_premises
                    rev_conclusion = revision.revised_conclusion
                    log(f"  >> [revise] new conclusion: {rev_conclusion[:120]}")
                    log(f"  >> [revised vcode] {revision.revised_verification_target.statement[:150]}")
                    event("retry_revise")

                    inf_str = build_inference_string(
                        current_premises, current_conclusion
                    )
                    prior_attempts.append(
                        (inf_str, current_vtype, current_failure)
                    )

                    target = revision.revised_verification_target
                    rp = (
                        revision.revised_reasoning_pattern.value
                        if revision.revised_reasoning_pattern
                        else current_pattern
                    )
                    router.set_prior_code([step.action for step in session.steps])
                    res2 = router.verify(
                        target, ReasoningPattern(rp),
                        problem_text=session.problem_text,
                        observation=observation,
                        result_variable=result_variable,
                        step_number=len(session.steps) + 1,
                        conclusion=rev_conclusion,
                        depends_on=depends_on,
                        prior_observations=[s.observation for s in session.steps],
                    )
                    vk2 = target.type.value
                    session.verification_type_counts[vk2] = (
                        session.verification_type_counts.get(vk2, 0) + 1
                    )
                    session.reasoning_pattern_counts[rp] = (
                        session.reasoning_pattern_counts.get(rp, 0) + 1
                    )
                    log(f"  >> [re-verify] {'PASS' if res2.passed else 'FAIL'}")
                    if res2.error_message:
                        log(f"  >> [error] {res2.error_message[:150]}")

                    if res2.passed or target.type == VerificationType.INFORMAL:
                        event("retry_revise_success", attempt=retry_num + 1)
                        bt.reset_cascade_counter()
                        completed = CompletedStep(
                            step_number=len(session.steps) + 1,
                            objective=objective,
                            depends_on=depends_on,
                            thought=thought,
                            action=action,
                            observation=observation,
                            result_variable=result_variable,
                            premises=rev_premises,
                            conclusion=rev_conclusion,
                            reasoning_pattern=ReasoningPattern(rp),
                            inference=build_inference_string(
                                rev_premises, rev_conclusion
                            ),
                            verification_target=target,
                            verification_result=res2,
                        )
                        session.steps.append(completed)
                        retry_succeeded = True
                        break

                    current_premises = rev_premises
                    current_conclusion = rev_conclusion
                    current_pattern = rp
                    current_vtype = vk2
                    current_failure = res2.error_message or "Verification failed"
                else:
                    inf_str = build_inference_string(
                        current_premises, current_conclusion
                    )
                    prior_attempts.append(
                        (inf_str, current_vtype, current_failure)
                    )
                    log(f"  >> [invalid revision] skipping")

            if not retry_succeeded:
                restart = bt.handle_inference_failure(session)
                event("cascade_backtrack", restart=restart,
                      total_bt=session.total_backtracks)
                if restart:
                    executor.reset_namespace()
                    log(f"  [CASCADE RESTART] total_backtracks={session.total_backtracks}")
                else:
                    log(f"  [BACKTRACK] removed last step, "
                        f"total_backtracks={session.total_backtracks}")

    total_elapsed = time.time() - total_start

    if bt.is_unsolvable(session) and session.final_answer is None:
        session.final_answer = "UNSOLVABLE"
        session.justification = "Exceeded backtrack limits."

    cost = (
        (session.total_input_tokens / 1_000_000) * 0.15
        + (session.total_output_tokens / 1_000_000) * 0.60
    )

    correct = _normalize(session.final_answer) == _normalize(PROBLEM["expected"])
    log(f"\n{'='*70}")
    log(f"RESULT: {'CORRECT' if correct else '*** WRONG ***'}")
    log(f"predicted={session.final_answer}  expected={PROBLEM['expected']}")
    log(f"steps={len(session.steps)} backtracks={session.total_backtracks} "
        f"inference_retries={session.total_inference_retries} "
        f"code_retries={session.total_code_retries}")
    log(f"llm_calls={session.total_llm_calls} cost=${cost:.4f} time={total_elapsed:.1f}s")
    log(f"patterns={session.reasoning_pattern_counts}")
    log(f"vtypes={session.verification_type_counts}")
    log(f"events={json.dumps([e for e in events if 'fail' in e['type'] or 'retry' in e['type'] or 'backtrack' in e['type']], indent=2)}")
    log(f"{'='*70}")

    # Save output
    with open("aime_2025_i_p13_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    log("\nSaved to aime_2025_i_p13_output.txt")


if __name__ == "__main__":
    run()
