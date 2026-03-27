"""Adversarial diagnostic: problems designed to trigger verification failures
and backtracking. Uses the FULL inference retry loop (Mode B) and cascade
backtracking (Mode C) — the actual VAR innovation.
"""

import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: Set GEMINI_API_KEY in .env first")
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
from var_reasoning.models.schemas import VerificationType
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


# ── Adversarial problems ────────────────────────────────────────────
# Designed to trigger: wrong first inferences, overcounting, off-by-one,
# faulty independence assumptions, losing solutions, and wrong quantifiers.

PROBLEMS = [
    {
        "id": "T1_fence_post",
        "category": "off-by-one trap",
        "text": (
            "A straight fence is 100 meters long. Fence posts are placed "
            "every 10 meters along the fence, including both ends. "
            "How many fence posts are there?"
        ),
        "expected": "11",
        "trap": "Model says 10 (divides 100/10, forgets the +1 for endpoints).",
    },
    {
        "id": "T2_conditional_prob",
        "category": "conditional probability trap",
        "text": (
            "A disease test has 99% sensitivity (true positive rate) and "
            "99% specificity (true negative rate). If 1 in 1000 people "
            "have the disease, what is the probability someone who tests "
            "positive actually has the disease? Give your answer as a "
            "percentage rounded to one decimal place."
        ),
        "expected": "9.0",
        "trap": "Model confuses P(positive|disease) with P(disease|positive). "
                "Likely to say 99% instead of ~9%.",
    },
    {
        "id": "T3_overcounting",
        "category": "overcounting trap",
        "text": (
            "How many 3-digit numbers have at least one digit equal to 5?"
        ),
        "expected": "252",
        "trap": "Common to double-count numbers with multiple 5s (e.g. 555). "
                "Must use inclusion-exclusion or complementary counting.",
    },
    {
        "id": "T4_modular_trap",
        "category": "false pattern trap",
        "text": (
            "What is the units digit of 2^2026?"
        ),
        "expected": "4",
        "trap": "If the model tests 2^1..2^4 and sees cycle [2,4,8,6], it "
                "must correctly compute 2026 mod 4 = 2, so units digit is 4. "
                "Off-by-one in cycle indexing gives wrong answer.",
    },
    {
        "id": "T5_monty_hall",
        "category": "probability reasoning trap",
        "text": (
            "In the Monty Hall problem, you pick door 1. The host, who "
            "knows what's behind each door, opens door 3, revealing a "
            "goat. You can switch to door 2 or stay with door 1. "
            "What is the probability of winning the car if you switch? "
            "Give your answer as a simplified fraction."
        ),
        "expected": "2/3",
        "trap": "Model may claim 1/2 by reasoning 'two doors left, 50/50'. "
                "Verification of the independence claim should fail.",
    },
    {
        "id": "T6_derangement",
        "category": "subtle formula trap",
        "text": (
            "How many derangements (permutations with no fixed points) "
            "are there of the set {1, 2, 3, 4, 5}?"
        ),
        "expected": "44",
        "trap": "Model may confuse derangement formula. D(5)=44 but easy "
                "to get 45 or 53 with wrong inclusion-exclusion.",
    },
    {
        "id": "T7_integral_trap",
        "category": "absolute value trap",
        "text": (
            "What is the total area (not net area) between the curve "
            "y = x^3 - x and the x-axis on the interval [-1, 2]? "
            "Give your answer as a simplified fraction."
        ),
        "expected": "11/4",
        "trap": "Naive integration without handling sign changes gives the "
                "net integral (wrong answer). Must split at roots and take "
                "absolute values: |integral[-1,0]| + |integral[0,1]| + integral[1,2].",
    },
    {
        "id": "T8_pigeonhole_false",
        "category": "false universal claim",
        "text": (
            "Is it true that for every set of 10 distinct positive integers "
            "less than or equal to 100, there must exist two disjoint "
            "subsets with equal sums? Answer 'yes' or 'no'."
        ),
        "expected": "yes",
        "trap": "Model may test a few cases and not prove the universal. "
                "Correct proof: 2^10 - 1 = 1023 subsets, max possible sum "
                "= 100+99+...+91 = 955, so by pigeonhole two subsets share "
                "a sum. Removing common elements gives disjoint pair.",
    },
    {
        "id": "T9_graph_coloring",
        "category": "subtle constraint trap",
        "text": (
            "How many ways can you color the vertices of a square with "
            "3 colors such that no two adjacent vertices share the same "
            "color? (Two colorings that differ only by rotation are "
            "considered DIFFERENT.)"
        ),
        "expected": "18",
        "trap": "Model may apply Burnside / chromatic polynomial incorrectly "
                "or forget the 'rotations are different' clause. "
                "Chromatic polynomial for C4: k^4 - 4k^3 + 6k^2 - 3k. "
                "With k=3: 81 - 108 + 54 - 9 = 18.",
    },
    {
        "id": "T10_series_convergence",
        "category": "infinite series trap",
        "text": (
            "What is the exact value of the infinite sum: "
            "sum from n=1 to infinity of n / 2^n? "
            "Give your answer as an integer."
        ),
        "expected": "2",
        "trap": "Model may compute partial sums then claim the answer is 'close "
                "to 2' without proving convergence. Must recognize this as "
                "a known series: S = sum n*x^n = x/(1-x)^2, with x=1/2 gives 2.",
    },
]


# ── Full VAR loop with inference retry ──────────────────────────────
def run_problem(gemini, executor, problem, log):
    pid = problem["id"]
    log(f"\n{'='*70}")
    log(f"PROBLEM: {pid} (category: {problem['category']})")
    log(f"{'='*70}")
    log(f"Q: {problem['text']}")
    log(f"Expected: {problem['expected']}")
    log(f"Trap: {problem['trap']}")
    log("")

    router = VerificationRouter(executor)
    bt = BacktrackManager(
        code_retries=3,
        inference_retries=5,
        cascade_limit=3,
        total_backtrack_limit=20,
        max_steps=25,
    )
    session = Session(problem_id=pid, problem_text=problem["text"])
    executor.reset_namespace()

    events = []  # detailed event log
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
        log(f"  [thought] {thought[:120]}...")

        # Phase 2: Execute with repair (Mode A)
        success, observation = executor.execute(action)
        if not success:
            prior_attempts = []
            current_code, current_error = action, observation
            repaired = False
            for retry in range(bt.code_retries):
                session.total_code_retries += 1
                event("code_retry", attempt=retry + 1, error=current_error[:100])
                log(f"  [code retry {retry+1}] {current_error[:80]}...")
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

        obs_preview = observation[:200].replace("\n", " | ")
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
        if inference_step.verification_target.informal_reason:
            log(f"  [informal_reason] {inference_step.verification_target.informal_reason}")

        # Phase 4: Verify
        t0 = time.time()
        result = router.verify(inference_step.verification_target)
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
                  error=fail_msg[:150])
            log(f"  [VERIFY FAILED] {fail_msg[:120]}")

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
                    log(f"  >> [investigate] {revision.thought[:80]}...")
                    event("retry_investigate")
                    success, new_obs = executor.execute(revision.action)
                    if not success:
                        log(f"  >> [investigate code failed] {new_obs[:80]}")
                        continue
                    new_obs_preview = new_obs[:150].replace("\n", " | ")
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
                    res2 = router.verify(inf2.verification_target)
                    vk2 = inf2.verification_target.type.value
                    session.verification_type_counts[vk2] = (
                        session.verification_type_counts.get(vk2, 0) + 1
                    )
                    pk2 = inf2.reasoning_pattern.value
                    session.reasoning_pattern_counts[pk2] = (
                        session.reasoning_pattern_counts.get(pk2, 0) + 1
                    )
                    log(f"  >> [re-verify] {'PASS' if res2.passed else 'FAIL'}")

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
                    log(f"  >> [revise] new conclusion: {rev_conclusion[:100]}")
                    event("retry_revise")

                    inf_str = build_inference_string(
                        current_premises, current_conclusion
                    )
                    prior_attempts.append(
                        (inf_str, current_vtype, current_failure)
                    )

                    target = revision.revised_verification_target
                    res2 = router.verify(target)
                    vk2 = target.type.value
                    session.verification_type_counts[vk2] = (
                        session.verification_type_counts.get(vk2, 0) + 1
                    )
                    rp = (
                        revision.revised_reasoning_pattern.value
                        if revision.revised_reasoning_pattern
                        else current_pattern
                    )
                    session.reasoning_pattern_counts[rp] = (
                        session.reasoning_pattern_counts.get(rp, 0) + 1
                    )
                    log(f"  >> [re-verify] {'PASS' if res2.passed else 'FAIL'}")

                    if res2.passed or target.type == VerificationType.INFORMAL:
                        from var_reasoning.models.schemas import ReasoningPattern
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
                # Mode C: cascade backtrack
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

    summary = {
        "problem_id": pid,
        "category": problem["category"],
        "expected": problem["expected"],
        "trap": problem["trap"],
        "predicted": session.final_answer,
        "correct": _normalize(session.final_answer) == _normalize(problem["expected"]),
        "num_steps": len(session.steps),
        "total_backtracks": session.total_backtracks,
        "total_code_retries": session.total_code_retries,
        "total_inference_retries": session.total_inference_retries,
        "total_llm_calls": session.total_llm_calls,
        "input_tokens": session.total_input_tokens,
        "output_tokens": session.total_output_tokens,
        "cost_usd": round(cost, 5),
        "elapsed_s": round(total_elapsed, 1),
        "verification_type_counts": session.verification_type_counts,
        "reasoning_pattern_counts": session.reasoning_pattern_counts,
        "informal_skip_count": session.informal_skip_count,
        "informal_without_reason_count": session.informal_without_reason_count,
        "events": events,
    }

    log(f"\n  RESULT: {'CORRECT' if summary['correct'] else '*** WRONG ***'}")
    log(f"  predicted={session.final_answer}  expected={problem['expected']}")
    log(f"  steps={len(session.steps)} backtracks={session.total_backtracks} "
        f"inference_retries={session.total_inference_retries} "
        f"code_retries={session.total_code_retries}")
    log(f"  llm_calls={session.total_llm_calls} cost=${cost:.4f} time={total_elapsed:.1f}s")
    log(f"  patterns={session.reasoning_pattern_counts}")
    log(f"  vtypes={session.verification_type_counts}")

    return summary


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
    s = ", ".join(part.strip() for part in s.split(","))
    return s.strip()


def main():
    output_lines = []

    def log(msg=""):
        print(msg, flush=True)
        output_lines.append(msg)

    log("=" * 70)
    log("VAR ADVERSARIAL RUN — 10 Trap Problems")
    log("Full inference retry loop (Mode B) + cascade backtracking (Mode C)")
    log("=" * 70)

    gemini = GeminiProvider(model_name="gemini-2.5-flash")
    executor = LocalExecutor()

    all_summaries = []
    for problem in PROBLEMS:
        try:
            summary = run_problem(gemini, executor, problem, log)
            all_summaries.append(summary)
        except Exception as e:
            import traceback
            log(f"\n[EXCEPTION] {problem['id']}: {e}")
            log(traceback.format_exc())
            all_summaries.append({
                "problem_id": problem["id"],
                "category": problem["category"],
                "error": str(e),
            })

    # ── Aggregate ───────────────────────────────────────────────────
    log(f"\n\n{'='*70}")
    log("AGGREGATE ANALYSIS")
    log(f"{'='*70}")

    correct = sum(1 for s in all_summaries if s.get("correct"))
    total = len(all_summaries)
    log(f"Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")

    total_bt = sum(s.get("total_backtracks", 0) for s in all_summaries)
    total_inf_retries = sum(s.get("total_inference_retries", 0) for s in all_summaries)
    total_code_retries = sum(s.get("total_code_retries", 0) for s in all_summaries)
    total_cost = sum(s.get("cost_usd", 0) for s in all_summaries)
    total_calls = sum(s.get("total_llm_calls", 0) for s in all_summaries)
    total_time = sum(s.get("elapsed_s", 0) for s in all_summaries)

    log(f"Total backtracks: {total_bt}")
    log(f"Total inference retries: {total_inf_retries}")
    log(f"Total code retries: {total_code_retries}")
    log(f"Total cost: ${total_cost:.4f}")
    log(f"Total LLM calls: {total_calls}")
    log(f"Total time: {total_time:.0f}s")

    # Problems that triggered recovery
    bt_problems = [s for s in all_summaries if s.get("total_backtracks", 0) > 0]
    retry_problems = [s for s in all_summaries if s.get("total_inference_retries", 0) > 0]
    log(f"\nProblems with backtracks: {len(bt_problems)}/{total}")
    log(f"Problems with inference retries: {len(retry_problems)}/{total}")

    # Recovery success rate
    bt_correct = sum(1 for s in bt_problems if s.get("correct"))
    retry_correct = sum(1 for s in retry_problems if s.get("correct"))
    if bt_problems:
        log(f"Backtrack recovery success rate: {bt_correct}/{len(bt_problems)}")
    if retry_problems:
        log(f"Inference retry recovery success rate: {retry_correct}/{len(retry_problems)}")

    log(f"\nPer-problem breakdown:")
    header = (f"{'ID':<22} {'Category':<28} {'OK?':<5} {'Steps':<6} "
              f"{'BT':<4} {'InfR':<5} {'Calls':<6} {'Cost':<8} {'Time':<6}")
    log(header)
    log("-" * len(header))
    for s in all_summaries:
        if "error" in s:
            log(f"{s['problem_id']:<22} {s['category']:<28} ERROR")
            continue
        ok = "Y" if s["correct"] else "N"
        log(
            f"{s['problem_id']:<22} {s['category']:<28} {ok:<5} "
            f"{s['num_steps']:<6} {s['total_backtracks']:<4} "
            f"{s['total_inference_retries']:<5} "
            f"{s['total_llm_calls']:<6} ${s['cost_usd']:<7.4f} "
            f"{s['elapsed_s']:<6.0f}"
        )

    # Pattern & vtype distribution
    all_patterns: dict[str, int] = {}
    all_vtypes: dict[str, int] = {}
    for s in all_summaries:
        for k, v in s.get("reasoning_pattern_counts", {}).items():
            all_patterns[k] = all_patterns.get(k, 0) + v
        for k, v in s.get("verification_type_counts", {}).items():
            all_vtypes[k] = all_vtypes.get(k, 0) + v

    log(f"\nReasoning patterns: {dict(sorted(all_patterns.items()))}")
    log(f"Verification types: {dict(sorted(all_vtypes.items()))}")

    # Event type summary
    all_event_types: dict[str, int] = {}
    for s in all_summaries:
        for e in s.get("events", []):
            et = e["type"]
            all_event_types[et] = all_event_types.get(et, 0) + 1
    log(f"Event distribution: {dict(sorted(all_event_types.items()))}")

    informal_total = sum(s.get("informal_skip_count", 0) for s in all_summaries)
    total_verifications = sum(all_vtypes.values())
    if total_verifications:
        log(f"Formalization rate: {(total_verifications - informal_total) / total_verifications:.1%}")

    # Save
    with open("adversarial_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    with open("adversarial_results.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    log(f"\nSaved: adversarial_output.txt, adversarial_results.json")


if __name__ == "__main__":
    main()
