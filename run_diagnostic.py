"""Run 10 problems of increasing difficulty through the full VAR pipeline.

Captures detailed per-step traces for analysis: reasoning patterns chosen,
verification types, backtracks, premises/conclusion structures, token costs.
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


# ── Local executor (no Docker) ──────────────────────────────────────
class LocalExecutor:
    """Runs Python code locally via subprocess (for testing only)."""

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


# ── 10 problems: increasing difficulty ──────────────────────────────
PROBLEMS = [
    {
        "id": "L1_arithmetic",
        "difficulty": "trivial",
        "text": "What is 247 * 38?",
        "expected": "9386",
    },
    {
        "id": "L2_modular",
        "difficulty": "easy",
        "text": "What is the remainder when 2^100 is divided by 7?",
        "expected": "2",
    },
    {
        "id": "L3_combinatorics",
        "difficulty": "easy",
        "text": (
            "How many ways can you choose 3 people from a group of 10 "
            "to form a committee?"
        ),
        "expected": "120",
    },
    {
        "id": "L4_number_theory",
        "difficulty": "medium",
        "text": "Find the sum of all prime numbers less than 50.",
        "expected": "328",
    },
    {
        "id": "L5_optimization",
        "difficulty": "medium",
        "text": (
            "A farmer has 200 meters of fencing. What is the maximum area "
            "in square meters of a rectangular pen he can enclose?"
        ),
        "expected": "2500",
    },
    {
        "id": "L6_probability",
        "difficulty": "medium-hard",
        "text": (
            "You roll two fair six-sided dice. What is the probability that "
            "the sum is at least 9? Give your answer as a simplified fraction."
        ),
        "expected": "5/18",
    },
    {
        "id": "L7_algebra",
        "difficulty": "hard",
        "text": (
            "Find all real solutions to the equation x^4 - 5x^2 + 4 = 0. "
            "List them in increasing order, separated by commas."
        ),
        "expected": "-2, -1, 1, 2",
    },
    {
        "id": "L8_counting",
        "difficulty": "hard",
        "text": (
            "How many distinct 4-letter strings can be formed from the "
            "letters of the word 'MISSISSIPPI'?"
        ),
        "expected": "176",
    },
    {
        "id": "L9_number_theory_hard",
        "difficulty": "very-hard",
        "text": (
            "Find the last three digits of 7^999. "
            "That is, compute 7^999 mod 1000."
        ),
        "expected": "143",
    },
    {
        "id": "L10_combinatorial_proof",
        "difficulty": "very-hard",
        "text": (
            "A standard 8x8 chessboard has two diagonally opposite corners "
            "removed. Can you tile the remaining 62 squares with 31 dominoes, "
            "where each domino covers exactly two adjacent squares? "
            "Answer 'yes' or 'no' and prove your answer."
        ),
        "expected": "no",
    },
]


# ── Trace data structure ────────────────────────────────────────────
def run_problem(gemini, executor, problem, log):
    """Run one problem through the full VAR loop with detailed tracing."""
    pid = problem["id"]
    log(f"\n{'='*70}")
    log(f"PROBLEM: {pid} (difficulty: {problem['difficulty']})")
    log(f"{'='*70}")
    log(f"Q: {problem['text']}")
    log(f"Expected: {problem['expected']}")
    log("")

    router = VerificationRouter(LocalExecutor, gemini=gemini)
    bt = BacktrackManager()
    session = Session(problem_id=pid, problem_text=problem["text"])
    executor.reset_namespace()

    step_traces = []
    total_start = time.time()

    while not bt.should_stop(session):
        step_start = time.time()
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
            log(f"  [reasoning call] {elapsed:.1f}s, {usage.input_tokens}+{usage.output_tokens} tokens")
            if raw_output and (raw_output.final_answer or raw_output.reasoning):
                step_output = raw_output
                break
            log(f"  [WARN] malformed output, retrying...")

        if step_output is None:
            log(f"  [BACKTRACK] reasoning failed")
            bt.handle_code_failure(session)
            continue

        if step_output.final_answer:
            session.final_answer = step_output.final_answer.answer
            session.justification = step_output.final_answer.justification
            log(f"  [FINAL ANSWER] {session.final_answer}")
            break

        thought = step_output.reasoning.thought
        action = step_output.reasoning.action
        objective = step_output.reasoning.objective
        depends_on = step_output.reasoning.depends_on
        result_variable = step_output.reasoning.result_variable
        log(f"  [thought] {thought[:120]}...")

        # Phase 2: Execute
        success, observation = executor.execute(action)
        if not success:
            # Try repair
            prior_attempts = []
            current_code, current_error = action, observation
            repaired = False
            for retry in range(bt.code_retries):
                session.total_code_retries += 1
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
                    log(f"  [repair] fixed on attempt {retry+1}")
                    break
                current_error = observation

            if not repaired:
                log(f"  [BACKTRACK] code repair exhausted")
                bt.handle_code_failure(session)
                continue

        obs_preview = observation[:150].replace("\n", " | ")
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
        log(f"  [inference call] {inf_elapsed:.1f}s, {inf_usage.input_tokens}+{inf_usage.output_tokens} tokens")
        log(f"  [premises] {inference_step.premises}")
        log(f"  [conclusion] {inference_step.conclusion}")
        log(f"  [pattern] {inference_step.reasoning_pattern.value}")
        log(f"  [vtype] {inference_step.verification_target.type.value}")
        if inference_step.verification_target.informal_reason:
            log(f"  [informal_reason] {inference_step.verification_target.informal_reason}")

        # Phase 4: Verify
        router.set_prior_code([step.action for step in session.steps])
        t0 = time.time()
        result = router.verify(
            inference_step.verification_target,
            problem_text=problem["text"],
            observation=observation,
            result_variable=result_variable,
            step_number=step_num,
            conclusion=inference_step.conclusion,
            depends_on=depends_on,
            prior_observations=[step.observation for step in session.steps],
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

        step_elapsed = time.time() - step_start

        if result.passed or inference_step.verification_target.type == VerificationType.INFORMAL:
            status = "PASSED" if result.passed else "INFORMAL_SKIP"
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

            step_traces.append({
                "step": completed.step_number,
                "pattern": pat_key,
                "vtype": vtype_key,
                "status": status,
                "premises_count": len(inference_step.premises),
                "step_time_s": round(step_elapsed, 1),
            })
        else:
            err = result.error_message or "failed"
            log(f"  [verify] FAILED: {err[:100]}")
            if result.counterexample:
                log(f"  [counterexample] {result.counterexample[:100]}")

            # Simplified: just backtrack (full retry loop omitted for speed)
            restart = bt.handle_inference_failure(session)
            if restart:
                executor.reset_namespace()
                log(f"  [CASCADE RESTART]")

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
        "difficulty": problem["difficulty"],
        "expected": problem["expected"],
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
        "step_traces": step_traces,
    }

    log(f"\n  RESULT: {'CORRECT' if summary['correct'] else 'WRONG'}")
    log(f"  predicted={session.final_answer}  expected={problem['expected']}")
    log(f"  steps={len(session.steps)} backtracks={session.total_backtracks} "
        f"llm_calls={session.total_llm_calls} cost=${cost:.4f} time={total_elapsed:.1f}s")
    log(f"  patterns={session.reasoning_pattern_counts}")
    log(f"  vtypes={session.verification_type_counts}")

    return summary


def _normalize(s: str | None) -> str:
    """Basic normalization for answer comparison."""
    if s is None:
        return ""
    s = s.strip().lower()
    # Strip common wrappers
    for prefix in ["$", "\\boxed{", "the answer is "]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    for suffix in ["$", "}", "."]:
        if s.endswith(suffix):
            s = s[:-1]
    # Normalize whitespace around commas
    s = ", ".join(part.strip() for part in s.split(","))
    return s.strip()


def main():
    output_lines = []

    def log(msg=""):
        print(msg, flush=True)
        output_lines.append(msg)

    log("=" * 70)
    log("VAR DIAGNOSTIC RUN — 10 Problems, Increasing Difficulty")
    log("=" * 70)

    gemini = GeminiProvider(model_name="gemini-2.5-flash")
    executor = LocalExecutor()

    all_summaries = []
    for problem in PROBLEMS:
        try:
            summary = run_problem(gemini, executor, problem, log)
            all_summaries.append(summary)
        except Exception as e:
            log(f"\n[EXCEPTION] {problem['id']}: {e}")
            all_summaries.append({
                "problem_id": problem["id"],
                "difficulty": problem["difficulty"],
                "error": str(e),
            })

    # ── Aggregate analysis ──────────────────────────────────────────
    log(f"\n\n{'='*70}")
    log("AGGREGATE ANALYSIS")
    log(f"{'='*70}")

    correct = sum(1 for s in all_summaries if s.get("correct"))
    total = len(all_summaries)
    log(f"Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")

    total_cost = sum(s.get("cost_usd", 0) for s in all_summaries)
    total_calls = sum(s.get("total_llm_calls", 0) for s in all_summaries)
    total_time = sum(s.get("elapsed_s", 0) for s in all_summaries)
    log(f"Total cost: ${total_cost:.4f}")
    log(f"Total LLM calls: {total_calls}")
    log(f"Total time: {total_time:.0f}s")

    log(f"\nPer-problem breakdown:")
    log(f"{'ID':<25} {'Diff':<12} {'OK?':<6} {'Steps':<6} {'BT':<4} {'Calls':<6} {'Cost':<8} {'Time':<6} {'Patterns'}")
    log("-" * 120)
    for s in all_summaries:
        if "error" in s:
            log(f"{s['problem_id']:<25} {s['difficulty']:<12} ERROR")
            continue
        ok = "Y" if s["correct"] else "N"
        pats = s.get("reasoning_pattern_counts", {})
        pat_str = ", ".join(f"{k}:{v}" for k, v in sorted(pats.items()))
        log(
            f"{s['problem_id']:<25} {s['difficulty']:<12} {ok:<6} "
            f"{s['num_steps']:<6} {s['total_backtracks']:<4} "
            f"{s['total_llm_calls']:<6} ${s['cost_usd']:<7.4f} "
            f"{s['elapsed_s']:<6.0f} {pat_str}"
        )

    # Pattern distribution
    all_patterns: dict[str, int] = {}
    all_vtypes: dict[str, int] = {}
    for s in all_summaries:
        for k, v in s.get("reasoning_pattern_counts", {}).items():
            all_patterns[k] = all_patterns.get(k, 0) + v
        for k, v in s.get("verification_type_counts", {}).items():
            all_vtypes[k] = all_vtypes.get(k, 0) + v

    log(f"\nReasoning pattern distribution: {dict(sorted(all_patterns.items()))}")
    log(f"Verification type distribution: {dict(sorted(all_vtypes.items()))}")

    informal_total = sum(s.get("informal_skip_count", 0) for s in all_summaries)
    informal_no_reason = sum(s.get("informal_without_reason_count", 0) for s in all_summaries)
    total_verifications = sum(all_vtypes.values())
    if total_verifications:
        log(f"Formalization rate: {(total_verifications - informal_total) / total_verifications:.1%}")
    if informal_total:
        log(f"Informal without reason: {informal_no_reason}/{informal_total}")

    # Save outputs
    with open("diagnostic_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    with open("diagnostic_results.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    log(f"\nSaved: diagnostic_output.txt, diagnostic_results.json")


if __name__ == "__main__":
    main()
