"""Manual test: run a single problem through the full VAR system with verbose output."""

import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: Set GEMINI_API_KEY in .env first")
    sys.exit(1)

from var_reasoning.engine.backtracking import BacktrackManager
from var_reasoning.engine.feedback import (
    build_conversation_history,
    build_inference_context,
    make_code_repair_prompt,
    make_inference_retry_prompt,
)
from var_reasoning.models.gemini_provider import GeminiProvider
from var_reasoning.models.schemas import VerificationType
from var_reasoning.models.state import CompletedStep, Session, build_inference_string
from var_reasoning.prompts.inference_prompt import INFERENCE_PROMPT
from var_reasoning.prompts.reasoning_prompt import REASONING_PROMPT
from var_reasoning.verification.verification_router import VerificationRouter

# ── Local executor (no Docker needed) ───────────────────────────────
import subprocess
import tempfile


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

        # Suppress stdout from prior steps
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


# ── Config ──────────────────────────────────────────────────────────
PROBLEM = (
    "In Texas Hold'em poker, find a specific scenario where:\n"
    "1. You have the absolute nuts (best possible hand) on the flop.\n"
    "2. You still have the absolute nuts on the turn.\n"
    "3. On the river, you cannot win or even chop against ANY possible opponent holding.\n"
    "These conditions must hold for all possible sets of opponent hole cards.\n"
    "Describe: your hole cards, the flop (3 cards), the turn (1 card), and the river (1 card)."
)
# You can swap this out for any problem text you want to test.

SEPARATOR = "=" * 70


def main():
    print(SEPARATOR)
    print("VAR System -- Single Problem Test")
    print(SEPARATOR)
    print(f"\nPROBLEM:\n{PROBLEM}\n")

    # Init components
    print("Initializing Gemini provider...")
    gemini = GeminiProvider(model_name="gemini-2.5-flash")

    print("Initializing local Python executor (no Docker)...")
    executor = LocalExecutor()

    router = VerificationRouter(executor)
    bt = BacktrackManager()

    session = Session(problem_id="test_001", problem_text=PROBLEM)
    executor.reset_namespace()

    print("\n" + SEPARATOR)
    print("Starting reasoning loop...")
    print(SEPARATOR)

    step_num = 0
    while not bt.should_stop(session):
        step_num += 1
        print(f"\n{'-' * 50}")
        print(f"  STEP {step_num}")
        print(f"{'-' * 50}")

        # Phase 1: Generate reasoning step (retry on empty/malformed)
        step_output = None
        for _attempt in range(3):
            conversation = build_conversation_history(session)
            print(f"\n>>> Calling LLM for reasoning step (attempt {_attempt + 1})...", flush=True)
            t0 = time.time()
            raw_output, usage = gemini.generate_reasoning_step(
                REASONING_PROMPT, conversation
            )
            print(f"<<< LLM returned in {time.time() - t0:.1f}s", flush=True)
            session.total_llm_calls += 1
            session.total_input_tokens += usage.input_tokens
            session.total_output_tokens += usage.output_tokens
            print(f"   (tokens: {usage.input_tokens} in, {usage.output_tokens} out)")
            if raw_output is not None and (raw_output.final_answer or raw_output.reasoning):
                step_output = raw_output
                break
            print("   [WARN] Empty/malformed response, retrying...")

        if step_output is None:
            print("   [FAIL] All retries failed, backtracking...")
            bt.handle_code_failure(session)
            continue

        if step_output.final_answer:
            print(f"\n[FINAL ANSWER] {step_output.final_answer.answer}")
            print(f"   Justification: {step_output.final_answer.justification}")
            session.final_answer = step_output.final_answer.answer
            session.justification = step_output.final_answer.justification
            break

        thought = step_output.reasoning.thought
        action = step_output.reasoning.action
        objective = step_output.reasoning.objective
        depends_on = step_output.reasoning.depends_on
        result_variable = step_output.reasoning.result_variable

        print(f"\n[THOUGHT]\n   {thought}")
        print(f"\n[ACTION (code)]\n{'-' * 40}")
        for line in action.splitlines():
            print(f"   {line}")
        print(f"{'-' * 40}")

        # Phase 2: Execute code
        print("\n[EXEC] Running code in sandbox...")
        success, observation = executor.execute(action)

        if not success:
            print(f"\n[FAIL] CODE FAILED:\n   {observation}")
            # Try repair
            print("\n[REPAIR] Attempting code repair...")
            prior_attempts = []
            current_code = action
            current_error = observation
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

                print(f"   Repair attempt {retry + 1}: {code_fix.explanation}")
                prior_attempts.append((current_code, current_error))
                current_code = code_fix.fixed_code

                success, observation = executor.execute(current_code)
                if success:
                    print(f"   [OK] Repair succeeded!")
                    action = current_code
                    repaired = True
                    break
                current_error = observation
                print(f"   [FAIL] Still failing: {current_error[:100]}")

            if not repaired:
                print("   [BACK] Code repair exhausted, backtracking...")
                bt.handle_code_failure(session)
                continue

        print(f"\n[OBSERVATION]\n{'-' * 40}")
        for line in observation.splitlines():
            print(f"   {line}")
        print(f"{'-' * 40}")

        # Phase 3: Generate inference + verification target
        ctx = build_inference_context(
            session, thought, action, observation,
            objective=objective, result_variable=result_variable,
        )
        print("\n>>> Calling LLM for inference...", flush=True)
        t0 = time.time()
        inference_step, inf_usage = gemini.generate_inference(
            INFERENCE_PROMPT, ctx
        )
        print(f"<<< LLM returned in {time.time() - t0:.1f}s", flush=True)
        session.total_llm_calls += 1
        session.total_input_tokens += inf_usage.input_tokens
        session.total_output_tokens += inf_usage.output_tokens

        print(f"\n[INFERENCE]\n   {inference_step.inference}")
        print(f"\n[VERIFICATION TARGET]")
        print(f"   Type: {inference_step.verification_target.type.value}")
        print(f"   Code:")
        for line in inference_step.verification_target.statement.splitlines():
            print(f"     {line}")

        # Phase 4: Verify
        print(f"\n[VERIFY] Running verification...")
        result = router.verify(inference_step.verification_target)
        vtype = inference_step.verification_target.type.value
        session.verification_type_counts[vtype] = (
            session.verification_type_counts.get(vtype, 0) + 1
        )

        if inference_step.verification_target.type == VerificationType.INFORMAL:
            session.informal_skip_count += 1

        if result.passed or inference_step.verification_target.type == VerificationType.INFORMAL:
            status = "PASSED" if result.passed else "SKIPPED (informal)"
            print(f"   {status}")
            bt.reset_cascade_counter()
            session.steps.append(
                CompletedStep(
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
            )
        else:
            print(f"   [FAIL] {result.error_message}")
            if result.counterexample:
                print(f"   Counterexample: {result.counterexample}")
            print("   [BACK] Backtracking (inference retry not shown in quick test)...")
            bt.handle_inference_failure(session)

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SESSION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Final answer:        {session.final_answer}")
    print(f"  Steps completed:     {len(session.steps)}")
    print(f"  LLM calls:           {session.total_llm_calls}")
    print(f"  Total input tokens:  {session.total_input_tokens:,}")
    print(f"  Total output tokens: {session.total_output_tokens:,}")
    print(f"  Backtracks:          {session.total_backtracks}")
    print(f"  Code retries:        {session.total_code_retries}")
    print(f"  Verification types:  {session.verification_type_counts}")
    print(f"  Informal skips:      {session.informal_skip_count}")

    # Cost estimate
    cost = (
        (session.total_input_tokens / 1_000_000) * 0.15
        + (session.total_output_tokens / 1_000_000) * 0.60
    )
    print(f"  Estimated cost:      ${cost:.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
