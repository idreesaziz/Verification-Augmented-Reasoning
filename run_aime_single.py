"""Focused test: AIME 2025 I Problem 13 (expected answer: 204).

VAR v2 — premise provenance + adversarial falsification.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import logging

from dotenv import load_dotenv
load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: Set GEMINI_API_KEY in .env or environment first")
    sys.exit(1)

from var_reasoning.engine.backtracking import BacktrackManager
from var_reasoning.engine.step_engine import StepEngine
from var_reasoning.models.gemini_provider import GeminiProvider
from var_reasoning.models.schemas import Verdict


# ── Local executor ──────────────────────────────────────────────────
class LocalExecutor:
    """Subprocess-based code executor with cumulative namespace."""

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
    # Set up logging so we see step_engine's INFO messages
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    output_lines: list[str] = []

    def log(msg=""):
        print(msg, flush=True)
        output_lines.append(msg)

    gemini = GeminiProvider()
    executor = LocalExecutor()
    bt = BacktrackManager(
        code_retries=3,
        inference_retries=5,
        cascade_limit=3,
        total_backtrack_limit=20,
        max_steps=25,
    )

    pid = PROBLEM["id"]
    log(f"{'='*70}")
    log(f"PROBLEM: {pid}")
    log(f"Expected: {PROBLEM['expected']}")
    log(f"{'='*70}")
    log(f"Q: {PROBLEM['text']}")
    log("")

    total_start = time.time()

    # ── Run the engine ──────────────────────────────────────────────
    engine = StepEngine(gemini=gemini, executor=executor, backtrack_manager=bt)
    session = engine.solve(problem_id=pid, problem_text=PROBLEM["text"])

    total_elapsed = time.time() - total_start

    # ── Report ──────────────────────────────────────────────────────
    correct = _normalize(session.final_answer) == _normalize(PROBLEM["expected"])

    log(f"\n{'='*70}")
    log(f"RESULT: {'CORRECT' if correct else '*** WRONG ***'}")
    log(f"predicted={session.final_answer}  expected={PROBLEM['expected']}")
    log(f"steps={len(session.steps)} backtracks={session.total_backtracks} "
        f"step_retries={session.total_step_retries} "
        f"code_retries={session.total_code_retries}")
    log(f"adversary_calls={session.total_adversary_calls} "
        f"falsifications={session.total_falsifications} "
        f"rejections={session.total_rejections}")
    log(f"compound_confidence={session.compound_confidence:.6f}")
    log(f"llm_calls={session.total_llm_calls}")
    cost = (
        (session.total_input_tokens / 1_000_000) * 0.15
        + (session.total_output_tokens / 1_000_000) * 0.60
    )
    log(f"cost=${cost:.4f} time={total_elapsed:.1f}s")

    # Fact pool dump
    log(f"\n--- FACT POOL ({len(session.fact_pool.facts)} facts) ---")
    for fid, f in session.fact_pool.facts.items():
        tag = f.type.value.upper()
        dep = f"  deps={f.depends_on}" if f.depends_on else ""
        log(f"  {fid} [{tag}] conf={f.confidence:.4f}: {f.statement[:100]}{dep}")

    # Step detail
    log(f"\n--- STEP DETAIL ---")
    for step in session.steps:
        log(f"  Step {step.step_number}: {step.objective[:80]}")
        log(f"    observation: {step.observation[:120]}")
        log(f"    computed: {step.computed_fact_ids}")
        log(f"    derived: {step.derived_fact_ids}")
        for i, fr in enumerate(step.falsification_results):
            d = step.derivations[i] if i < len(step.derivations) else None
            prem = d.premise[:60] if d else "?"
            log(f"    falsification[{i}]: {fr.verdict.value} e={fr.e_value:.2f} "
                f"— {prem}")

    if session.fact_chain:
        log(f"\n--- FACT CHAIN ---")
        for fid in session.fact_chain:
            f = session.fact_pool.get_fact(fid)
            if f:
                log(f"  {fid}: {f.statement[:100]}")

    log(f"\n{'='*70}")

    # Save output
    with open("aime_v2_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    log("Saved to aime_v2_output.txt")


if __name__ == "__main__":
    run()
