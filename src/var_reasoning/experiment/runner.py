"""Experiment runner: executes conditions on benchmarks."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from var_reasoning.benchmarks.loader import BenchmarkLoader, Problem
from var_reasoning.engine.step_engine import ConditionBEngine, StepEngine
from var_reasoning.experiment.conditions import (
    ONE_SHOT_SYSTEM_PROMPT,
    Condition,
)
from var_reasoning.experiment.cost_tracker import CostTracker, PRICING
from var_reasoning.models.gemini_provider import GeminiProvider
from var_reasoning.sandbox.executor import CodeExecutor
from var_reasoning.verification.verification_router import VerificationRouter

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs a single experimental condition on a benchmark."""

    def __init__(
        self,
        condition: Condition,
        benchmark_loader: BenchmarkLoader,
        num_problems: int = 200,
        output_dir: str = "results",
        seed: int = 42,
    ) -> None:
        self._condition = condition
        self._loader = benchmark_loader
        self._num_problems = num_problems
        self._output_dir = Path(output_dir)
        self._seed = seed
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _get_output_path(self, benchmark_name: str) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return self._output_dir / f"{self._condition.value}_{benchmark_name}_{ts}.jsonl"

    def _load_existing_results(self, output_path: Path) -> set[str]:
        """Load IDs of already-completed problems for resume."""
        done: set[str] = set()
        if output_path.exists():
            with open(output_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        done.add(data["problem_id"])
        return done

    def _compute_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        rates = PRICING.get(model, PRICING["gemini-2.5-flash"])
        return (input_tokens / 1_000_000) * rates[
            "input_per_million"
        ] + (output_tokens / 1_000_000) * rates["output_per_million"]

    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer from one-shot LLM output."""
        lines = text.strip().splitlines()
        # Return the last non-empty line
        for line in reversed(lines):
            line = line.strip()
            if line:
                # Remove common prefixes
                line = re.sub(
                    r"^(?:final\s*answer\s*[:=]?\s*|the\s+answer\s+is\s*[:=]?\s*)",
                    "",
                    line,
                    flags=re.IGNORECASE,
                )
                return line.strip()
        return text.strip()

    def run(self, benchmark_name: str, resume_path: Path | None = None) -> Path:
        """Run the experiment. Returns path to result file."""
        problems = self._loader.load(self._num_problems, self._seed)
        output_path = resume_path or self._get_output_path(benchmark_name)
        done_ids = self._load_existing_results(output_path)

        remaining = [p for p in problems if p.id not in done_ids]
        if not remaining:
            logger.info("All problems already completed in %s", output_path)
            return output_path

        logger.info(
            "Running condition %s on %s: %d problems (%d already done)",
            self._condition.value,
            benchmark_name,
            len(remaining),
            len(done_ids),
        )

        if self._condition in (Condition.A, Condition.D):
            self._run_one_shot(remaining, output_path, benchmark_name)
        elif self._condition == Condition.B:
            self._run_condition_b(remaining, output_path)
        elif self._condition == Condition.C:
            self._run_condition_c(remaining, output_path)

        return output_path

    def _run_one_shot(
        self, problems: list[Problem], output_path: Path, benchmark_name: str
    ) -> None:
        model = "gemini-2.5-flash"
        model_override = None
        if self._condition == Condition.D:
            model = "gemini-2.5-pro"
            model_override = "gemini-2.5-pro"

        gemini = GeminiProvider(model_name=model)

        with open(output_path, "a", encoding="utf-8") as f:
            for problem in tqdm(problems, desc=f"Condition {self._condition.value}"):
                gemini.reset_usage()
                answer_text, usage = gemini.generate_one_shot(
                    problem.text,
                    ONE_SHOT_SYSTEM_PROMPT,
                    model_override=model_override,
                )
                predicted = self._extract_final_answer(answer_text)
                is_correct = self._loader.evaluate(problem, predicted)
                cost = self._compute_cost(
                    model, usage.input_tokens, usage.output_tokens
                )

                result = {
                    "problem_id": problem.id,
                    "problem_text": problem.text,
                    "predicted_answer": predicted,
                    "correct_answer": problem.expected_answer,
                    "is_correct": is_correct,
                    "difficulty": problem.difficulty,
                    "num_steps": 0,
                    "total_backtracks": 0,
                    "total_code_retries": 0,
                    "total_inference_retries": 0,
                    "total_llm_calls": 1,
                    "total_input_tokens": usage.input_tokens,
                    "total_output_tokens": usage.output_tokens,
                    "cost_usd": round(cost, 6),
                    "verification_type_counts": {},
                    "reasoning_pattern_counts": {},
                    "informal_skip_count": 0,
                    "informal_without_reason_count": 0,
                }
                f.write(json.dumps(result) + "\n")
                f.flush()

    def _run_condition_b(
        self, problems: list[Problem], output_path: Path
    ) -> None:
        gemini = GeminiProvider(model_name="gemini-2.5-flash")
        executor = CodeExecutor()
        engine = ConditionBEngine(gemini=gemini, executor=executor)

        with open(output_path, "a", encoding="utf-8") as f:
            for problem in tqdm(problems, desc="Condition B"):
                session = engine.solve(problem.id, problem.text)
                predicted = session.final_answer or ""
                is_correct = self._loader.evaluate(problem, predicted)
                cost = self._compute_cost(
                    "gemini-2.5-flash",
                    session.total_input_tokens,
                    session.total_output_tokens,
                )

                result = {
                    "problem_id": problem.id,
                    "problem_text": problem.text,
                    "predicted_answer": predicted,
                    "correct_answer": problem.expected_answer,
                    "is_correct": is_correct,
                    "difficulty": problem.difficulty,
                    "num_steps": len(session.steps),
                    "total_backtracks": session.total_backtracks,
                    "total_code_retries": session.total_code_retries,
                    "total_inference_retries": session.total_inference_retries,
                    "total_llm_calls": session.total_llm_calls,
                    "total_input_tokens": session.total_input_tokens,
                    "total_output_tokens": session.total_output_tokens,
                    "cost_usd": round(cost, 6),
                    "verification_type_counts": session.verification_type_counts,
                    "reasoning_pattern_counts": session.reasoning_pattern_counts,
                    "informal_skip_count": session.informal_skip_count,
                    "informal_without_reason_count": session.informal_without_reason_count,
                }
                f.write(json.dumps(result) + "\n")
                f.flush()

    def _run_condition_c(
        self, problems: list[Problem], output_path: Path
    ) -> None:
        gemini = GeminiProvider(model_name="gemini-2.5-flash")
        executor = CodeExecutor()
        router = VerificationRouter(executor)
        engine = StepEngine(
            gemini=gemini, executor=executor, router=router
        )

        with open(output_path, "a", encoding="utf-8") as f:
            for problem in tqdm(problems, desc="Condition C"):
                session = engine.solve(problem.id, problem.text)
                predicted = session.final_answer or ""
                is_correct = self._loader.evaluate(problem, predicted)
                cost = self._compute_cost(
                    "gemini-2.5-flash",
                    session.total_input_tokens,
                    session.total_output_tokens,
                )

                result = {
                    "problem_id": problem.id,
                    "problem_text": problem.text,
                    "predicted_answer": predicted,
                    "correct_answer": problem.expected_answer,
                    "is_correct": is_correct,
                    "difficulty": problem.difficulty,
                    "num_steps": len(session.steps),
                    "total_backtracks": session.total_backtracks,
                    "total_code_retries": session.total_code_retries,
                    "total_inference_retries": session.total_inference_retries,
                    "total_llm_calls": session.total_llm_calls,
                    "total_input_tokens": session.total_input_tokens,
                    "total_output_tokens": session.total_output_tokens,
                    "cost_usd": round(cost, 6),
                    "verification_type_counts": session.verification_type_counts,
                    "reasoning_pattern_counts": session.reasoning_pattern_counts,
                    "informal_skip_count": session.informal_skip_count,
                    "informal_without_reason_count": session.informal_without_reason_count,
                }
                f.write(json.dumps(result) + "\n")
                f.flush()
