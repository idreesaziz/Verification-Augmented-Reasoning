"""HumanEval code generation benchmark loader."""

from __future__ import annotations

import random

from datasets import load_dataset

from var_reasoning.benchmarks.loader import BenchmarkLoader, Problem
from var_reasoning.sandbox.executor import CodeExecutor


class HumanEvalLoader(BenchmarkLoader):
    """Loads HumanEval code generation problems."""

    def __init__(self, executor: CodeExecutor | None = None) -> None:
        self._executor = executor

    def load(self, num_problems: int = 200, seed: int = 42) -> list[Problem]:
        dataset = load_dataset("openai/openai_humaneval", split="test")
        indices = list(range(len(dataset)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = indices[:num_problems]

        problems = []
        for idx in indices:
            item = dataset[idx]
            problems.append(
                Problem(
                    id=item["task_id"],
                    text=item["prompt"],
                    expected_answer=item["canonical_solution"],
                    metadata={
                        "test": item["test"],
                        "entry_point": item["entry_point"],
                    },
                )
            )
        return problems

    def evaluate(self, problem: Problem, predicted_answer: str) -> bool:
        if self._executor is None:
            return False
        test_code = problem.metadata["test"]
        entry_point = problem.metadata["entry_point"]
        # Build the full test: function definition + test harness
        full_code = (
            f"{problem.text}\n"
            f"{predicted_answer}\n\n"
            f"{test_code}\n\n"
            f"check({entry_point})\n"
        )
        success, _ = self._executor.execute(full_code, timeout=30)
        return success
