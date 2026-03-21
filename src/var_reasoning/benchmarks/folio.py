"""FOLIO first-order logic benchmark loader."""

from __future__ import annotations

import random

from datasets import load_dataset

from var_reasoning.benchmarks.loader import BenchmarkLoader, Problem


class FOLIOLoader(BenchmarkLoader):
    """Loads FOLIO logic inference problems."""

    def load(self, num_problems: int = 200, seed: int = 42) -> list[Problem]:
        dataset = load_dataset("yale-nlp/FOLIO", split="validation")
        indices = list(range(len(dataset)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = indices[:num_problems]

        problems = []
        for idx in indices:
            item = dataset[idx]
            premises = item.get("premises", "")
            conclusion = item.get("conclusion", "")
            text = (
                f"Premises:\n{premises}\n\n"
                f"Conclusion:\n{conclusion}\n\n"
                f"Is the conclusion True, False, or Unknown given the premises?"
            )
            problems.append(
                Problem(
                    id=f"folio_{idx}",
                    text=text,
                    expected_answer=item["label"],
                )
            )
        return problems

    def evaluate(self, problem: Problem, predicted_answer: str) -> bool:
        predicted = predicted_answer.strip().lower()
        expected = problem.expected_answer.strip().lower()
        return predicted == expected
