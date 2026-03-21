"""GSM8K benchmark loader."""

from __future__ import annotations

import random
import re

from datasets import load_dataset

from var_reasoning.benchmarks.loader import BenchmarkLoader, Problem


class GSM8KLoader(BenchmarkLoader):
    """Loads GSM8K grade school math problems."""

    def load(self, num_problems: int = 200, seed: int = 42) -> list[Problem]:
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        indices = list(range(len(dataset)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = indices[:num_problems]

        problems = []
        for idx in indices:
            item = dataset[idx]
            answer = self._extract_answer(item["answer"])
            problems.append(
                Problem(
                    id=f"gsm8k_{idx}",
                    text=item["question"],
                    expected_answer=answer,
                )
            )
        return problems

    def _extract_answer(self, answer_text: str) -> str:
        """Extract numeric answer from '#### N' format."""
        match = re.search(r"####\s*(.+)", answer_text)
        if match:
            return match.group(1).strip().replace(",", "")
        return answer_text.strip()

    def evaluate(self, problem: Problem, predicted_answer: str) -> bool:
        predicted = self._normalize_number(predicted_answer)
        expected = self._normalize_number(problem.expected_answer)
        return predicted == expected

    def _normalize_number(self, s: str) -> str:
        s = s.strip().replace(",", "").replace("$", "").replace("%", "")
        # Remove trailing .0 or .00
        s = re.sub(r"\.0+$", "", s)
        try:
            val = float(s)
            if val == int(val):
                return str(int(val))
            return str(val)
        except ValueError:
            return s
