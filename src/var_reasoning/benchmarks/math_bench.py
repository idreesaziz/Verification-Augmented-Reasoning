"""MATH competition benchmark loader."""

from __future__ import annotations

import random
import re

from datasets import load_dataset

from var_reasoning.benchmarks.loader import BenchmarkLoader, Problem


class MATHLoader(BenchmarkLoader):
    """Loads MATH competition problems."""

    def load(self, num_problems: int = 200, seed: int = 42) -> list[Problem]:
        dataset = load_dataset("lighteval/MATH", "all", split="test")
        indices = list(range(len(dataset)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = indices[:num_problems]

        problems = []
        for idx in indices:
            item = dataset[idx]
            answer = self._extract_answer(item["solution"])
            difficulty = item.get("level", None)
            problems.append(
                Problem(
                    id=f"math_{idx}",
                    text=item["problem"],
                    expected_answer=answer,
                    difficulty=difficulty,
                    metadata={"type": item.get("type", "unknown")},
                )
            )
        return problems

    def _extract_answer(self, solution: str) -> str:
        """Extract the boxed answer from \\boxed{...}."""
        # Find the last \boxed{...} in the solution
        pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        matches = re.findall(pattern, solution)
        if matches:
            return matches[-1].strip()
        return solution.strip()

    def evaluate(self, problem: Problem, predicted_answer: str) -> bool:
        predicted = self._normalize_answer(predicted_answer)
        expected = self._normalize_answer(problem.expected_answer)
        return predicted == expected

    def _normalize_answer(self, s: str) -> str:
        """Normalize LaTeX math answer for comparison."""
        s = s.strip()
        # Remove \boxed{} wrapper if present
        match = re.match(r"\\boxed\{(.+)\}", s)
        if match:
            s = match.group(1)
        # Remove $ delimiters
        s = s.strip("$")
        # Normalize whitespace
        s = re.sub(r"\s+", " ", s).strip()
        # Remove \text{}, \mathrm{}, etc.
        s = re.sub(r"\\(?:text|mathrm|textbf)\{([^}]*)\}", r"\1", s)
        # Normalize common LaTeX
        s = s.replace("\\left", "").replace("\\right", "")
        s = s.replace("\\,", " ")
        # Normalize fractions: \frac{a}{b} -> a/b
        s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", s)
        # Remove trailing .0
        s = re.sub(r"\.0+$", "", s)
        try:
            val = float(s)
            if val == int(val):
                return str(int(val))
            return str(val)
        except ValueError:
            return s
