"""Abstract benchmark loader interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Problem:
    id: str
    text: str
    expected_answer: str
    difficulty: str | None = None
    metadata: dict | None = None


class BenchmarkLoader(ABC):
    """Base class for benchmark dataset loaders."""

    @abstractmethod
    def load(self, num_problems: int = 200, seed: int = 42) -> list[Problem]:
        """Load problems from the dataset."""
        ...

    @abstractmethod
    def evaluate(self, problem: Problem, predicted_answer: str) -> bool:
        """Check if the predicted answer is correct."""
        ...
