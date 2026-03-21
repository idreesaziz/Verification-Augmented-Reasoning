"""Experimental conditions definition."""

from __future__ import annotations

from enum import Enum


class Condition(str, Enum):
    A = "A"  # Baseline: one-shot, no tools
    B = "B"  # Code execution, no verification
    C = "C"  # Full VAR
    D = "D"  # Ceiling: Gemini 2.5 Pro, one-shot


CONDITION_DESCRIPTIONS = {
    Condition.A: "Baseline: Gemini 2.5 Flash, one-shot, no tools",
    Condition.B: "Code execution: Gemini 2.5 Flash with ReAct loop, no verification",
    Condition.C: "Full VAR: Gemini 2.5 Flash with code execution + verification + backtracking",
    Condition.D: "Ceiling: Gemini 2.5 Pro, one-shot, no tools",
}

ONE_SHOT_SYSTEM_PROMPT = """\
Solve the following problem. Show your reasoning step by step, then
provide your final answer.

For math problems, give ONLY the numeric answer on the last line.
For logic problems, answer only True, False, or Unknown on the last line.
For code problems, provide only the function body.

Final answer:
"""
