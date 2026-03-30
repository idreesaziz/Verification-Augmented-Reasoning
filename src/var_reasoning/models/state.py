"""Internal state models for the reasoning session.

VAR v2 — FactPool-based premise provenance tracking.
"""

from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field, PrivateAttr

from var_reasoning.models.schemas import (
    Derivation,
    FactType,
    Verdict,
)


# ── Fact: the atomic unit of knowledge ───────────────────────────────


class Fact(BaseModel):
    """A single fact in the reasoning chain."""

    id: str  # e.g. "given_1", "computed_3", "derived_5"
    type: FactType
    statement: str  # Human-readable: "There are 25 segments"
    source_step: Optional[int] = None  # None for GIVEN
    confidence: float = 1.0  # 1.0 for GIVEN/COMPUTED, 1/e_value for DERIVED
    depends_on: list[str] = Field(default_factory=list)  # Fact IDs
    value: Optional[float] = None  # Numeric value if applicable


# ── FactPool: growing knowledge base ─────────────────────────────────


class FactPool(BaseModel):
    """Tracks all facts across the reasoning chain."""

    facts: dict[str, Fact] = Field(default_factory=dict)
    _given_counter: int = PrivateAttr(default=0)
    _computed_counter: int = PrivateAttr(default=0)
    _derived_counter: int = PrivateAttr(default=0)

    def add_given(self, statement: str, value: float | None = None) -> Fact:
        self._given_counter += 1
        fid = f"given_{self._given_counter}"
        fact = Fact(
            id=fid,
            type=FactType.GIVEN,
            statement=statement,
            confidence=1.0,
            value=value,
        )
        self.facts[fid] = fact
        return fact

    def add_computed(
        self,
        statement: str,
        value: float | None,
        step: int,
    ) -> Fact:
        self._computed_counter += 1
        fid = f"computed_{self._computed_counter}"
        fact = Fact(
            id=fid,
            type=FactType.COMPUTED,
            statement=statement,
            source_step=step,
            confidence=1.0,
            value=value,
        )
        self.facts[fid] = fact
        return fact

    def add_derived(
        self,
        statement: str,
        value: float | None,
        step: int,
        depends_on: list[str],
        e_value: float,
    ) -> Fact:
        self._derived_counter += 1
        fid = f"derived_{self._derived_counter}"
        confidence = 1.0 - (1.0 / max(e_value, 1.0))
        fact = Fact(
            id=fid,
            type=FactType.DERIVED,
            statement=statement,
            source_step=step,
            confidence=confidence,
            depends_on=depends_on,
            value=value,
        )
        self.facts[fid] = fact
        return fact

    def get_fact(self, fid: str) -> Fact | None:
        return self.facts.get(fid)

    def compound_confidence(self, fact_ids: list[str]) -> float:
        """Product of confidences for a set of facts."""
        result = 1.0
        for fid in fact_ids:
            f = self.facts.get(fid)
            if f:
                result *= f.confidence
        return result

    def render_for_reasoner(self) -> str:
        """Format the full fact pool for the reasoner's prompt."""
        if not self.facts:
            return "(no facts established yet)"
        lines: list[str] = []
        for fid, f in self.facts.items():
            tag = f.type.value.upper()
            val_str = f"  [value: {f.value}]" if f.value is not None else ""
            lines.append(f"  {fid} [{tag}]: {f.statement}{val_str}")
        return "\n".join(lines)

    def render_for_adversary(self, fact_ids: list[str]) -> str:
        """Render only the cited facts for the adversary."""
        lines: list[str] = []
        for fid in fact_ids:
            f = self.facts.get(fid)
            if f:
                val_str = f"  [value: {f.value}]" if f.value is not None else ""
                lines.append(f"  {fid}: {f.statement}{val_str}")
        return "\n".join(lines) if lines else "(none)"


# ── FalsificationResult ──────────────────────────────────────────────


class FalsificationResult(BaseModel):
    """Result of adversarial falsification of a single derivation."""

    executed: bool = False
    verdict: Verdict = Verdict.INCONCLUSIVE
    e_value: float = 1.0
    claimed_value: Optional[float] = None
    empirical_value: Optional[float] = None
    sample_size: int = 0
    hidden_assumptions: list[str] = Field(default_factory=list)
    feedback: str = ""  # Human-readable for the reasoner on rejection


# ── CompletedStep ────────────────────────────────────────────────────


class CompletedStep(BaseModel):
    """A step that has been executed and verified."""

    step_number: int
    objective: str
    facts_used: list[str]  # Fact IDs from pool
    thought: str
    action: str  # Python code
    observation: str  # stdout
    result_variable: str
    derivations: list[Derivation] = Field(default_factory=list)
    falsification_results: list[FalsificationResult] = Field(default_factory=list)
    computed_fact_ids: list[str] = Field(default_factory=list)
    derived_fact_ids: list[str] = Field(default_factory=list)


# ── Session ──────────────────────────────────────────────────────────


class Session(BaseModel):
    """Full state of a reasoning session."""

    problem_id: str
    problem_text: str
    fact_pool: FactPool = Field(default_factory=FactPool)
    steps: list[CompletedStep] = Field(default_factory=list)
    final_answer: Optional[str] = None
    fact_chain: list[str] = Field(default_factory=list)
    compound_confidence: float = 1.0

    # Cost tracking
    total_llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_adversary_calls: int = 0
    total_falsifications: int = 0
    total_code_retries: int = 0
    total_step_retries: int = 0
    total_backtracks: int = 0
    total_rejections: int = 0
