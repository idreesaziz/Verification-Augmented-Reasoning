"""Token/cost tracking for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

# Pricing per million tokens
PRICING: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {
        "input_per_million": 0.15,
        "output_per_million": 0.60,
    },
    "gemini-2.5-pro": {
        "input_per_million": 1.25,
        "output_per_million": 10.00,
    },
}


@dataclass
class CostEntry:
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def cost_usd(self) -> float:
        rates = PRICING.get(self.model, PRICING["gemini-2.5-flash"])
        input_cost = (self.input_tokens / 1_000_000) * rates["input_per_million"]
        output_cost = (self.output_tokens / 1_000_000) * rates["output_per_million"]
        return input_cost + output_cost


@dataclass
class CostTracker:
    model: str = "gemini-2.5-flash"
    entries: list[CostEntry] = field(default_factory=list)

    def record(self, input_tokens: int, output_tokens: int) -> CostEntry:
        entry = CostEntry(
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.entries.append(entry)
        return entry

    @property
    def total_input_tokens(self) -> int:
        return sum(e.input_tokens for e in self.entries)

    @property
    def total_output_tokens(self) -> int:
        return sum(e.output_tokens for e in self.entries)

    @property
    def total_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    def reset(self) -> None:
        self.entries.clear()

    def summary(self) -> dict:
        return {
            "model": self.model,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "num_calls": len(self.entries),
        }
