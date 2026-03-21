"""Tests for cost tracking."""

import pytest

from var_reasoning.experiment.cost_tracker import CostEntry, CostTracker, PRICING


class TestCostEntry:
    def test_flash_cost(self):
        entry = CostEntry(
            model="gemini-2.5-flash",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        # $0.15 input + $0.60 output = $0.75
        assert abs(entry.cost_usd - 0.75) < 1e-6

    def test_pro_cost(self):
        entry = CostEntry(
            model="gemini-2.5-pro",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        # $1.25 input + $10.00 output = $11.25
        assert abs(entry.cost_usd - 11.25) < 1e-6

    def test_zero_tokens(self):
        entry = CostEntry(model="gemini-2.5-flash", input_tokens=0, output_tokens=0)
        assert entry.cost_usd == 0.0

    def test_small_token_count(self):
        entry = CostEntry(
            model="gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=500,
        )
        expected = (1000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
        assert abs(entry.cost_usd - expected) < 1e-10

    def test_unknown_model_uses_flash_rates(self):
        entry = CostEntry(
            model="unknown-model",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        )
        # Should fall back to flash rates
        assert abs(entry.cost_usd - 0.75) < 1e-6


class TestCostTracker:
    def test_record_and_totals(self):
        tracker = CostTracker(model="gemini-2.5-flash")
        tracker.record(input_tokens=500_000, output_tokens=200_000)
        tracker.record(input_tokens=300_000, output_tokens=100_000)

        assert tracker.total_input_tokens == 800_000
        assert tracker.total_output_tokens == 300_000

        expected_cost = (
            (800_000 / 1_000_000) * 0.15 + (300_000 / 1_000_000) * 0.60
        )
        assert abs(tracker.total_cost_usd - expected_cost) < 1e-6

    def test_reset(self):
        tracker = CostTracker(model="gemini-2.5-flash")
        tracker.record(input_tokens=1000, output_tokens=500)
        tracker.reset()
        assert tracker.total_input_tokens == 0
        assert tracker.total_output_tokens == 0
        assert tracker.total_cost_usd == 0.0
        assert len(tracker.entries) == 0

    def test_summary(self):
        tracker = CostTracker(model="gemini-2.5-pro")
        tracker.record(input_tokens=1_000_000, output_tokens=500_000)
        summary = tracker.summary()
        assert summary["model"] == "gemini-2.5-pro"
        assert summary["total_input_tokens"] == 1_000_000
        assert summary["total_output_tokens"] == 500_000
        assert summary["num_calls"] == 1
        expected = (1_000_000 / 1_000_000) * 1.25 + (500_000 / 1_000_000) * 10.00
        assert abs(summary["total_cost_usd"] - round(expected, 6)) < 1e-4

    def test_pricing_rates_correct(self):
        assert PRICING["gemini-2.5-flash"]["input_per_million"] == 0.15
        assert PRICING["gemini-2.5-flash"]["output_per_million"] == 0.60
        assert PRICING["gemini-2.5-pro"]["input_per_million"] == 1.25
        assert PRICING["gemini-2.5-pro"]["output_per_million"] == 10.00
