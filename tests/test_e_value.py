"""Tests for the e-value statistical engine."""

import math

import pytest

from var_reasoning.verification.e_value import (
    E_REJECT,
    E_SUSPECT,
    EValueResult,
    SimulationStats,
    Verdict,
    exact_e_value,
    mean_e_value,
    proportion_e_value,
    test_claim as run_test_claim,
)


class TestProportionEValue:
    def test_exact_match_returns_one(self):
        """When p̂ == p₀ exactly, e-value should be 1 (no evidence)."""
        e = proportion_e_value(successes=333, trials=1000, p0=0.333)
        assert math.isclose(e, 1.0, rel_tol=0.01)

    def test_strong_mismatch_gives_high_e(self):
        """p₀ = 1/3 but empirical is ~0.47 → strong rejection."""
        e = proportion_e_value(successes=47000, trials=100000, p0=1 / 3)
        assert e > E_REJECT

    def test_slight_deviation_low_e(self):
        """Small deviation from p₀ with few samples → weak evidence."""
        e = proportion_e_value(successes=35, trials=100, p0=0.333)
        assert e < E_REJECT

    def test_zero_trials(self):
        e = proportion_e_value(successes=0, trials=0, p0=0.5)
        assert e == 1.0

    def test_extreme_p0_near_zero(self):
        e = proportion_e_value(successes=50000, trials=100000, p0=0.001)
        assert e > E_REJECT

    def test_extreme_p0_near_one(self):
        e = proportion_e_value(successes=1000, trials=100000, p0=0.999)
        assert e > E_REJECT


class TestMeanEValue:
    def test_exact_match(self):
        e = mean_e_value(sample_mean=42.0, sample_std=5.0, sample_size=10000, mu0=42.0)
        assert math.isclose(e, 1.0, abs_tol=0.01)

    def test_large_deviation(self):
        """Mean = 204 but claim is 162 → strong rejection."""
        e = mean_e_value(sample_mean=204.0, sample_std=15.0, sample_size=10000, mu0=162.0)
        assert e > E_REJECT

    def test_small_sample_weak_evidence(self):
        e = mean_e_value(sample_mean=205.0, sample_std=50.0, sample_size=10, mu0=200.0)
        assert e < E_REJECT

    def test_degenerate_std(self):
        e = mean_e_value(sample_mean=42.0, sample_std=0.0, sample_size=100, mu0=42.0)
        assert e == 1.0


class TestExactEValue:
    def test_match(self):
        assert exact_e_value(42.0, 42.0) == 1.0

    def test_close_match(self):
        assert exact_e_value(42.0, 42.0 + 1e-12) == 1.0

    def test_mismatch(self):
        assert exact_e_value(42.0, 43.0) == float("inf")


class TestTestClaim:
    def test_proportion_accept(self):
        stats = SimulationStats(
            sample_mean=0.333, sample_size=100000, sample_std=0.0, successes=33300,
        )
        result = run_test_claim(claimed_value=1 / 3, stats=stats)
        assert result.verdict == Verdict.ACCEPT

    def test_proportion_reject(self):
        stats = SimulationStats(
            sample_mean=0.472, sample_size=200000, sample_std=0.0, successes=94400,
        )
        result = run_test_claim(claimed_value=1 / 3, stats=stats)
        assert result.verdict == Verdict.REJECT
        assert result.e_value > E_REJECT

    def test_continuous_accept(self):
        stats = SimulationStats(
            sample_mean=204.1, sample_size=50000, sample_std=15.0,
        )
        result = run_test_claim(claimed_value=204.0, stats=stats)
        assert result.verdict == Verdict.ACCEPT

    def test_continuous_reject(self):
        stats = SimulationStats(
            sample_mean=204.0, sample_size=50000, sample_std=15.0,
        )
        result = run_test_claim(claimed_value=162.0, stats=stats)
        assert result.verdict == Verdict.REJECT

    def test_deterministic_match(self):
        stats = SimulationStats(sample_mean=42.0, sample_size=1)
        result = run_test_claim(claimed_value=42.0, stats=stats)
        assert result.verdict == Verdict.ACCEPT

    def test_deterministic_mismatch(self):
        stats = SimulationStats(sample_mean=42.0, sample_size=1)
        result = run_test_claim(claimed_value=43.0, stats=stats)
        assert result.verdict == Verdict.REJECT

    def test_result_has_all_fields(self):
        stats = SimulationStats(
            sample_mean=0.5, sample_size=1000, sample_std=0.0, successes=500,
        )
        result = run_test_claim(claimed_value=0.5, stats=stats)
        assert isinstance(result, EValueResult)
        assert result.claimed_value == 0.5
        assert result.sample_size == 1000
        assert len(result.confidence_interval) == 2
