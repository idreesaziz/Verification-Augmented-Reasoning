"""Tests for the simulation verifier."""

import pytest

from var_reasoning.verification.simulation_verifier import (
    SimulationResult,
    _parse_simulation_output,
    try_parse_numeric,
)
from var_reasoning.verification.e_value import SimulationStats


class TestParseSimulationOutput:
    def test_full_bernoulli_output(self):
        output = (
            "Running simulation...\n"
            "SIMULATION_RESULT: 0.4722\n"
            "SAMPLE_SIZE: 200000\n"
            "SAMPLE_STD: 0.4993\n"
            "SUCCESSES: 94440"
        )
        stats = _parse_simulation_output(output)
        assert stats is not None
        assert abs(stats.sample_mean - 0.4722) < 1e-6
        assert stats.sample_size == 200000
        assert abs(stats.sample_std - 0.4993) < 1e-6
        assert stats.successes == 94440

    def test_continuous_output(self):
        output = (
            "SIMULATION_RESULT: 204.3\n"
            "SAMPLE_SIZE: 100000\n"
            "SAMPLE_STD: 15.2"
        )
        stats = _parse_simulation_output(output)
        assert stats is not None
        assert abs(stats.sample_mean - 204.3) < 1e-6
        assert stats.sample_size == 100000
        assert stats.successes is None

    def test_missing_result_returns_none(self):
        output = "SAMPLE_SIZE: 100"
        assert _parse_simulation_output(output) is None

    def test_missing_size_returns_none(self):
        output = "SIMULATION_RESULT: 42.0"
        assert _parse_simulation_output(output) is None

    def test_zero_sample_size(self):
        output = "SIMULATION_RESULT: 42.0\nSAMPLE_SIZE: 0"
        assert _parse_simulation_output(output) is None

    def test_scientific_notation(self):
        output = "SIMULATION_RESULT: 1.5e-3\nSAMPLE_SIZE: 50000\nSAMPLE_STD: 2.1e-4"
        stats = _parse_simulation_output(output)
        assert stats is not None
        assert abs(stats.sample_mean - 0.0015) < 1e-8

    def test_case_insensitive(self):
        output = "simulation_result: 42.0\nsample_size: 1000"
        stats = _parse_simulation_output(output)
        assert stats is not None


class TestTryParseNumeric:
    def test_simple_integer(self):
        assert try_parse_numeric("42") == 42.0

    def test_simple_float(self):
        assert abs(try_parse_numeric("3.14159") - 3.14159) < 1e-6

    def test_with_whitespace(self):
        assert try_parse_numeric("  42  \n") == 42.0

    def test_last_line(self):
        assert try_parse_numeric("Some text\nMore text\n204") == 204.0

    def test_single_number_in_text(self):
        assert try_parse_numeric("The result is 42") == 42.0

    def test_multiple_numbers_returns_none(self):
        assert try_parse_numeric("x=10, y=20") is None

    def test_no_number_returns_none(self):
        assert try_parse_numeric("no numbers here") is None

    def test_negative_number(self):
        assert try_parse_numeric("-5.5") == -5.5

    def test_scientific_notation(self):
        result = try_parse_numeric("1.5e10")
        assert result is not None
        assert abs(result - 1.5e10) < 1e5


class TestSimulationResult:
    def test_passed_when_not_ran(self):
        r = SimulationResult(ran=False)
        assert r.passed

    def test_passed_when_accept(self):
        from var_reasoning.verification.e_value import Verdict
        r = SimulationResult(ran=True, verdict=Verdict.ACCEPT)
        assert r.passed

    def test_failed_when_reject(self):
        from var_reasoning.verification.e_value import Verdict
        r = SimulationResult(ran=True, verdict=Verdict.REJECT)
        assert not r.passed

    def test_passed_when_suspect(self):
        from var_reasoning.verification.e_value import Verdict
        r = SimulationResult(ran=True, verdict=Verdict.SUSPECT)
        assert r.passed
