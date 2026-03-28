"""Tests for static sanity gates."""

import pytest

from var_reasoning.verification.static_sanity import (
    SanityResult,
    check_integrality,
    check_no_claimed_value_in_code,
    check_provenance,
    check_range,
)


class TestCheckRange:
    def test_nan_rejected(self):
        r = check_range(float("nan"), "algebraic")
        assert not r.passed

    def test_inf_rejected(self):
        r = check_range(float("inf"), "product_rule")
        assert not r.passed

    def test_normal_value_accepted(self):
        r = check_range(42.0, "algebraic")
        assert r.passed

    def test_negative_count_rejected(self):
        r = check_range(-5.0, "exhaustive_enumeration")
        assert not r.passed


class TestCheckIntegrality:
    def test_integer_aime_accepted(self):
        r = check_integrality(204.0, "AIME 2025 Problem 13", is_final_answer=True)
        assert r.passed

    def test_non_integer_aime_rejected(self):
        r = check_integrality(162.33, "AIME 2025 Problem 13", is_final_answer=True)
        assert not r.passed

    def test_non_final_always_passes(self):
        r = check_integrality(3.14, "AIME problem", is_final_answer=False)
        assert r.passed

    def test_non_aime_float_accepted(self):
        r = check_integrality(3.14, "Find the expected value", is_final_answer=True)
        assert r.passed


class TestCheckProvenance:
    def test_trivial_values_accepted(self):
        code = "x = 2 + 3"
        r = check_provenance(code, "Unused problem text", [])
        assert r.passed

    def test_problem_number_accepted(self):
        code = "x = 25 + 2"
        r = check_provenance(code, "There are 25 line segments", [])
        assert r.passed

    def test_smuggled_constant_rejected(self):
        code = "p = 17/36"
        r = check_provenance(code, "There are 25 segments", [])
        assert not r.passed
        assert "17" in r.reason

    def test_observation_number_accepted(self):
        code = "result = 204"
        r = check_provenance(code, "Some problem", ["The count is 204"])
        assert r.passed

    def test_syntax_error_code_accepted(self):
        code = "x = a b c def"
        r = check_provenance(code, "problem", [])
        assert r.passed  # Can't parse → no literals found → passes


class TestCheckNoClaimedValue:
    def test_trivial_values_ok(self):
        r = check_no_claimed_value_in_code("x = 2 + 1", 2.0)
        assert r.passed

    def test_claimed_value_in_code_rejected(self):
        r = check_no_claimed_value_in_code("p = 0.333333", 0.333333)
        assert not r.passed
        assert "CIRCULAR" in r.reason

    def test_different_value_accepted(self):
        r = check_no_claimed_value_in_code("p = 0.5", 0.333)
        assert r.passed
