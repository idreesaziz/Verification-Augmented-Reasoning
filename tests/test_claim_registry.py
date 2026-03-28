"""Tests for the claim registry."""

import pytest

from var_reasoning.verification.claim_registry import (
    ClaimRegistry,
    ConsistencyResult,
    RegisteredClaim,
)


class TestClaimRegistry:
    def test_register_and_get(self):
        reg = ClaimRegistry()
        reg.register(1, "x", 42.0, "x equals 42", [])
        assert reg.get_value("x") == 42.0

    def test_get_missing_variable(self):
        reg = ClaimRegistry()
        assert reg.get_value("missing") is None

    def test_consistent_distinct_variables(self):
        reg = ClaimRegistry()
        reg.register(1, "x", 10.0, "x is 10", [])
        reg.register(2, "y", 20.0, "y is 20", ["x"])
        result = reg.check_consistency()
        assert result.consistent

    def test_conflict_same_variable_different_values(self):
        reg = ClaimRegistry()
        reg.register(1, "x", 10.0, "x is 10", [])
        reg.register(2, "x", 20.0, "x is 20", [])
        result = reg.check_consistency()
        assert not result.consistent
        assert len(result.conflicting_steps) == 1

    def test_same_variable_same_value_ok(self):
        reg = ClaimRegistry()
        reg.register(1, "x", 42.0, "x is 42", [])
        reg.register(3, "x", 42.0, "x is still 42", [])
        result = reg.check_consistency()
        assert result.consistent

    def test_reset_clears_all(self):
        reg = ClaimRegistry()
        reg.register(1, "x", 5.0, "x is 5", [])
        reg.reset()
        assert reg.get_value("x") is None
        assert reg.claims == []

    def test_all_observations(self):
        reg = ClaimRegistry()
        reg.register(1, "a", 1.0, "a=1", [])
        reg.register(2, "b", 2.0, "b=2", [])
        obs = reg.all_observations()
        assert obs == {"a": 1.0, "b": 2.0}
