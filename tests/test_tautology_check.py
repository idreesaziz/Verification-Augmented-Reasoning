"""Tests for the AST-based tautology checker."""

import pytest

from var_reasoning.verification.tautology_check import check_tautological


class TestCheckTautological:
    def test_pure_literal_comparison_is_tautological(self):
        code = "assert 6 * 10 * 20 == 1200"
        is_taut, reason = check_tautological(code)
        assert is_taut is True
        assert "hardcoded" in reason.lower() or "literal" in reason.lower()

    def test_multiple_literal_asserts_tautological(self):
        code = (
            "assert 25 * 24 / 2 == 300\n"
            "assert 300 * 17 / 36 == 141.67\n"
        )
        is_taut, _ = check_tautological(code)
        assert is_taut is True

    def test_assert_with_function_call_not_tautological(self):
        code = (
            "import math\n"
            "assert math.comb(5, 3) == 10\n"
        )
        is_taut, _ = check_tautological(code)
        assert is_taut is False

    def test_assert_referencing_variable_not_tautological(self):
        code = "assert total == 110"
        is_taut, _ = check_tautological(code)
        assert is_taut is False

    def test_assert_with_range_iteration_not_tautological(self):
        code = (
            "count = sum(1 for x in range(100, 1000) if x % 7 == 0)\n"
            "assert count == 128\n"
        )
        is_taut, _ = check_tautological(code)
        assert is_taut is False

    def test_assert_with_list_comprehension_not_tautological(self):
        code = (
            "vals = [n for n in range(100, 1000) if n == sum(int(d)**3 for d in str(n))]\n"
            "assert vals == [153, 370, 371, 407]\n"
        )
        is_taut, _ = check_tautological(code)
        assert is_taut is False

    def test_empty_code_not_tautological(self):
        is_taut, _ = check_tautological("")
        assert is_taut is False

    def test_syntax_error_not_tautological(self):
        is_taut, _ = check_tautological("assert ==== broken")
        assert is_taut is False

    def test_import_only_not_tautological(self):
        code = "from z3 import *"
        is_taut, _ = check_tautological(code)
        assert is_taut is False

    def test_mixed_literal_and_variable_not_tautological(self):
        # One assert uses a variable — not fully tautological
        code = (
            "assert total == 110\n"
            "assert 5 + 5 == 10\n"
        )
        is_taut, _ = check_tautological(code)
        assert is_taut is False

    def test_z3_code_not_tautological(self):
        code = (
            "from z3 import *\n"
            "x = Int('x')\n"
            "s = Solver()\n"
            "s.add(x > 0)\n"
            "s.add((x * (x + 1)) % 2 != 0)\n"
            "assert s.check() == unsat\n"
        )
        is_taut, _ = check_tautological(code)
        assert is_taut is False

    def test_sandbox_variable_reference_not_tautological(self):
        code = "assert len(even_numbers) == 10"
        is_taut, _ = check_tautological(code)
        assert is_taut is False
