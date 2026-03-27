"""AST-based tautology detector for verification statements.

Catches verification code that cannot possibly fail because it does
no independent computation — only manipulates hardcoded constants.

Flagged as tautological if:
  1. Every assert compares expressions built from numeric literals only.
  2. The code has no function calls and no variable references.
"""

from __future__ import annotations

import ast


def _is_pure_literal(node: ast.expr) -> bool:
    """Return True if the expression tree is only numeric literals and operators."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex)):
        return True
    if isinstance(node, ast.UnaryOp):
        return _is_pure_literal(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_pure_literal(node.left) and _is_pure_literal(node.right)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_pure_literal(e) for e in node.elts)
    return False


def _has_function_call(tree: ast.AST) -> bool:
    """Return True if the AST contains any function call."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            return True
    return False


def _has_name_reference(tree: ast.AST, skip_imports: bool = True) -> bool:
    """Return True if the AST references any variable name (not just literals)."""
    import_names: set[str] = set()
    if skip_imports:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_names.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        import_names.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in import_names:
            return True
    return False


def _all_asserts_literal_only(tree: ast.Module) -> bool:
    """Return True if every assert compares only literal expressions."""
    asserts = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    if not asserts:
        return False
    for a in asserts:
        test = a.test
        if isinstance(test, ast.Compare):
            parts = [test.left] + test.comparators
            if not all(_is_pure_literal(p) for p in parts):
                return False
        elif not _is_pure_literal(test):
            return False
    return True


def check_tautological(statement: str) -> tuple[bool, str]:
    """Check whether a verification statement is tautological.

    Returns (is_tautological, reason).
    """
    try:
        tree = ast.parse(statement)
    except SyntaxError:
        return False, ""

    asserts = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    if not asserts:
        return False, ""

    # Pattern 1: All asserts compare only numeric literals
    if _all_asserts_literal_only(tree):
        return True, (
            "Verification only compares hardcoded numeric literals — it "
            "can never fail. Reference sandbox variables or call functions "
            "to independently check the claim."
        )

    # Pattern 2: No function calls and no variable references
    if not _has_function_call(tree) and not _has_name_reference(tree):
        return True, (
            "Verification has no function calls and no variable references. "
            "It cannot independently check anything."
        )

    return False, ""
