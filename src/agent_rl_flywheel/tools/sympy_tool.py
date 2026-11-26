"""AST-sandboxed SymPy evaluator and symbolic equivalence checker."""

from __future__ import annotations

import ast
import re
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)
_ALLOWED_NAMES = frozenset(
    {
        "x",
        "y",
        "z",
        "t",
        "n",
        "k",
        "a",
        "b",
        "c",
        "pi",
        "E",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "log",
        "exp",
        "Abs",
        "Rational",
        "simplify",
        "expand",
        "factor",
        "solve",
        "diff",
        "integrate",
        "Eq",
    }
)
_ALLOWED_FUNCS = frozenset(
    {
        "sqrt",
        "sin",
        "cos",
        "tan",
        "log",
        "exp",
        "Abs",
        "Rational",
        "simplify",
        "expand",
        "factor",
        "solve",
        "diff",
        "integrate",
        "Eq",
    }
)

_SYMPY_LOCALS: dict[str, Any] = {
    "x": sp.Symbol("x"),
    "y": sp.Symbol("y"),
    "z": sp.Symbol("z"),
    "t": sp.Symbol("t"),
    "n": sp.Symbol("n"),
    "k": sp.Symbol("k"),
    "a": sp.Symbol("a"),
    "b": sp.Symbol("b"),
    "c": sp.Symbol("c"),
    "pi": sp.pi,
    "E": sp.E,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "log": sp.log,
    "exp": sp.exp,
    "Abs": sp.Abs,
    "Rational": sp.Rational,
    "simplify": sp.simplify,
    "expand": sp.expand,
    "factor": sp.factor,
    "solve": sp.solve,
    "diff": sp.diff,
    "integrate": sp.integrate,
    "Eq": sp.Eq,
}

_ANSWER_EXTRACT_RE = re.compile(r"(?:answer\s*[:=]|最终答案\s*[:：]|\\boxed\{)([^}\n;；]+)", re.IGNORECASE)


class UnsafeExpressionError(ValueError):
    """Raised when an expression violates the SymPy AST sandbox policy."""


def _validate_ast(node: ast.AST, max_exponent: int = 200) -> None:
    """Recursively validate an expression AST against the safe math whitelist."""
    if isinstance(node, ast.Expression):
        _validate_ast(node.body, max_exponent=max_exponent)
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return
        raise UnsafeExpressionError(f"Unsupported constant type: {type(node.value).__name__}")
    if isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES:
            raise UnsafeExpressionError(f"Disallowed identifier: {node.id}")
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
        _validate_ast(node.operand, max_exponent=max_exponent)
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        _validate_ast(node.left, max_exponent=max_exponent)
        _validate_ast(node.right, max_exponent=max_exponent)
        if isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and abs(float(node.right.value)) > max_exponent:
                raise UnsafeExpressionError(f"Exponent exceeds limit {max_exponent}")
            if isinstance(node.right, ast.UnaryOp) and isinstance(node.right.operand, ast.Constant):
                if abs(float(node.right.operand.value)) > max_exponent:
                    raise UnsafeExpressionError(f"Exponent exceeds limit {max_exponent}")
            if isinstance(node.right, ast.BinOp) and isinstance(node.right.op, ast.Pow):
                raise UnsafeExpressionError("Nested power towers are disallowed")
        return
    if isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            _validate_ast(elt, max_exponent=max_exponent)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise UnsafeExpressionError("Only whitelisted math function calls are allowed")
        if node.keywords:
            raise UnsafeExpressionError("Keyword arguments are disallowed in sandboxed calls")
        for arg in node.args:
            _validate_ast(arg, max_exponent=max_exponent)
        return
    raise UnsafeExpressionError(f"Disallowed AST node: {type(node).__name__}")


_SYMPY_GLOBALS: dict[str, Any] = {
    "__builtins__": {},
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Symbol": sp.Symbol,
}


def _safe_parse(expr: str, max_len: int = 300, max_exponent: int = 200) -> Any:
    """Validate and parse a mathematical expression without arbitrary code execution."""
    cleaned = expr.strip().replace("^", "**")
    if not cleaned:
        raise UnsafeExpressionError("Empty expression")
    if len(cleaned) > max_len:
        raise UnsafeExpressionError(f"Expression exceeds max length {max_len}")
    if "__" in cleaned or ";" in cleaned:
        raise UnsafeExpressionError("Disallowed token in expression")

    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Syntax error: {exc.msg}") from exc
    _validate_ast(tree, max_exponent=max_exponent)
    return parse_expr(cleaned, local_dict=_SYMPY_LOCALS, global_dict=_SYMPY_GLOBALS, evaluate=True)


def _extract_math_candidate(text: str) -> str:
    """Extract numeric or algebraic expression candidate from an answer string."""
    stripped = text.strip()
    match = _ANSWER_EXTRACT_RE.search(stripped)
    if match:
        return match.group(1).strip().rstrip(".,;。，；")
    # Strip trailing citation tag e.g. "42 [doc_1]" or "42 (doc_1)"
    cleaned = re.sub(r"[\[(（]doc_[a-zA-Z0-9_-]+[\])）]", "", stripped).strip()
    cleaned = re.sub(r"\bdoc_[a-zA-Z0-9_-]+\b", "", cleaned).strip()
    cleaned = cleaned.rstrip(".,;。，；").strip()
    return cleaned


class SymPyTool:
    """Sandboxed SymPy evaluator and symbolic/numeric equivalence verifier."""

    def __init__(self, max_len: int = 300, max_exponent: int = 200) -> None:
        self.max_len = max_len
        self.max_exponent = max_exponent

    def evaluate(self, expr: str) -> dict[str, Any]:
        """Evaluate a safe math expression and return normalized string and numeric float if finite."""
        try:
            obj = _safe_parse(expr, max_len=self.max_len, max_exponent=self.max_exponent)
            simplified = sp.simplify(obj) if isinstance(obj, sp.Basic) else obj
            numeric: float | None = None
            if isinstance(simplified, sp.Basic) and not simplified.free_symbols:
                try:
                    cval = complex(simplified.evalf(15))
                    if abs(cval.imag) < 1e-12:
                        numeric = float(cval.real)
                except (TypeError, ValueError):
                    numeric = None
            elif isinstance(simplified, list) and len(simplified) == 1 and isinstance(simplified[0], sp.Basic):
                try:
                    cval = complex(simplified[0].evalf(15))
                    if abs(cval.imag) < 1e-12:
                        numeric = float(cval.real)
                except (TypeError, ValueError):
                    numeric = None
            return {
                "ok": True,
                "result": str(simplified),
                "numeric": numeric,
            }
        except Exception as exc:  # noqa: BLE001 - intentional tool boundary
            return {"ok": False, "error": str(exc)}

    def are_equivalent(self, pred: str | None, gold: str | None, tol: float = 1e-6) -> bool:
        """Check symbolic or numerical equivalence between `pred` and `gold`, never raising."""
        if not pred or not gold:
            return False
        cand_pred = _extract_math_candidate(pred)
        cand_gold = _extract_math_candidate(gold)
        if cand_pred == cand_gold:
            return True
        try:
            p_obj = _safe_parse(cand_pred, max_len=self.max_len, max_exponent=self.max_exponent)
            g_obj = _safe_parse(cand_gold, max_len=self.max_len, max_exponent=self.max_exponent)
            if isinstance(p_obj, list) and len(p_obj) == 1:
                p_obj = p_obj[0]
            if isinstance(g_obj, list) and len(g_obj) == 1:
                g_obj = g_obj[0]
            if not isinstance(p_obj, sp.Basic) or not isinstance(g_obj, sp.Basic):
                return p_obj == g_obj
            diff = sp.simplify(p_obj - g_obj)
            if diff == 0 or diff.equals(0) is True:
                return True
            if not diff.free_symbols:
                cval = complex(diff.evalf(15))
                return abs(cval) <= tol
            return False
        except Exception:  # noqa: BLE001 - equivalence check must never crash rollout
            return False
