"""Work tools available to the executor agent."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool, tool

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 100

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval(expression: str) -> int | float:
    """Evaluate a pure arithmetic expression without using eval().

    Raises:
        ValueError: if the expression is too long, malformed or uses anything but numbers,
            parentheses and + - * / // % **.
        ZeroDivisionError: on division by zero.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"expression longer than {MAX_EXPRESSION_LENGTH} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {expression!r}") from exc
    return _eval_node(tree.body)


def _eval_node(node: ast.expr) -> int | float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent larger than {MAX_EXPONENT}")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def format_number(value: int | float) -> str:
    """Render integers exactly and floats with up to 12 significant digits."""
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(12 + 3) * 4 / 5'.

    Supports numbers, parentheses and + - * / // % **. Returns the result, or a message
    starting with 'Error:' if the expression cannot be evaluated.
    """
    try:
        return format_number(safe_eval(expression))
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, OverflowError) as exc:
        return f"Error: {exc}"


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Return the current date and time (ISO 8601) in an IANA timezone.

    Example timezones: 'UTC', 'Europe/London', 'America/New_York'.
    """
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return f"Error: unknown timezone {timezone!r}"
    return datetime.now(zone).isoformat(timespec="seconds")


WORK_TOOLS: tuple[BaseTool, ...] = (calculator, get_current_time)
WORK_TOOL_NAMES: frozenset[str] = frozenset(t.name for t in WORK_TOOLS)
