from __future__ import annotations

import time

import pytest

from swarmops.llm.fake import extract_expression
from swarmops.swarm.tools import calculator, get_current_time


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(12 + 3) * 4", "60"),
        ("2 ** 10", "1024"),
        ("7 / 2", "3.5"),
        ("-5 + 2", "-3"),
        ("10 // 3", "3"),
        ("0.1 + 0.2", "0.3"),
    ],
)
def test_calculator_evaluates_arithmetic(expression: str, expected: str) -> None:
    assert calculator.invoke({"expression": expression}) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hacked')",
        "1 + abc",
        "2 ** 1000",
        "1 +",
        "x" * 300,
    ],
)
def test_calculator_rejects_unsafe_or_invalid_input(expression: str) -> None:
    assert calculator.invoke({"expression": expression}).startswith("Error:")


def test_calculator_keeps_exact_large_integers_within_the_limit() -> None:
    assert calculator.invoke({"expression": "2 ** 100"}) == str(2**100)
    assert calculator.invoke({"expression": "(2 ** 64) * (2 ** 64)"}) == str(2**128)


@pytest.mark.parametrize(
    "expression",
    [
        # Each exponent is <= MAX_EXPONENT, but nesting them used to build a number with
        # ~10^8 digits and freeze the API process (the big-int pow holds the GIL).
        "(((9**99)**99)**99)**99",
        "(9**99)**99",
        "(2**100)**41",
        "(9**99*9**99)**99",
        "((2**100)**40)*((2**100)**40)",
    ],
)
def test_calculator_rejects_results_that_are_too_large(expression: str) -> None:
    started = time.perf_counter()
    assert calculator.invoke({"expression": expression}) == "Error: result too large"
    assert time.perf_counter() - started < 1.0


def test_calculator_division_by_zero() -> None:
    assert calculator.invoke({"expression": "1 / 0"}) == "Error: division by zero"


def test_get_current_time() -> None:
    assert get_current_time.invoke({"timezone": "Europe/London"}).startswith("20")
    assert get_current_time.invoke({"timezone": "Mars/Olympus"}).startswith("Error:")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What is (12 + 3) * 4?", "(12 + 3) * 4"),
        ("12*7是多少", "12*7"),
        ("what time is it?", None),
    ],
)
def test_extract_expression(text: str, expected: str | None) -> None:
    assert extract_expression(text) == expected
