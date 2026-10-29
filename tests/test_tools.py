from __future__ import annotations

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
