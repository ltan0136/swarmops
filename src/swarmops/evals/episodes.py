"""Recorded traffic. One episode = one user request plus every work-tool return it saw.

Episodes are stored as JSON Lines, one episode per line:

    {"id": "old-01", "input": "What is (12 + 3) * 4?",
     "tool_returns": [{"tool": "calculator", "args": {"expression": "(12 + 3) * 4"},
                       "output": "60"}],
     "expected": {"contains": ["60"]},
     "note": "optional free text"}

`tool_returns` freezes the environment; `expected` is the domain expert's definition of a
correct answer. Episodes without `expected` still run, but are reported as unscored.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from swarmops.messages import message_text


@dataclass(frozen=True)
class ToolReturn:
    """One recorded call to a work tool and what the world answered."""

    tool: str
    args: dict[str, Any]
    output: str


@dataclass(frozen=True)
class Expected:
    """What a correct answer must and must not contain. Written by a domain expert."""

    contains: tuple[str, ...] = ()
    not_contains: tuple[str, ...] = ()

    @property
    def defined(self) -> bool:
        return bool(self.contains or self.not_contains)

    def check(self, answer: str) -> bool:
        return all(s in answer for s in self.contains) and not any(
            s in answer for s in self.not_contains
        )


@dataclass(frozen=True)
class Episode:
    id: str
    input: str
    tool_returns: tuple[ToolReturn, ...] = ()
    expected: Expected = field(default_factory=Expected)
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "input": self.input,
            "tool_returns": [
                {"tool": r.tool, "args": r.args, "output": r.output} for r in self.tool_returns
            ],
        }
        if self.expected.defined:
            data["expected"] = {
                key: list(value)
                for key, value in (
                    ("contains", self.expected.contains),
                    ("not_contains", self.expected.not_contains),
                )
                if value
            }
        if self.note:
            data["note"] = self.note
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Episode:
        if not isinstance(data.get("id"), str) or not isinstance(data.get("input"), str):
            raise ValueError('each episode needs string fields "id" and "input"')
        expected = data.get("expected") or {}
        return cls(
            id=data["id"],
            input=data["input"],
            tool_returns=tuple(
                ToolReturn(
                    tool=str(r["tool"]), args=dict(r.get("args", {})), output=str(r["output"])
                )
                for r in data.get("tool_returns", [])
            ),
            expected=Expected(
                contains=tuple(expected.get("contains", ())),
                not_contains=tuple(expected.get("not_contains", ())),
            ),
            note=str(data.get("note", "")),
        )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Parse a JSON Lines file, skipping blank lines. Errors name the offending line."""
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(row)
    return rows


def load_episodes(path: str | Path) -> list[Episode]:
    episodes: list[Episode] = []
    for number, row in enumerate(read_jsonl(path), 1):
        try:
            episodes.append(Episode.from_json(row))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{path}: episode {number}: {exc}") from exc
    ids = [e.id for e in episodes]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path}: duplicate episode ids {duplicates}")
    if not episodes:
        raise ValueError(f"{path}: no episodes")
    return episodes


def save_episodes(path: str | Path, episodes: Iterable[Episode]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e.to_json(), ensure_ascii=False) for e in episodes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_tool_returns(
    turn: Sequence[BaseMessage], work_tool_names: Collection[str]
) -> tuple[ToolReturn, ...]:
    """Pair each work-tool call in a turn with its ToolMessage. Handoffs are not recorded:
    they are the agents' own decisions, not part of the environment."""
    calls = {
        call["id"]: call
        for message in turn
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] in work_tool_names
    }
    returns = []
    for message in turn:
        if isinstance(message, ToolMessage) and message.tool_call_id in calls:
            call = calls[message.tool_call_id]
            returns.append(
                ToolReturn(tool=call["name"], args=dict(call["args"]), output=message_text(message))
            )
    return tuple(returns)
