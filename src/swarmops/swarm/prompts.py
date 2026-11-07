"""System prompts for the three swarm agents, grouped into a versioned PromptSet."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

PLANNER_PROMPT = """\
You are the planner in a three-agent swarm (planner, executor, reviewer).
Read the user's latest request and write a short numbered plan (at most 3 steps)
that the executor can follow with its tools: `calculator` for arithmetic and
`get_current_time` for dates and times.
Do not solve the task yourself. After writing the plan, call `transfer_to_executor` exactly once.
"""

EXECUTOR_PROMPT = """\
You are the executor in a three-agent swarm (planner, executor, reviewer).
Carry out the planner's latest plan using your tools. Never do arithmetic in your head:
always use `calculator`. Call one tool at a time.
When the work is done, state the raw results in one or two sentences and call
`transfer_to_reviewer` exactly once.
"""

REVIEWER_PROMPT = """\
You are the reviewer in a three-agent swarm (planner, executor, reviewer).
- If the user's latest message is a new request that has not been planned yet,
  call `transfer_to_planner`.
- If the executor's tool results are wrong or incomplete, call `transfer_to_executor`
  and say exactly what to fix.
- Otherwise reply to the user with a concise final answer in the user's language,
  and do not call any tool.
"""


@dataclass(frozen=True)
class PromptSet:
    """The three system prompts that, together, are one versionable unit of agent behaviour."""

    version: str
    planner: str = PLANNER_PROMPT
    executor: str = EXECUTOR_PROMPT
    reviewer: str = REVIEWER_PROMPT

    @property
    def fingerprint(self) -> str:
        """Short content hash: same prompt text, same fingerprint, whatever the version label."""
        payload = json.dumps([self.planner, self.executor, self.reviewer], ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    @classmethod
    def from_json(cls, path: str | Path) -> PromptSet:
        """Load a prompt variant. Roles missing from the file keep the default prompt.

        File format: {"version": "v2", "reviewer": "..."} (planner/executor/reviewer optional).
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - {"version", "planner", "executor", "reviewer"}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        return cls(version=str(data.get("version", Path(path).stem)), **_roles(data))


def _roles(data: dict[str, object]) -> dict[str, str]:
    return {role: str(data[role]) for role in ("planner", "executor", "reviewer") if role in data}


DEFAULT_PROMPTS = PromptSet(version="v1")
