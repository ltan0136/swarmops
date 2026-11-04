"""Replayed tools: stand-ins for the live work tools that answer from a recording.

Each stand-in keeps the live tool's name, description and argument schema, so the model sees
exactly the same tool. Only the answer changes: it comes from the recording, which freezes
the environment (tool outputs, external services, the clock) at the time it was recorded.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ValidationError

from swarmops.evals.episodes import ToolReturn

MissPolicy = Literal["live", "error"]
REPLAY_MISS_OUTPUT = "Error: no recorded output for this tool call (replay miss)"


@dataclass
class ReplayStats:
    hits: int = 0  # answered from the recording
    reused: int = 0  # same call repeated more often than recorded: last output reused
    misses: list[str] = field(default_factory=list)  # calls the recording never saw


class ReplayToolbox:
    """Builds replayed copies of `live_tools` for one episode and counts how well they matched.

    A call the recording never saw is a miss: the new agent left the recorded path, so there is
    no frozen answer for it. `on_miss="live"` calls the real tool (the result is then not frozen);
    `on_miss="error"` returns an error message to the agent. Misses are always counted, so a
    report can say how much of a comparison actually ran inside the recording.
    """

    def __init__(
        self,
        recording: Sequence[ToolReturn],
        live_tools: Sequence[BaseTool],
        on_miss: MissPolicy = "live",
    ) -> None:
        self._live = {tool.name: tool for tool in live_tools}
        self._queues: dict[str, deque[str]] = defaultdict(deque)
        self._last: dict[str, str] = {}
        self._lock = threading.Lock()  # sync tools may run in a worker thread
        self.on_miss = on_miss
        self.stats = ReplayStats()
        for record in recording:
            self._queues[self._key(record.tool, record.args)].append(record.output)

    def tools(self) -> list[BaseTool]:
        return [self._replayed(tool) for tool in self._live.values()]

    def _key(self, name: str, args: dict[str, Any]) -> str:
        """Canonical call key: arguments are validated through the tool's schema so that
        {} and {"timezone": "UTC"} match when "UTC" is the default."""
        schema = getattr(self._live.get(name), "args_schema", None)
        normalised = args
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                normalised = schema(**args).model_dump()
            except ValidationError:
                normalised = args
        return json.dumps([name, normalised], sort_keys=True, ensure_ascii=False, default=str)

    def _answer(self, name: str, args: dict[str, Any]) -> str:
        key = self._key(name, args)
        with self._lock:
            queue = self._queues.get(key)
            if queue:
                output = queue.popleft()
                self._last[key] = output
                self.stats.hits += 1
                return output
            if key in self._last:
                self.stats.reused += 1
                return self._last[key]
            self.stats.misses.append(key)
        if self.on_miss == "live":
            return str(self._live[name].invoke(args))
        return REPLAY_MISS_OUTPUT

    def _replayed(self, tool: BaseTool) -> BaseTool:
        def run(**kwargs: Any) -> str:
            return self._answer(tool.name, kwargs)

        return StructuredTool.from_function(
            func=run, name=tool.name, description=tool.description, args_schema=tool.args_schema
        )
