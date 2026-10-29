# ADR 0003: A scripted offline model for demos, tests and CI

- Status: accepted
- Date: 2026-10-30

## Context

Calling a real LLM in tests is slow, costs money, needs secrets in CI and gives
non-deterministic results. Mocking the whole graph, on the other hand, would stop the tests from
exercising LangGraph routing, tool execution, handoffs and checkpointing.

## Decision

Add `ScriptedSwarmModel`, a `BaseChatModel` that plays one swarm role and emits the next step
of a fixed planner -> executor -> reviewer script, including genuine tool calls. It is selected
with `LLM_PROVIDER=fake`, which is the default. Real models are selected with
`LLM_PROVIDER=openai_compatible`, which covers Ollama, vLLM, DeepSeek, DashScope (Qwen) and
OpenAI through one client.

## Consequences

- `docker compose up` works on a fresh clone with no API key, and CI needs no secrets.
- Tests cover the real runtime: graph routing, tool execution, handoffs, persistence and the
  HTTP/SSE layer. They do not measure model quality; that is the job of the eval suite in v0.4.
- The script must stay in sync with the agent names and tools. It imports both from
  `roles.py` and `swarm/tools.py` rather than repeating string literals.
