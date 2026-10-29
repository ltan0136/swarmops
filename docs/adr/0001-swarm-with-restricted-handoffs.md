# ADR 0001: A swarm with restricted handoff edges

- Status: accepted
- Date: 2026-10-30

## Context

A multi-agent system can be organised as a supervisor (one router agent delegates to workers)
or as a swarm (agents hand control directly to each other). The research goal of this project
is to study decentralised coordination, so the swarm pattern is the natural fit. Fully connected
swarms, however, are hard to reason about and easy to send into loops.

## Decision

Use `langgraph-swarm` with three agents and an explicit, minimal set of handoff edges:

| From     | Can hand off to      |
|----------|----------------------|
| planner  | executor             |
| executor | reviewer             |
| reviewer | planner, executor    |

Only the executor holds work tools. The graph-wide `RECURSION_LIMIT` caps the number of steps
per run, so a reviewer/executor rework loop cannot run forever.

## Consequences

- The topology is data, not prose in a prompt: changing it means editing `swarm/graph.py`,
  which makes it easy to compare topologies in later experiments (v0.4).
- `active_agent` is persisted per thread, so the next user message goes to whoever spoke last
  (usually the reviewer). The reviewer must therefore recognise a new request and route it to
  the planner. This is tested in `tests/test_swarm.py`.
