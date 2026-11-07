# CLAUDE.md — SwarmOps

A LangGraph swarm (planner → executor → reviewer) behind an async FastAPI service with a
Postgres checkpointer. The README, `README.zh-CN.md` and `docs/adr/` are the source of truth for
behaviour and design; this file is about **how to change the code without drifting from them**.

## Commands

```bash
pip install -e ".[dev]"     # or: make install
ruff check . && ruff format --check .   # CI runs both; fix with `make format`
pytest -q -rs               # offline model + in-memory state; Postgres tests skip without a DB
docker compose up -d --build --wait     # full stack, no API key needed
```

Postgres integration tests run only when `SWARMOPS_TEST_DATABASE_URL` is set (CI sets it).
A skipped test is not a passing test: say which tests were skipped when you report.

## How to work in this repo

1. **Plan first for anything non-trivial.** Before editing, list: files you will touch, new or
   changed functions/classes with signatures, any change to graph state or the API, and the tests
   you will add. Stop and wait for approval if the plan touches a protected area (below) or more
   than 3 files.
2. **Stay inside the task.** Touch only the files the task needs. No drive-by refactors, renames,
   reformatting of untouched code, or "while I'm here" improvements — list them as suggestions instead.
3. **Reuse before you add.** Extend the existing module (`roles.py`, `messages.py`, `llm/`,
   `checkpoint.py`, `config.py`) instead of creating a parallel one. No new dependency, wrapper
   or abstraction layer without asking first.
4. **Ambiguity is a question, not a decision.** If the request can be read two ways, ask. If you
   must proceed, state the assumption in your final report.
5. **One task, one small commit**, on a branch; never push to `main` or force-push. Do not add
   `Co-Authored-By: Claude` (or any Claude attribution) lines to commits.

## Tests are the contract

- Never edit, delete, skip, `xfail` or weaken an existing test or assertion to make it pass.
  If you think a test is wrong, stop and explain why.
- New behaviour gets a test. Do not mock the thing under test, and do not hard-code an expected
  value the code is supposed to compute.
- Tests and CI never call a real LLM and never need secrets (ADR 0003). Use the `settings`,
  `graph` and `client` fixtures in `tests/conftest.py`.

## Errors must stay visible

- No bare `except:`. The only broad `except Exception` blocks are the two run handlers in
  `api/routes.py`, which log and translate through `_translate_error`. A new failure mode
  extends `_translate_error` and the status-code notes in the README; it is not swallowed.
- No silent fallbacks. In-memory state is used **only** when `DATABASE_URL` is empty; a Postgres
  connection failure must raise, never fall back to `InMemorySaver`.
- Tool errors are returned as strings starting with `Error:` (see `swarm/tools.py`) so the
  reviewer can report them; they are not raised into the graph.

## Protected areas — ask before changing

| Area | Invariant | Where it is checked |
|---|---|---|
| Handoff topology | planner→executor, executor→reviewer, reviewer→{planner, executor}. Topology is data in `swarm/graph.py`, not prompt text. A change needs a new ADR. | ADR 0001, `tests/test_swarm.py` |
| Agent and tool names | Agent names come only from `roles.py`; handoff tool names only from `handoff_tool_name()`. No string literals for them in `src/`. | `llm/fake.py` imports both |
| Tool ownership | Only the executor holds `WORK_TOOLS`. | `swarm/graph.py` |
| Offline model | `ScriptedSwarmModel` is the default provider and must stay in sync with agent names and tools. | ADR 0003, all tests |
| Checkpointer | Chosen only in `checkpoint.py`; pool opened and closed in the FastAPI lifespan; `saver.setup()` at startup. | ADR 0002, `tests/test_postgres.py` |
| Concurrency | Per-thread `asyncio.Lock` is single-process only. Do not claim multi-replica safety. | ADR 0002 |
| Async API | Handlers never block the event loop: no sync I/O, `time.sleep` or blocking DB calls. | `api/routes.py` |
| Auth | Every route except `/healthz` is registered on `protected`. | `tests/test_api.py` |
| Calculator | Parsed with `ast`, never `eval`; length, exponent and result-size caps stay. | `tests/test_tools.py` |
| Config | Every setting is a field on `Settings` in `config.py`, with a line in `.env.example`. No `os.environ` reads in `src/`. | `tests/test_config.py` |
| Secrets | Never commit `.env` or a real key. | `.gitignore` |

## Agent-specific drift to avoid

- **Do not patch behaviour with prompt rules.** If an agent routes or loops wrongly, first check
  the graph, the handoff edges, `active_agent` and the message state. Edit `swarm/prompts.py` only
  when the cause really is the instruction, and say why in the report.
- **State has one definition.** Graph state comes from `langgraph-swarm` (`messages`,
  `active_agent`). Do not add a second store of conversation state (module globals, caches,
  files); anything new goes into the graph state and therefore the checkpointer.
- **Keep `RECURSION_LIMIT` as the loop guard.** Do not raise it to make a run pass.
- **Roadmap items are not in scope** unless asked: MCP tool servers, human-in-the-loop (v0.2),
  Kubernetes/Redis/OTel (v0.3), eval suite (v0.4).

## Code style

English identifiers, comments and docstrings. Type hints on public functions. Line length 100
(ruff). Prefer explicit, simple code over clever code.

## Final report (every task)

End with:

1. What changed and why, file by file, plus `git diff --stat`.
2. Assumptions you made and anything you were unsure about.
3. Commands you ran and their results, including skipped tests.
4. Anything in "Protected areas" you touched, and anything you chose not to do.

## Rules learned from real drift

When the same kind of drift happens twice, add a rule here (and a test or lint if possible).
Format: `- YYYY-MM-DD — what went wrong → the rule.`

- (none yet)
