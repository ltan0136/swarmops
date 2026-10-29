# ADR 0002: Postgres as the checkpointer

- Status: accepted
- Date: 2026-10-30

## Context

LangGraph saves graph state after every step through a checkpointer. Without a durable one,
conversation state disappears whenever a container restarts, which defeats the point of running
on a platform that restarts containers freely. Later versions also need durable state for
human-in-the-loop approvals (v0.2) and for several API replicas sharing threads (v0.3).

## Decision

Use `AsyncPostgresSaver` from `langgraph-checkpoint-postgres`, backed by a `psycopg_pool`
`AsyncConnectionPool` that is opened in the FastAPI lifespan and closed on shutdown.
`saver.setup()` runs at startup and is idempotent. When `DATABASE_URL` is empty the app falls
back to `InMemorySaver`, so local development and unit tests need no database.

## Consequences

- State survives restarts. `tests/test_postgres.py` and the CI compose smoke test both restart
  the process and check that a thread continues where it left off.
- The API stays stateless apart from the database, so it can be scaled horizontally later.
  The one exception is the per-thread `asyncio.Lock`, which only works inside one process and
  must move to Redis or Postgres advisory locks before running more than one replica.
- Postgres is a new operational dependency, managed by docker compose for now.
