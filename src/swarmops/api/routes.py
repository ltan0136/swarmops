"""HTTP routes. All handlers are async and never block the event loop."""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import uuid
import weakref
from collections.abc import AsyncIterator
from typing import Annotated, Any

import openai
from fastapi import APIRouter, Depends, HTTPException, Path, Request, Response, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from swarmops import __version__
from swarmops.api.schemas import (
    THREAD_ID_PATTERN,
    HealthResponse,
    MessageOut,
    RunRequest,
    RunResponse,
    ThreadCreated,
    ThreadState,
)
from swarmops.config import Settings
from swarmops.messages import current_turn, final_answer

logger = logging.getLogger(__name__)
router = APIRouter()

ThreadId = Annotated[str, Path(pattern=THREAD_ID_PATTERN, description="Conversation thread id.")]


class ThreadLocks:
    """One asyncio.Lock per thread so two runs never interleave on the same conversation.

    This only protects a single process. With several replicas (v0.3) the lock has to move
    to Redis or Postgres advisory locks.
    """

    def __init__(self) -> None:
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def get(self, thread_id: str) -> asyncio.Lock:
        lock = self._locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[thread_id] = lock
        return lock


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.graph


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_thread_lock(thread_id: ThreadId, request: Request) -> asyncio.Lock:
    lock: asyncio.Lock = request.app.state.thread_locks.get(thread_id)
    if lock.locked():
        raise HTTPException(status.HTTP_409_CONFLICT, f"Thread {thread_id!r} already has a run.")
    return lock


Graph = Annotated[CompiledStateGraph, Depends(get_graph)]
AppSettings = Annotated[Settings, Depends(get_settings_dep)]
ThreadLock = Annotated[asyncio.Lock, Depends(get_thread_lock)]

bearer = HTTPBearer(auto_error=False, description="Required when SWARMOPS_API_KEY is set.")


def require_api_key(
    settings: AppSettings,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> None:
    """Check the bearer token in constant time. No-op when SWARMOPS_API_KEY is not set."""
    if settings.swarmops_api_key is None:
        return
    expected = settings.swarmops_api_key.get_secret_value().encode()
    given = credentials.credentials.encode() if credentials else b""
    if not secrets.compare_digest(given, expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Missing or invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Every route except /healthz goes on `protected`; the auth check runs before the
# thread-lock check, so an unauthenticated caller cannot probe which threads are busy.
protected = APIRouter(dependencies=[Depends(require_api_key)])


def _run_config(thread_id: str, settings: Settings) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": settings.recursion_limit}


def _run_input(body: RunRequest) -> dict[str, Any]:
    return {"messages": [HumanMessage(content=body.message)]}


def _translate_error(exc: Exception) -> HTTPException:
    if isinstance(exc, GraphRecursionError):
        return HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "The agents did not finish within RECURSION_LIMIT steps.",
        )
    if isinstance(exc, openai.APIError):
        return HTTPException(status.HTTP_502_BAD_GATEWAY, f"LLM provider error: {exc}")
    return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Internal error.")


@router.get("/healthz", response_model=HealthResponse, tags=["ops"])
async def healthz(request: Request, settings: AppSettings) -> HealthResponse:
    checkpointer = type(request.app.state.graph.checkpointer).__name__
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_provider=settings.llm_provider,
        checkpointer=checkpointer,
    )


@protected.post("/threads", response_model=ThreadCreated, status_code=201, tags=["threads"])
async def create_thread() -> ThreadCreated:
    return ThreadCreated(thread_id=uuid.uuid4().hex)


@protected.get("/threads/{thread_id}", response_model=ThreadState, tags=["threads"])
async def get_thread(thread_id: ThreadId, graph: Graph) -> ThreadState:
    snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    messages = snapshot.values.get("messages", []) if snapshot.values else []
    if not messages:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Thread {thread_id!r} not found.")
    return ThreadState(
        thread_id=thread_id,
        active_agent=snapshot.values.get("active_agent"),
        messages=[MessageOut.from_message(m) for m in messages],
    )


@protected.delete("/threads/{thread_id}", status_code=204, tags=["threads"])
async def delete_thread(thread_id: ThreadId, graph: Graph, lock: ThreadLock) -> Response:
    """Delete every checkpoint of a thread. A thread with a run in progress returns 409."""
    async with lock:
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        if not (snapshot.values or {}).get("messages"):
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Thread {thread_id!r} not found.")
        await graph.checkpointer.adelete_thread(thread_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@protected.post("/threads/{thread_id}/runs", response_model=RunResponse, tags=["runs"])
async def run_thread(
    thread_id: ThreadId, body: RunRequest, graph: Graph, settings: AppSettings, lock: ThreadLock
) -> RunResponse:
    """Run the swarm on one user message and return the final answer with every step."""
    async with lock:
        try:
            result = await graph.ainvoke(_run_input(body), _run_config(thread_id, settings))
        except Exception as exc:
            logger.exception("Run failed on thread %s", thread_id)
            raise _translate_error(exc) from exc

    messages = result["messages"]
    _, turn = current_turn(messages)
    return RunResponse(
        thread_id=thread_id,
        answer=final_answer(messages),
        active_agent=result.get("active_agent"),
        steps=[MessageOut.from_message(m) for m in turn],
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@protected.post("/threads/{thread_id}/runs/stream", tags=["runs"])
async def stream_thread(
    thread_id: ThreadId, body: RunRequest, graph: Graph, settings: AppSettings, lock: ThreadLock
) -> StreamingResponse:
    """Stream each agent step as Server-Sent Events: `message`*, then `done` or `error`."""
    config = _run_config(thread_id, settings)

    async def events() -> AsyncIterator[str]:
        async with lock:
            try:
                # subgraphs=True surfaces the steps inside each agent (model call, tool call).
                async for namespace, chunk in graph.astream(
                    _run_input(body), config, stream_mode="updates", subgraphs=True
                ):
                    if not namespace:
                        continue  # parent-level updates repeat the full history; skip them
                    agent = namespace[0].split(":", 1)[0]
                    for update in chunk.values():
                        for message in (update or {}).get("messages", []):
                            payload = MessageOut.from_message(message).model_dump()
                            yield _sse("message", {"agent": agent, **payload})
                snapshot = await graph.aget_state(config)
                yield _sse(
                    "done",
                    {
                        "thread_id": thread_id,
                        "answer": final_answer(snapshot.values.get("messages", [])),
                        "active_agent": snapshot.values.get("active_agent"),
                    },
                )
            except Exception as exc:
                logger.exception("Streamed run failed on thread %s", thread_id)
                yield _sse("error", {"status": _translate_error(exc).status_code})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


router.include_router(protected)
