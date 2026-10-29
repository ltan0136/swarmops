from __future__ import annotations

import json

import httpx

from swarmops.api.app import create_app
from swarmops.config import Settings


async def test_concurrent_run_on_same_thread_returns_409(settings: Settings) -> None:
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        lock = app.state.thread_locks.get("busy")
        await lock.acquire()  # simulate a run that is still in progress
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                busy = await http.post("/threads/busy/runs", json={"message": "1 + 1"})
                free = await http.post("/threads/other/runs", json={"message": "1 + 1"})
        finally:
            lock.release()
    assert busy.status_code == 409
    assert free.status_code == 200


async def test_healthz(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "fake"
    assert body["checkpointer"] == "InMemorySaver"


async def test_run_returns_answer_and_steps(client: httpx.AsyncClient) -> None:
    thread_id = (await client.post("/threads")).json()["thread_id"]

    response = await client.post(f"/threads/{thread_id}/runs", json={"message": "What is 9 * 9?"})
    assert response.status_code == 200
    body = response.json()
    assert "81" in body["answer"]
    assert body["active_agent"] == "reviewer"
    assert [s["name"] for s in body["steps"] if s["type"] == "ai"][0] == "planner"


async def test_thread_state_persists_across_requests(client: httpx.AsyncClient) -> None:
    await client.post("/threads/demo-1/runs", json={"message": "What is 3 + 4?"})
    await client.post("/threads/demo-1/runs", json={"message": "And 10 - 1?"})

    state = (await client.get("/threads/demo-1")).json()
    human_messages = [m["content"] for m in state["messages"] if m["type"] == "human"]
    assert human_messages == ["What is 3 + 4?", "And 10 - 1?"]


async def test_unknown_thread_returns_404(client: httpx.AsyncClient) -> None:
    assert (await client.get("/threads/does-not-exist")).status_code == 404


async def test_validation_errors(client: httpx.AsyncClient) -> None:
    assert (await client.post("/threads/ok/runs", json={"message": ""})).status_code == 422
    assert (await client.post("/threads/bad%20id/runs", json={"message": "hi"})).status_code == 422


async def test_stream_emits_agent_steps_then_done(client: httpx.AsyncClient) -> None:
    events: list[tuple[str, dict]] = []
    async with client.stream(
        "POST", "/threads/stream-1/runs/stream", json={"message": "What is 5 * 5?"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        event = None
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: ") and event:
                events.append((event, json.loads(line.removeprefix("data: "))))

    names = [e for e, _ in events]
    assert names[-1] == "done"
    assert "error" not in names
    agents = [data["agent"] for e, data in events if e == "message"]
    assert agents[0] == "planner" and "executor" in agents and agents[-1] == "reviewer"
    assert "25" in events[-1][1]["answer"]


async def test_delete_thread_removes_its_history(client: httpx.AsyncClient) -> None:
    await client.post("/threads/to-delete/runs", json={"message": "What is 2 + 2?"})
    assert (await client.get("/threads/to-delete")).status_code == 200

    assert (await client.delete("/threads/to-delete")).status_code == 204
    assert (await client.get("/threads/to-delete")).status_code == 404
    assert (await client.delete("/threads/to-delete")).status_code == 404

    # The id can be reused and starts a fresh conversation with the planner.
    response = await client.post("/threads/to-delete/runs", json={"message": "What is 3 + 3?"})
    state = (await client.get("/threads/to-delete")).json()
    assert response.status_code == 200
    assert [m["content"] for m in state["messages"] if m["type"] == "human"] == ["What is 3 + 3?"]


async def test_delete_busy_thread_returns_409(settings: Settings) -> None:
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            await http.post("/threads/busy/runs", json={"message": "1 + 1"})
            lock = app.state.thread_locks.get("busy")
            await lock.acquire()
            try:
                response = await http.delete("/threads/busy")
            finally:
                lock.release()
            assert response.status_code == 409
            assert (await http.get("/threads/busy")).status_code == 200


async def test_api_key_protects_every_route_but_healthz() -> None:
    settings = Settings(
        _env_file=None, llm_provider="fake", database_url=None, swarmops_api_key="s3cret-token"
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            assert (await http.get("/healthz")).status_code == 200

            missing = await http.post("/threads")
            assert missing.status_code == 401
            assert missing.headers["www-authenticate"] == "Bearer"
            wrong = {"Authorization": "Bearer nope"}
            assert (
                await http.post("/threads/t1/runs", json={"message": "1"}, headers=wrong)
            ).status_code == 401
            assert (await http.get("/threads/t1", headers=wrong)).status_code == 401
            assert (await http.delete("/threads/t1", headers=wrong)).status_code == 401

            good = {"Authorization": "Bearer s3cret-token"}
            assert (await http.post("/threads", headers=good)).status_code == 201
            run = await http.post(
                "/threads/t1/runs", json={"message": "What is 6 * 7?"}, headers=good
            )
            assert run.status_code == 200 and "42" in run.json()["answer"]


async def test_auth_runs_before_busy_check() -> None:
    settings = Settings(
        _env_file=None, llm_provider="fake", database_url=None, swarmops_api_key="k" * 16
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        lock = app.state.thread_locks.get("busy")
        await lock.acquire()
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                response = await http.post("/threads/busy/runs", json={"message": "1 + 1"})
        finally:
            lock.release()
    assert response.status_code == 401
