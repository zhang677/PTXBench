"""Tests for the FIBServe dispatcher API."""

from typing import Any

import pytest
from flashinfer_bench.serve import dispatcher as dispatcher_module
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.asyncio


class FakeDispatcher:
    def __init__(self) -> None:
        self.urls = ["http://backend-a"]
        self.task_to_url: dict[str, str] = {}
        self.forwarded: list[dict[str, Any]] = []

    async def pick_submit_backend(self) -> str:
        return self.urls[0]

    async def forward(
        self,
        method: str,
        url: str,
        path: str,
        **kwargs: Any,
    ) -> tuple[int, Any]:
        self.forwarded.append({"method": method, "url": url, "path": path, "kwargs": kwargs})
        if path.startswith("/tasks/"):
            return 200, {
                "task_id": "debug-task",
                "kind": "debug",
                "status": "completed",
                "definition": "test",
                "solution": "solution_1",
            }
        return 200, {
            "task_id": "debug-task",
            "normalized_solution_name": "solution_1",
        }

    async def close(self) -> None:
        pass


async def test_debug_submit_is_forwarded_tracked_and_polled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatcher = FakeDispatcher()
    monkeypatch.setattr(dispatcher_module, "_dispatcher", dispatcher)

    async with AsyncClient(
        transport=ASGITransport(app=dispatcher_module.app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/debug",
            json={"solution": {"name": "solution_1"}},
        )
        assert response.status_code == 200
        assert dispatcher.task_to_url == {"debug-task": "http://backend-a"}

        response = await client.get("/tasks/debug-task", params={"timeout": 30})

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert dispatcher.forwarded == [
        {
            "method": "POST",
            "url": "http://backend-a",
            "path": "/debug",
            "kwargs": {"json": {"solution": {"name": "solution_1"}}},
        },
        {
            "method": "GET",
            "url": "http://backend-a",
            "path": "/tasks/debug-task",
            "kwargs": {"params": {"timeout": 30.0}},
        },
    ]
