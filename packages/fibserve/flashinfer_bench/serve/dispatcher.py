"""HTTP dispatcher for multiple FIBServe backends.

Each backend owns one or more CUDA devices and its own task store. Submit
requests are sent to the least-loaded healthy backend, and task ownership is
remembered so polling is forwarded to the backend that created the task.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request

logger = logging.getLogger("fibserve-dispatcher")


def _normalize_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return url.rstrip("/")


class Dispatcher:
    """Route requests across FIBServe backends and track task ownership."""

    def __init__(self, urls: list[str], request_timeout: float = 3600.0):
        self.urls = [_normalize_url(url) for url in urls]
        self.client = httpx.AsyncClient(
            timeout=request_timeout,
            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=256),
        )
        self.health_client = httpx.AsyncClient(
            timeout=10.0,
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=32),
        )
        self.task_to_url: dict[str, str] = {}
        self._rr_idx = 0
        self._rr_lock = asyncio.Lock()

    async def close(self) -> None:
        await asyncio.gather(self.client.aclose(), self.health_client.aclose())

    async def queue_sizes(self) -> list[tuple[int | None, str]]:
        """Return ``(queue_size, url)`` for every configured backend."""

        async def probe(url: str) -> tuple[int | None, str]:
            try:
                response = await self.health_client.get(f"{url}/health", timeout=5.0)
                if response.status_code == 200:
                    body = response.json()
                    workers = body.get("workers")
                    if isinstance(workers, list) and workers:
                        healthy_workers = (
                            bool(worker.get("healthy")) for worker in workers if isinstance(worker, dict)
                        )
                        if not any(healthy_workers):
                            return None, url
                    return body.get("queue_size", 0), url
            except (httpx.HTTPError, ValueError, TypeError, AttributeError) as error:
                logger.debug("Health probe failed for %s: %s", url, error)
            return None, url

        return list(await asyncio.gather(*(probe(url) for url in self.urls)))

    async def pick_submit_backend(self) -> str:
        """Pick the least-loaded healthy backend, round-robin among ties."""
        sizes = await self.queue_sizes()
        healthy = [(size, url) for size, url in sizes if size is not None]
        if healthy:
            min_size = min(size for size, _ in healthy)
            candidates = [url for size, url in healthy if size == min_size]
        else:
            candidates = self.urls

        async with self._rr_lock:
            url = candidates[self._rr_idx % len(candidates)]
            self._rr_idx += 1
        return url

    async def forward(
        self,
        method: str,
        url: str,
        path: str,
        **kwargs: Any,
    ) -> tuple[int, Any]:
        response = await self.client.request(method, f"{url}{path}", **kwargs)
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text
        return response.status_code, body

    async def try_each(self, method: str, path: str, **kwargs: Any) -> tuple[int, Any]:
        """Try backends in order until one returns a non-5xx response."""
        last: tuple[int, Any] = (503, "no backends available")
        for url in self.urls:
            try:
                code, body = await self.forward(method, url, path, **kwargs)
                if code < 500:
                    return code, body
                last = (code, body)
            except httpx.HTTPError as error:
                last = (502, str(error))
        return last


_dispatcher: Dispatcher | None = None


def _get_dispatcher() -> Dispatcher:
    if _dispatcher is None:
        raise RuntimeError("Dispatcher not initialized")
    return _dispatcher


def _raise(code: int, body: Any) -> None:
    detail = body["detail"] if isinstance(body, dict) and "detail" in body else body
    raise HTTPException(code, detail=detail)


def _raise_backend_error(url: str, path: str, error: httpx.RequestError) -> None:
    if isinstance(error, httpx.TimeoutException):
        status_code = 504
        detail = f"Backend timed out while handling {path}: {url}"
    else:
        status_code = 502
        detail = f"Backend request failed while handling {path}: {url}: {error}"
    logger.warning("%s", detail)
    raise HTTPException(status_code, detail=detail) from error


@asynccontextmanager
async def _lifespan(app: FastAPI):
    del app
    yield
    if _dispatcher is not None:
        await _dispatcher.close()


app = FastAPI(title="FIBServe Dispatcher", lifespan=_lifespan)


@app.get("/")
async def root() -> dict[str, Any]:
    dispatcher = _get_dispatcher()
    return {
        "name": "FIBServe Dispatcher",
        "backends": dispatcher.urls,
        "tracked_tasks": len(dispatcher.task_to_url),
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    dispatcher = _get_dispatcher()
    sizes = await dispatcher.queue_sizes()
    backends = [{"url": url, "queue_size": size, "healthy": size is not None} for size, url in sizes]
    total = sum(size for size, _ in sizes if size is not None)
    any_healthy = any(size is not None for size, _ in sizes)
    return {
        "status": "ok" if any_healthy else "unhealthy",
        "backends": backends,
        "queue_size": total,
    }


async def _forward_any_get(path: str) -> Any:
    code, body = await _get_dispatcher().try_each("GET", path)
    if code >= 400:
        _raise(code, body)
    return body


@app.get("/definitions")
async def list_definitions() -> Any:
    return await _forward_any_get("/definitions")


@app.get("/definitions/{name}")
async def get_definition(name: str) -> Any:
    return await _forward_any_get(f"/definitions/{name}")


@app.get("/definitions/{name}/workloads")
async def list_workloads(name: str) -> Any:
    return await _forward_any_get(f"/definitions/{name}/workloads")


@app.get("/workloads/{uuid}")
async def get_workload(uuid: str) -> Any:
    return await _forward_any_get(f"/workloads/{uuid}")


async def _submit(path: str, body: dict[str, Any]) -> dict[str, Any]:
    dispatcher = _get_dispatcher()
    url = await dispatcher.pick_submit_backend()
    try:
        code, data = await dispatcher.forward("POST", url, path, json=body)
    except httpx.RequestError as error:
        _raise_backend_error(url, path, error)
    if code >= 400:
        _raise(code, data)
    if isinstance(data, dict) and "task_id" in data:
        dispatcher.task_to_url[data["task_id"]] = url
        logger.info("%s task_id=%s -> %s", path, data["task_id"], url)
    return data


@app.post("/evaluate")
async def evaluate(request: Request) -> Any:
    return await _submit("/evaluate", await request.json())


@app.post("/profile")
async def profile(request: Request) -> Any:
    return await _submit("/profile", await request.json())


@app.post("/sanitize")
async def sanitize(request: Request) -> Any:
    return await _submit("/sanitize", await request.json())


@app.post("/debug")
async def debug(request: Request) -> Any:
    return await _submit("/debug", await request.json())


@app.get("/tasks/{task_id}")
async def get_task(task_id: str, timeout: float = Query(default=0, ge=0, le=3600)) -> Any:
    dispatcher = _get_dispatcher()
    url = dispatcher.task_to_url.get(task_id)
    if url is None:
        raise HTTPException(404, detail=f"Task not found: {task_id}")
    path = f"/tasks/{task_id}"
    try:
        code, data = await dispatcher.forward(
            "GET",
            url,
            path,
            params={"timeout": timeout},
        )
    except httpx.RequestError as error:
        _raise_backend_error(url, path, error)
    if code >= 400:
        _raise(code, data)
    return data


@app.post("/tasks/batch")
async def tasks_batch(request: Request) -> list[Any]:
    """Query owning backends in parallel and preserve task order."""
    dispatcher = _get_dispatcher()
    body = await request.json()
    task_ids: list[str] = body.get("task_ids", [])
    timeout = float(body.get("timeout", 0))

    by_url: dict[str, list[str]] = {}
    for task_id in task_ids:
        url = dispatcher.task_to_url.get(task_id)
        if url is None:
            raise HTTPException(404, detail=f"Task not found: {task_id}")
        by_url.setdefault(url, []).append(task_id)

    async def fetch(url: str, ids: list[str]) -> list[Any]:
        try:
            code, data = await dispatcher.forward(
                "POST",
                url,
                "/tasks/batch",
                json={"task_ids": ids, "timeout": timeout},
            )
        except httpx.RequestError as error:
            _raise_backend_error(url, "/tasks/batch", error)
        if code >= 400:
            _raise(code, data)
        return data

    grouped = await asyncio.gather(*(fetch(url, ids) for url, ids in by_url.items()))
    flat: dict[str, Any] = {}
    for ids, results in zip(by_url.values(), grouped):
        for task_id, result in zip(ids, results):
            flat[task_id] = result
    return [flat[task_id] for task_id in task_ids]


@app.post("/shutdown")
async def shutdown() -> dict[str, Any]:
    """Ask every backend to shut down."""
    dispatcher = _get_dispatcher()

    async def kill(url: str) -> None:
        try:
            await dispatcher.client.post(f"{url}/shutdown", timeout=5.0)
        except httpx.HTTPError as error:
            logger.warning("Shutdown %s failed: %s", url, error)

    await asyncio.gather(*(kill(url) for url in dispatcher.urls))
    return {"status": "shutting_down"}


def main() -> None:
    parser = argparse.ArgumentParser(description="FIBServe backend dispatcher")
    parser.add_argument(
        "--urls",
        nargs="+",
        required=True,
        help="Backend URLs as host:port or complete HTTP URLs",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    global _dispatcher
    _dispatcher = Dispatcher(args.urls)
    logger.info(
        "Dispatcher listening on %s:%d, backends=%s",
        args.host,
        args.port,
        _dispatcher.urls,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
