from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import requests

from .recovery import ProfileRequestInterrupted


class FibServeError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        requires_restart: bool = False,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.requires_restart = requires_restart


class FibServeClient:
    def __init__(
        self,
        base_url: str,
        *,
        poll_interval: float = 5.0,
        session: requests.Session | None = None,
        request_permit: Callable[[], None] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.session = session or requests.Session()
        self.request_permit = request_permit

    def submit_and_poll(
        self,
        endpoint: str,
        payload: dict[str, Any],
        timeout: float,
        emit: Callable[..., None],
    ) -> dict[str, Any]:
        try:
            self._check_request_permit(emit, endpoint=endpoint)
            emit("profile_submit_request", endpoint=endpoint, request=payload)
            response = self.session.post(
                f"{self.base_url}/{endpoint}", json=payload, timeout=(5, 30)
            )
            body = self._json(response)
            emit(
                "profile_submit_response",
                endpoint=endpoint,
                http_status=response.status_code,
                response=body,
            )
            if response.status_code >= 400:
                raise FibServeError(
                    f"/{endpoint} returned HTTP {response.status_code}: {body}",
                    retryable=response.status_code == 429 or response.status_code >= 500,
                    requires_restart=response.status_code >= 500,
                )
            task_id = body.get("task_id")
            if not task_id:
                raise FibServeError(f"/{endpoint} response has no task_id: {body}")
        except FibServeError:
            raise
        except (requests.RequestException, TypeError, ValueError) as exc:
            emit("profile_transport_error", endpoint=endpoint, error=str(exc))
            raise FibServeError(
                f"/{endpoint} transport error: {exc}", requires_restart=True
            ) from exc

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                self._check_request_permit(emit, endpoint=endpoint, task_id=task_id)
                response = self.session.get(
                    f"{self.base_url}/tasks/{task_id}?timeout=30", timeout=(5, 40)
                )
                body = self._json(response)
                emit(
                    "profile_poll_response",
                    endpoint=endpoint,
                    task_id=task_id,
                    http_status=response.status_code,
                    response=body,
                )
                if response.status_code >= 400:
                    raise FibServeError(
                        f"task {task_id} returned HTTP {response.status_code}: {body}",
                        retryable=response.status_code == 429 or response.status_code >= 500,
                        requires_restart=response.status_code >= 500,
                    )
            except FibServeError:
                raise
            except (requests.RequestException, TypeError, ValueError) as exc:
                emit(
                    "profile_transport_error",
                    endpoint=endpoint,
                    task_id=task_id,
                    error=str(exc),
                )
                raise FibServeError(
                    f"task {task_id} poll error: {exc}", requires_restart=True
                ) from exc
            status = body.get("status")
            if status in {"completed", "failed"}:
                return body
            time.sleep(self.poll_interval)
        raise FibServeError(
            f"timed out waiting for /{endpoint} task {task_id}",
            requires_restart=True,
        )

    def _check_request_permit(
        self,
        emit: Callable[..., None],
        *,
        endpoint: str,
        task_id: str | None = None,
    ) -> None:
        if self.request_permit is None:
            return
        try:
            self.request_permit()
        except ProfileRequestInterrupted as exc:
            emit(
                "profile_request_paused",
                endpoint=endpoint,
                task_id=task_id,
                reason=str(exc),
            )
            raise FibServeError(str(exc), requires_restart=True) from exc

    @staticmethod
    def _json(response: requests.Response) -> dict[str, Any]:
        value = response.json()
        if not isinstance(value, dict):
            raise TypeError("response is not a JSON object")
        return value
