from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import requests

from .util import atomic_write_json


REQUEST_SUFFIX = ".request.json"
RESPONSE_SUFFIX = ".response.json"


class EvalExchangeBroker:
    """Forward file-based evaluator requests outside an agent command sandbox."""

    def __init__(
        self,
        directory: Path,
        *,
        gateway_url: str,
        run_id: str,
        run_token: str,
        poll_interval: float = 0.05,
        session: requests.Session | None = None,
    ) -> None:
        self.directory = directory
        self.gateway_url = gateway_url.rstrip("/")
        self.run_id = run_id
        self.run_token = run_token
        self.poll_interval = poll_interval
        self.session = session or requests.Session()
        self.directory.mkdir(parents=True, exist_ok=True)

    def serve(self, stop: threading.Event) -> None:
        while not stop.is_set():
            if not self.serve_once():
                stop.wait(self.poll_interval)

    def serve_once(self) -> bool:
        requests_found = sorted(self.directory.glob(f"*{REQUEST_SUFFIX}"))
        if not requests_found:
            return False
        for request_path in requests_found:
            self._serve_request(request_path)
        return True

    def _serve_request(self, request_path: Path) -> None:
        response_name = request_path.name[: -len(REQUEST_SUFFIX)] + RESPONSE_SUFFIX
        response_path = request_path.with_name(response_name)
        if response_path.exists():
            request_path.unlink(missing_ok=True)
            return
        try:
            envelope = json.loads(request_path.read_text())
            if not isinstance(envelope, dict) or envelope.get("schema_version") != 1:
                raise ValueError("exchange envelope must be a schema-version-1 object")
            payload = envelope.get("request")
            if not isinstance(payload, dict):
                raise ValueError("exchange request payload must be an object")
            if payload.get("run_id") != self.run_id:
                raise ValueError("exchange request run_id does not match its broker")
            timeout = envelope.get("timeout_seconds", 900.0)
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise ValueError("exchange timeout_seconds must be numeric")
            if timeout <= 0 or timeout > 86400:
                raise ValueError("exchange timeout_seconds must be in (0, 86400]")
            response = self.session.post(
                self.gateway_url + "/v1/evaluate",
                json=payload,
                headers={"Authorization": f"Bearer {self.run_token}"},
                timeout=(10, float(timeout)),
            )
            try:
                body: Any = response.json()
            except ValueError as exc:
                raise ValueError(
                    f"gateway HTTP {response.status_code} returned invalid JSON"
                ) from exc
            result = {
                "schema_version": 1,
                "status_code": response.status_code,
                "body": body,
            }
        except (OSError, ValueError, TypeError, requests.RequestException) as exc:
            result = {
                "schema_version": 1,
                "transport_error": str(exc),
            }
        atomic_write_json(response_path, result)
        request_path.unlink(missing_ok=True)
