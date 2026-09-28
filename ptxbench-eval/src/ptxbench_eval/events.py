from __future__ import annotations

from typing import Any

from .registry import Registry
from .util import append_jsonl, utc_now


class RawEventLog:
    def __init__(self, registry: Registry, run_id: str, action_id: str):
        self.registry = registry
        self.run_id = run_id
        self.action_id = action_id
        self.sequence = 0

    def emit(self, event: str, **payload: Any) -> None:
        self.sequence += 1
        append_jsonl(
            self.registry.raw_events_path(self.run_id),
            {
                "schema_version": 1,
                "timestamp": utc_now(),
                "run_id": self.run_id,
                "action_id": self.action_id,
                "sequence": self.sequence,
                "event": event,
                **payload,
            },
        )

