from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .util import json_digest


class ContractError(ValueError):
    pass


def validate_language_contract(value: dict[str, Any]) -> None:
    language = value.get("language")
    if language == "cuda":
        if value.get("binding") != "tvm-ffi":
            raise ContractError("CUDA requires binding=tvm-ffi")
        if not isinstance(value.get("nvcc_gencode"), str) or not value["nvcc_gencode"].strip():
            raise ContractError("CUDA requires nvcc_gencode")
    else:
        raise ContractError("language must be cuda")


@dataclass(frozen=True)
class WorkloadManifest:
    data: dict[str, Any]

    @classmethod
    def parse(cls, value: Any) -> "WorkloadManifest":
        if not isinstance(value, dict):
            raise ContractError("workload manifest must be a JSON object")
        required = {
            "schema_version",
            "task_id",
            "definition",
            "workload_uuid",
            "language",
            "binding",
            "nvcc_gencode",
            "target_hardware",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise ContractError(f"workload manifest missing: {', '.join(missing)}")
        if value["schema_version"] != 1:
            raise ContractError("unsupported workload manifest schema_version")
        validate_language_contract(value)
        if not isinstance(value["target_hardware"], list) or not value["target_hardware"]:
            raise ContractError("target_hardware must be a non-empty list")
        for key in ("task_id", "definition", "workload_uuid"):
            if not isinstance(value[key], str) or not value[key].strip():
                raise ContractError(f"{key} must be a non-empty string")
        definition_payload = value.get("definition_payload")
        if not isinstance(definition_payload, dict):
            raise ContractError("workload manifest requires the canonical definition_payload")
        if definition_payload.get("name") != value["definition"]:
            raise ContractError("definition_payload name does not match definition")
        reference = definition_payload.get("reference")
        if not isinstance(reference, str) or not reference.strip():
            raise ContractError("definition_payload requires a non-empty reference")
        return cls(dict(value))

    @property
    def digest(self) -> str:
        return json_digest(self.data)

    def solution(self, source: str) -> dict[str, Any]:
        return {
            "name": "eval_kernel",
            "definition": self.data["definition"],
            "spec": {
                "language": "cuda",
                "binding": "tvm-ffi",
                "target_hardware": self.data["target_hardware"],
                "entry_point": "kernel.cu::run",
                "dependencies": [],
                "destination_passing_style": True,
            },
            "author": "eval",
            "sources": [{"path": "kernel.cu", "content": source}],
        }


def normalized_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        value = {}
    fields = ("input_tokens", "output_tokens", "thinking_tokens", "cache_read_tokens")
    result: dict[str, int] = {}
    for field in fields:
        raw = value.get(field, 0)
        try:
            result[field] = max(int(raw or 0), 0)
        except (TypeError, ValueError):
            result[field] = 0
    raw_total = value.get("total_tokens")
    try:
        total = (
            int(raw_total)
            if raw_total is not None
            else result["input_tokens"] + result["output_tokens"]
        )
    except (TypeError, ValueError):
        total = result["input_tokens"] + result["output_tokens"]
    result["total_tokens"] = max(total, 0)
    return result
