#!/usr/bin/env python3
"""Collect instruction-usage notes from wrong/fixed CUDA kernel pairs.

The input can be the CSV emitted by fix_kernels/collect_success_kernel_pairs.py
or any CSV with base prompt/error kernel/fixed kernel columns. Each row is sent
as a single-turn LLM request. Outputs are:

- notes_jsonl: compact records with extracted lessons and metadata.
- output_root/trajectories/exp_NNN.json: mini-swe-agent-style trajectories.
- provenance_json: run/config/stats metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm
import yaml

MULTITURN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MULTITURN_DIR))
from common import make_model  # noqa: E402


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "GLM-5.2"

SYSTEM_PROMPT = """\
You are a senior CUDA kernel reviewer collecting exact instruction-usage notes
from repair examples.

Your job is to compare an error kernel against a fixed kernel and extract
operator-agnostic CUDA/PTX/API instruction contracts that should help a future
model use the same primitive correctly in unrelated kernels. Focus on exact
operand contracts, required instruction sequences, shape/layout preconditions,
diagnostics, and small valid/invalid examples. Do not produce a replacement
kernel.

## Base prompt / task context
{base_prompt}"""

USER_PROMPT_TEMPLATE = """\
We are collecting CUDA instruction usage notes from a kernel repair.

## Error kernel
```cpp
{error_kernel}
```

## Error feedback
{error_feedback}

## Fixed kernel
```cpp
{fixed_kernel}
```

## Output requirements
Return only valid JSON with this schema:
{{
  "summary": "one concise sentence describing the repair",
  "instruction_notes": [
    {{
      "instruction": "specific CUDA/PTX/API primitive or instruction family",
      "shape_context": "dtype/rank/tile/layout/alignment context where this example is valid",
      "correct_example": "minimal self-contained valid snippet for this primitive",
      "wrong_example": "minimal invalid snippet showing the broken primitive use",
      "operand_contract": ["required operand ordering, address spaces, immediates, or descriptor fields"],
      "required_sequence": ["ordered instructions or API calls that must appear together"],
      "diagnostics": ["compiler/runtime/profiler messages this note helps explain"],
      "do_not_do": ["specific invalid shortcut or cargo-cult pattern to avoid"],
      "example_completeness": "complete or partial",
      "missing_details": ["details omitted from the example, empty when complete"]
    }}
  ],
  "tags": ["short-topic-tag"]
}}

Rules:
- Extract instruction-use contracts, not optimization advice. Do not say "use TMA/WGMMA"; say how to use a specific primitive correctly.
- Task/operator names are metadata only. The note must be reusable for non-attention kernels such as reductions, scans, normalization, routing, or fused MoE.
- Preserve exact operand order, address-space qualifiers, immediates, descriptor assumptions, synchronization order, and shape/layout constraints when present.
- Keep tensor-shape context when it is needed to make the instruction example legal.
- Do not use "..." inside correct_example for operands, template parameters, address spaces, descriptor arguments, barrier sequence, or launch arguments.
- If the surrounding code is long, extract a smaller complete helper/wrapper instead of using ellipses.
- correct_example must include the complete instruction/API call signature and all operands that determine legality.
- wrong_example may abbreviate unrelated context, but the invalid instruction/API call itself must be complete.
- Set example_completeness to "partial" and list missing_details when the supplied kernels do not contain enough information for a complete minimal example. Do not invent missing operands.
- Prefer concrete CUDA/PTX/API, synchronization, memory-layout, launch-configuration, and boundary-condition contracts.
- Do not quote large code blocks.
- Do not invent profiling results beyond the supplied feedback.
- Keep 3 to 8 high-signal instruction notes."""

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ExperimentConfig:
    input_csv: Path
    output_root: Path
    notes_jsonl: Path
    config_yaml: Path | None = None
    provenance_json: Path | None = None
    name: str = "kernel_fix_notes"
    description: str = ""
    model: str = DEFAULT_MODEL
    max_concurrent: int = 8
    timeout: float = 600.0
    limit: int | None = None
    overwrite: bool = False
    min_insights: int = 1


def resolve_config_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def load_config(path: Path) -> ExperimentConfig:
    path = path.expanduser()
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    fields = set(ExperimentConfig.__dataclass_fields__)
    unknown = set(raw).difference(fields)
    if unknown:
        raise ValueError(f"{path}: unknown keys: {', '.join(sorted(unknown))}")
    missing = {"input_csv", "output_root", "notes_jsonl"}.difference(raw)
    if missing:
        raise ValueError(f"{path}: missing required keys: {', '.join(sorted(missing))}")
    base_dir = path.resolve().parent
    raw["input_csv"] = resolve_config_path(raw["input_csv"], base_dir=base_dir)
    raw["output_root"] = resolve_config_path(raw["output_root"], base_dir=base_dir)
    raw["notes_jsonl"] = resolve_config_path(raw["notes_jsonl"], base_dir=base_dir)
    raw["provenance_json"] = resolve_config_path(raw.get("provenance_json"), base_dir=base_dir)
    raw["config_yaml"] = path.resolve()
    return ExperimentConfig(**raw)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_yaml", type=Path)
    return load_config(parser.parse_args().config_yaml)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_text_file(path_text: str, *, label: str, required: bool = True) -> str:
    if not str(path_text or "").strip():
        if required:
            raise ValueError(f"missing {label} path")
        return ""
    path = Path(path_text).expanduser()
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"missing {label}: {path}")
        return ""
    return path.read_text(errors="replace")


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])
    has_kernel_pair = {"wrong_kernel_path", "correct_kernel_path"}.issubset(fieldnames)
    has_explicit_pair = {"error_kernel_path", "fixed_kernel_path"}.issubset(fieldnames)
    if not has_kernel_pair and not has_explicit_pair:
        raise ValueError(
            f"{path}: need wrong_kernel_path/correct_kernel_path or "
            "error_kernel_path/fixed_kernel_path columns"
        )
    return rows


def compact_trajectory_prompt(traj_path_text: str) -> str:
    if not traj_path_text:
        return ""
    path = Path(traj_path_text).expanduser()
    if not path.is_file():
        return ""
    try:
        data = read_json(path)
    except (json.JSONDecodeError, OSError):
        return ""
    messages = data.get("messages")
    if not isinstance(messages, list):
        return ""
    system = next((m for m in messages if isinstance(m, dict) and m.get("role") == "system"), None)
    user = next((m for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)
    parts: list[str] = []
    if system and system.get("content"):
        parts.extend(["### System context", str(system["content"])])
    if user and user.get("content"):
        parts.extend(["### First user task", str(user["content"])])
    return "\n\n".join(parts)


def base_prompt_for_row(row: dict[str, str]) -> str:
    if row.get("base_prompt"):
        return row["base_prompt"]
    for key in ("base_prompt_path", "prompt_path", "prompt_file"):
        if row.get(key):
            return read_text_file(row[key], label=key)
    prompt = compact_trajectory_prompt(row.get("wrong_trajectory_path", ""))
    if prompt:
        return prompt
    prompt = compact_trajectory_prompt(row.get("trajectory_path", ""))
    if prompt:
        return prompt
    prompt = compact_trajectory_prompt(
        str(Path(row.get("exp_dir", "")).expanduser() / "trajectories" / f"{row.get('trajectory_id', '')}.json")
        if row.get("exp_dir") and row.get("trajectory_id")
        else ""
    )
    if prompt:
        return prompt
    return row.get("definition", "")


def row_key(row: dict[str, str]) -> str:
    parts = [
        row.get("exp_dir", ""),
        row.get("trajectory_id", ""),
        row.get("wrong_kernel_path") or row.get("error_kernel_path", ""),
        row.get("correct_kernel_path") or row.get("fixed_kernel_path", ""),
        row.get("correct_kernel_version", ""),
    ]
    return "\n".join(parts)


def load_completed_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    keys: set[str] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("metadata", {}).get("row_key")
            if key:
                keys.add(str(key))
    return keys


def parse_json_object(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(content)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("model response JSON is not an object")
    instruction_notes = data.get("instruction_notes")
    if not isinstance(instruction_notes, list):
        raise ValueError("model response missing list field: instruction_notes")
    return data


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return jsonable(value.dict())
    return str(value)


def build_prompt(row: dict[str, str]) -> tuple[str, str, dict[str, Any]]:
    error_kernel_path = row.get("wrong_kernel_path") or row.get("error_kernel_path", "")
    fixed_kernel_path = row.get("correct_kernel_path") or row.get("fixed_kernel_path", "")
    error_log_path = row.get("wrong_log_path") or row.get("error_log_path", "")

    base_prompt = base_prompt_for_row(row)
    error_kernel = read_text_file(error_kernel_path, label="error kernel")
    fixed_kernel = read_text_file(fixed_kernel_path, label="fixed kernel")
    error_feedback = row.get("error_feedback") or row.get("feedback_raw_output") or ""
    if not error_feedback:
        error_feedback = read_text_file(error_log_path, label="error feedback", required=False)
    if not error_feedback:
        error_feedback = "(no error feedback provided)"

    system_prompt = SYSTEM_PROMPT.format(base_prompt=base_prompt)
    prompt = USER_PROMPT_TEMPLATE.format(
        error_kernel=error_kernel,
        error_feedback=error_feedback,
        fixed_kernel=fixed_kernel,
    )
    metadata = {
        "row_key": row_key(row),
        "input_csv_row": row.get("_csv_row"),
        "exp_dir": row.get("exp_dir", ""),
        "trajectory_id": row.get("trajectory_id", ""),
        "definition": row.get("definition", ""),
        "prompt_tag": row.get("prompt_tag", ""),
        "arch": row.get("arch", ""),
        "sass_arch_tag": row.get("sass_arch_tag", ""),
        "wrong_kernel_path": error_kernel_path,
        "wrong_log_path": error_log_path,
        "wrong_trajectory_path": row.get("wrong_trajectory_path", ""),
        "wrong_turn": row.get("wrong_turn", ""),
        "correct_kernel_path": fixed_kernel_path,
        "correct_kernel_version": row.get("correct_kernel_version", ""),
        "base_prompt_sha256": sha256_text(base_prompt),
        "error_kernel_sha256": sha256_text(error_kernel),
        "fixed_kernel_sha256": sha256_text(fixed_kernel),
    }
    return system_prompt, prompt, metadata


async def generate_one(
    row: dict[str, str],
    exp_name: str,
    config: ExperimentConfig,
    sem: asyncio.Semaphore,
) -> tuple[dict[str, Any], dict[str, Any]]:
    system_prompt, prompt, metadata = build_prompt(row)
    model = make_model(config.model)
    model_kwargs: dict[str, Any] = dict(model.config.model_kwargs)
    model_kwargs.setdefault("timeout", config.timeout)

    started = time.time()
    async with sem:
        response = await litellm.acompletion(
            model=model.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            **model_kwargs,
        )

    message = response.choices[0].message
    content = message.content or ""
    payload = parse_json_object(content)
    instruction_notes = payload.get("instruction_notes") or []
    if len(instruction_notes) < config.min_insights:
        raise ValueError(f"only {len(instruction_notes)} instruction notes, expected at least {config.min_insights}")

    metadata.update(
        {
            "exp_name": exp_name,
            "model": config.model,
            "litellm_model": model.config.model_name,
            "response_model": getattr(response, "model", None),
            "request_extra_body": model_kwargs.get("extra_body"),
            "duration_sec": time.time() - started,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": payload.get("summary", ""),
            "tags": payload.get("tags", []),
            "instruction_note_count": len(instruction_notes),
        }
    )
    note_record = {
        "summary": payload.get("summary", ""),
        "instruction_notes": instruction_notes,
        "tags": payload.get("tags", []),
        "metadata": metadata,
    }
    trajectory = {
        "info": {
            "model_stats": jsonable(getattr(response, "usage", None) or {}),
            "config": {
                "model": {
                    "model_name": model.config.model_name,
                    "model_kwargs": {
                        key: value
                        for key, value in model_kwargs.items()
                        if key != "api_key"
                    },
                },
                "step_limit": 1,
            },
            "mini_version": "collect_notes.direct_single_turn",
            "exit_status": "submitted",
            "submission": payload,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": content, "extra": {"response_model": getattr(response, "model", None)}},
        ],
        "trajectory_format": "mini-swe-agent-v1",
    }
    return note_record, trajectory


def jsonable_config(config: ExperimentConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def default_provenance_path(config: ExperimentConfig) -> Path:
    return config.provenance_json or config.notes_jsonl.with_suffix(config.notes_jsonl.suffix + ".provenance.json")


def write_provenance(config: ExperimentConfig, stats: dict[str, Any]) -> None:
    path = default_provenance_path(config)
    model = make_model(config.model)
    payload = {
        "run_name": config.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "config": jsonable_config(config),
        "environment": {
            "openrouter_api_key_set": bool(os.environ.get("OPENROUTER_API_KEY")),
            "llm_api_timeout_set": bool(os.environ.get("LLM_API_TIMEOUT")),
        },
        "resolved_model": model.config.model_name,
        "resolved_model_kwargs": {
            key: value
            for key, value in model.config.model_kwargs.items()
            if key != "api_key"
        },
        "stats": stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("wrote provenance=%s", path)


async def run(config: ExperimentConfig) -> None:
    rows = read_csv_rows(config.input_csv)
    for index, row in enumerate(rows, 1):
        row["_csv_row"] = str(index)
    if config.limit is not None:
        rows = rows[: config.limit]

    completed = set() if config.overwrite else load_completed_keys(config.notes_jsonl)
    pending = [(i, row) for i, row in enumerate(rows) if row_key(row) not in completed]
    config.output_root.mkdir(parents=True, exist_ok=True)
    (config.output_root / "trajectories").mkdir(parents=True, exist_ok=True)
    config.notes_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if config.overwrite:
        config.notes_jsonl.write_text("")

    LOGGER.info("rows=%d completed=%d pending=%d", len(rows), len(completed), len(pending))
    sem = asyncio.Semaphore(config.max_concurrent)
    lock = asyncio.Lock()
    written = 0
    failed = 0

    async def process(index: int, row: dict[str, str]) -> None:
        nonlocal written, failed
        exp_name = f"exp_{index:03d}"
        try:
            note, trajectory = await generate_one(row, exp_name, config, sem)
        except Exception as exc:
            failed += 1
            LOGGER.warning("failed row %d %s: %s", index + 1, row.get("trajectory_id", ""), exc)
            return
        trajectory_path = config.output_root / "trajectories" / f"{exp_name}.json"
        async with lock:
            trajectory_path.write_text(json.dumps(trajectory, indent=2) + "\n")
            with config.notes_jsonl.open("a") as f:
                f.write(json.dumps(note) + "\n")
            written += 1
            LOGGER.info(
                "[%d/%d] wrote %s instruction_notes=%d",
                written,
                len(pending),
                exp_name,
                len(note["instruction_notes"]),
            )

    await asyncio.gather(*(process(index, row) for index, row in pending))
    stats = {
        "rows": len(rows),
        "completed_before_run": len(completed),
        "pending": len(pending),
        "written": written,
        "failed": failed,
        "notes_jsonl": str(config.notes_jsonl),
        "output_root": str(config.output_root),
        "provenance_json": str(default_provenance_path(config)),
    }
    write_provenance(config, stats)
    LOGGER.info("done written=%d failed=%d notes=%s", written, failed, config.notes_jsonl)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
