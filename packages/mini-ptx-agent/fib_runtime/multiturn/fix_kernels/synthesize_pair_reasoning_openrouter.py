#!/usr/bin/env python3
"""Synthesize reasoning for wrong/correct CUDA kernel pairs via OpenRouter.

Input is the CSV emitted by collect_success_kernel_pairs.py. Output is JSONL in
the same broad shape as accrl/distill/run_experiment.py's reasoning_pairs.jsonl:
system_prompt, input, reasoning, thinking, and metadata.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm
import yaml


LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "openrouter/z-ai/glm-5.1"
MODEL_PROVIDER_CONFIGS = {
    "openrouter/z-ai/glm-5.1": {"provider": {"only": ["z-ai/fp8", "fireworks"]}},
    "openrouter/z-ai/glm-5.2": {"provider": {"only": ["z-ai/fp8", "fireworks"]}},
    "openrouter/moonshotai/kimi-k2.6": {
        "provider": {"only": ["moonshotai/int4", "moonshotai/highspeed"]},
    },
    "openrouter/moonshotai/kimi-k2.7-code": {
        "provider": {"only": ["moonshotai/int4", "moonshotai/highspeed"]},
    },
}
DISTILL_SYSTEM_PROMPT = """\
You are a CUDA kernel optimization expert.
Your task is to produce high-quality internal monologue that describes how to repair a wrong CUDA kernel into a correct optimized kernel.

Requirements:
- Think at maximum depth and effort.
- Be concrete: reference exact code structure, indexing, memory movement, synchronization, math, architecture features, and failure modes.
- Explain why the wrong kernel is wrong and how to fix it.
- Do not include a full copy of either kernel in the reasoning. You may include code snippets, pseudocode, or partial implementations in your reasoning if it helps illustrate your thought process
- Output the synthesized internal monologue inside <my_reasoning>...</my_reasoning> tags."""

REASONING_PROMPTS = {
    "v1_trajectory_adopted": """## Your Task
Above you can see an expert CUDA developer ("the Expert") is faced with a wrong kernel. The Expert's Kernel is what the Expert wrote next.

Your job: WRITE the reasoning trace as if you ARE the Expert, in first person, BEFORE you wrote the Expert's Kernel. Wrapped between `<my_reasoning>` and `</my_reasoning>`, 
this is the internal monologue the Expert would have gone through — what they noticed, what they considered, what they decided, and why.

This trace will be used as training data, so it must be detailed and authentic.

### Hard requirements for the output:
- **First person** — "I notice the previous kernel had X. I should try Y because..."
- **NEVER refer to the Expert's Kernel** - The internal monologue should lead to the expert's kernel but MUST NOT refer to the kernel because the kernel doesn't exist yet at the reasoning time! The internal monologue MUST NOT contain phrases like "looking at the expert kernel" or "in the expert kenel" which could pollute the reasoning.
- **Meticulous and comprehensive** -- target 40,000 characters (don't mention this in the internal monologue). Capture the Expert's full thought process, not a summary. Be exhaustive.
- **Concrete** — reference specific tile sizes, specific error messages, specific numbers from the kernel and the error message
- **Forward-flowing** — start from the problem/feedback, show every step of the decision chain, end with the strategy about to be implemented
- **Include alternatives considered and rejected** — "I could try A, but that would cause B, so instead I'll do C"
- **Show calculations** — e.g. "shared memory: 128*64*2 = 16KB per stage, 3 stages = 48KB, fits in 228KB SMEM"
- **Reference the CUDA/PTX knowledge** — quote or paraphrase relevant sections when justifying decisions
- **Walk through small code patterns inline** -- Feel free to write kernel code snippets in the thinking process just as an expert put some scratch on paper.

### Format requirements (CRITICAL):
- Output ONLY the <my_reasoning>...</my_reasoning> block, nothing before, nothing after
- DO NOT add a summary or conclusion outside the block

Begin now the internal monologue with `<my_reasoning>` and end with `</my_reasoning>`.""",
    "v3_trajectory_adopted":  """## Your Task
Above you can see an expert CUDA developer ("the Expert") is faced with a wrong kernel. The Expert's Kernel is what the Expert wrote next.

Your job: WRITE the reasoning trace as if you ARE the Expert, in first person, BEFORE you wrote the Expert's Kernel. Wrapped between `<my_reasoning>` and `</my_reasoning>`, 
this is the internal monologue the Expert would have gone through — what they noticed, what they considered, what they decided, and why.

This trace will be used as training data, so it must be detailed and authentic.

### Hard requirements for the output:
- **First person** — "I notice the previous kernel had X. I should try Y because..."
- **VERY LONG** — target 10,000 to 50,000 characters. Capture the Expert's full thought process, not a summary. Be exhaustive.
- **Concrete** — reference specific tile sizes, specific error messages, specific numbers from the kernel and the error message
- **Forward-flowing** — start from the problem/feedback, show every step of the decision chain, end with the strategy about to be implemented
- **Include alternatives considered and rejected** — "I could try A, but that would cause B, so instead I'll do C"
- **Show calculations** — e.g. "shared memory: 128*64*2 = 16KB per stage, 3 stages = 48KB, fits in 228KB SMEM"
- **Reference the reference manual** — quote or paraphrase relevant sections when justifying decisions
- **Walk through small code patterns inline** if it helps illustrate a thought
- **Internal monologue should not refer to the Expert's Kernel** - The internal monologue should lead to the expert's kernel but don't refer to the kernel because the kernel doesn't exist yet at the reasoning time! 
The internal monologue MUST NOT contain phrases like "looking at the expert kernel" which could pollute the reasoning

### Format requirements (CRITICAL):
- Output ONLY the <my_reasoning>...</my_reasoning> block, nothing before, nothing after
- Feel free to write kernel code snipet in the thinking process just as an expert put some scratch on the paper
- DO NOT add a summary or conclusion outside the block

Begin now the internal monologue with `<my_reasoning>` and end with `</my_reasoning>`."""
}



_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_REASONING_RE = re.compile(r"<my_reasoning>(.*?)</my_reasoning>", re.DOTALL)
_SPEEDUP_RE = re.compile(r"speedup:\s*([\d.]+)x")


@dataclass
class ExperimentConfig:
    pairs_csv: Path
    output_jsonl: Path
    config_yaml: Path | None = None
    name: str = "pair_reasoning_openrouter"
    description: str = ""
    reasoning_model: str = DEFAULT_MODEL
    prompt_version: str = "v3_trajectory_adopted"
    max_tokens: int = 196000
    min_reasoning_chars: int = 200
    max_reasoning_chars: int = 190000
    temperature: float = 1.0
    top_p: float = 0.95
    max_concurrent: int = 8
    timeout: float = 600.0
    limit: int | None = None
    overwrite: bool = False
    provenance_json: Path | None = None


def resolve_config_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def load_config(path: Path) -> ExperimentConfig:
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"config YAML not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping")

    raw.pop("enable_thinking", None)
    fields = set(ExperimentConfig.__dataclass_fields__)
    unknown = set(raw).difference(fields)
    if unknown:
        raise ValueError(f"{path}: unknown config keys: {', '.join(sorted(unknown))}")
    missing = {"pairs_csv", "output_jsonl"}.difference(raw)
    if missing:
        raise ValueError(f"{path}: missing required config keys: {', '.join(sorted(missing))}")

    base_dir = path.resolve().parent
    raw["pairs_csv"] = resolve_config_path(raw["pairs_csv"], base_dir=base_dir)
    raw["output_jsonl"] = resolve_config_path(raw["output_jsonl"], base_dir=base_dir)
    raw["provenance_json"] = resolve_config_path(raw.get("provenance_json"), base_dir=base_dir)
    raw["config_yaml"] = path.resolve()
    config = ExperimentConfig(**raw)
    if config.prompt_version not in REASONING_PROMPTS:
        raise ValueError(
            f"{path}: unknown prompt_version {config.prompt_version!r}. "
            f"Available: {', '.join(sorted(REASONING_PROMPTS))}"
        )
    return config


def normalize_openrouter_model(model: str) -> str:
    return model.rsplit(":", 1)[0] if model.startswith("openrouter/") else model


def model_extra_body(model: str) -> dict[str, Any] | None:
    return MODEL_PROVIDER_CONFIGS.get(normalize_openrouter_model(model))


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_yaml", type=Path, help="YAML file matching ExperimentConfig fields.")
    args = parser.parse_args()
    return load_config(args.config_yaml)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        required = {"correct_kernel_path"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        return rows


def pair_key(row: dict[str, str]) -> str:
    parts = [
        row.get("exp_dir", ""),
        row.get("trajectory_id", ""),
        row.get("correct_kernel_path", ""),
        row.get("correct_kernel_version", ""),
    ]
    return "\n".join(parts)


def pair_key_from_metadata(metadata: dict[str, Any]) -> str:
    parts = [
        metadata.get("exp_dir", ""),
        metadata.get("trajectory_id", "") or metadata.get("exp_id", ""),
        metadata.get("correct_kernel_path", ""),
        metadata.get("correct_kernel_version", ""),
    ]
    return "\n".join(str(part) for part in parts)


def load_completed_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = record.get("metadata", {})
            key = metadata.get("pair_key")
            if key:
                completed.add(key)
            normalized_key = pair_key_from_metadata(metadata)
            if normalized_key.strip():
                completed.add(normalized_key)
    return completed


def data_path(path_text: str) -> Path:
    path = Path(os.path.expandvars(path_text)).expanduser()
    data_root = os.environ.get("PTXBENCH_DATA_ROOT")
    if data_root and "eval_runs" in path.parts:
        eval_runs_index = path.parts.index("eval_runs")
        relocated = Path(data_root).expanduser().joinpath(*path.parts[eval_runs_index:])
        if relocated.exists():
            return relocated
    return path


def read_text(path_text: str, label: str) -> str:
    path = data_path(path_text)
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path.read_text(errors="replace")


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def trajectory_path(row: dict[str, str]) -> Path:
    return data_path(row["exp_dir"]) / "trajectories" / f"{row['trajectory_id']}.json"


def load_trajectory(row: dict[str, str]) -> dict:
    path = trajectory_path(row)
    if not path.is_file():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def success_record_path(row: dict[str, str]) -> Path:
    correct_path = data_path(row.get("correct_kernel_path", ""))
    return correct_path.parent / "record.json"


def load_success_record(row: dict[str, str]) -> dict:
    path = success_record_path(row)
    if not path.is_file():
        return {}
    data = read_json(path)
    if not isinstance(data, list):
        return {}
    requested_version = parse_optional_int(row.get("correct_kernel_version"))
    for record in data:
        if isinstance(record, dict) and record.get("version") == requested_version:
            return record
    return data[0] if data and isinstance(data[0], dict) else {}


def success_evaluation(success_record: dict) -> dict:
    traces = success_record.get("traces")
    if not isinstance(traces, list) or not traces:
        return {}
    first_trace = traces[0]
    if not isinstance(first_trace, dict):
        return {}
    evaluation = first_trace.get("evaluation")
    return evaluation if isinstance(evaluation, dict) else {}


def extract_speedup(evaluation: dict) -> float | None:
    performance = evaluation.get("performance")
    if not isinstance(performance, dict):
        return None
    value = performance.get("speedup_factor")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_failure(text: str) -> str | None:
    lower = text.lower()
    if not text or "correct" == lower.strip():
        return None
    if "could not extract" in lower or "no code" in lower:
        return "no_code"
    if "compilation" in lower or ("error:" in lower and "returncode" in lower):
        return "compile"
    if "timeout" in lower:
        return "timeout"
    if "numerical" in lower or "incorrect" in lower:
        return "incorrect"
    if "failed" in lower:
        return "runtime"
    return None


def trajectory_model(traj: dict) -> str:
    model = (
        traj.get("info", {})
        .get("config", {})
        .get("model", {})
        .get("model_name")
    )
    return model or ""


def original_eval_prompt_block(traj: dict) -> str:
    messages = traj.get("messages")
    if not isinstance(messages, list):
        return ""

    system_message = next((m for m in messages if m.get("role") == "system"), None)
    first_user_message = next((m for m in messages if m.get("role") == "user"), None)

    parts = [
        "# Original Eval Conversation Context",
        "The following knowledge, task, wrong kernel, and its error message were shown to the Expert.",
    ]
    if system_message is not None:
        content = system_message.get("content") or ""
        parts.extend(["", "## CUDA/PTX Knowledge", content])
    if first_user_message is not None:
        content = first_user_message.get("content") or ""
        parts.extend(["", "## Task, Wrong Kernel and Error Message", content])
    return "\n".join(parts).strip()


def trajectory_turn_metrics(traj: dict, target_turn: int | None) -> dict[str, Any]:
    messages = traj.get("messages")
    if not isinstance(messages, list) or target_turn is None:
        return {}
    best_speedup = 0.0
    assistant_turn = 0
    for i, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        feedback = ""
        for next_message in messages[i + 1:]:
            if next_message.get("role") == "user":
                feedback = next_message.get("content") or ""
                break
        match = _SPEEDUP_RE.search(feedback)
        speedup = float(match.group(1)) if match else None
        passed = speedup is not None
        improved = passed and speedup > best_speedup

        if assistant_turn == target_turn:
            return {
                "trajectory_speedup": speedup,
                "trajectory_passed": passed,
                "trajectory_improved": improved,
                "trajectory_failure_type": None if passed else classify_failure(feedback),
            }
        if passed and speedup > best_speedup:
            best_speedup = speedup
        assistant_turn += 1
    return {}


def build_metadata(
    row: dict[str, str],
    config: ExperimentConfig,
    *,
    traj: dict,
    success_record: dict,
    reasoning_len: int | None = None,
) -> dict[str, Any]:
    success_eval = success_evaluation(success_record)
    speedup = extract_speedup(success_eval)
    status = success_eval.get("status")
    passed = status == "PASSED" if status else bool(success_record)
    correct_turn = parse_optional_int(success_record.get("turn"))
    turn_metrics = trajectory_turn_metrics(traj, correct_turn)
    if speedup is None:
        speedup = turn_metrics.get("trajectory_speedup")

    metadata: dict[str, Any] = {
        "run_id": Path(row.get("exp_dir", "")).expanduser().name,
        "exp_id": row.get("trajectory_id", ""),
        "turn": correct_turn,
        "correct_turn": correct_turn,
        "speedup": speedup,
        "passed": passed,
        "improved": turn_metrics.get("trajectory_improved", passed and speedup is not None),
        "failure_type": turn_metrics.get("trajectory_failure_type"),
        "definition_name": row.get("definition", ""),
        "model": trajectory_model(traj) or row.get("model", ""),
        "exp_dir": row.get("exp_dir", ""),
        "arch": row.get("arch", ""),
        "workload": row.get("workload", ""),
        "trajectory_id": row.get("trajectory_id", ""),
        "prompt_tag": row.get("prompt_tag", ""),
        "arch_tag": row.get("arch_tag", ""),
        "correct_kernel_path": row.get("correct_kernel_path", ""),
        "correct_kernel_version": row.get("correct_kernel_version", ""),
        "plan_path": row.get("plan_path", ""),
        "turn_csv": row.get("turn_csv", ""),
        "success_record_path": str(success_record_path(row)),
        "reasoning_model": config.reasoning_model,
        "prompt_version": config.prompt_version,
        "temperature": config.temperature,
        "top_p": config.top_p,
    }
    if status:
        metadata["correct_eval_status"] = status
    if reasoning_len is not None:
        metadata["reasoning_len"] = reasoning_len
        metadata["reasoning_chars"] = reasoning_len
    return metadata


def build_prompt_and_metadata(
    row: dict[str, str],
    *,
    prompt_version: str,
    config: ExperimentConfig,
) -> tuple[str, dict[str, Any]]:
    correct_code = read_text(row["correct_kernel_path"], "correct kernel")
    traj = load_trajectory(row)
    success_record = load_success_record(row)
    original_prompt_block = original_eval_prompt_block(traj)

    original_block = f"{original_prompt_block}\n\n" if original_prompt_block else ""

    context = f"""\
{original_block}
## Expert's Kernel
```cpp
{correct_code}
```
"""
    if prompt_version not in REASONING_PROMPTS:
        raise ValueError(f"Unknown prompt version: {prompt_version}. Available: {list(REASONING_PROMPTS.keys())}")
    prompt = context + "\n\n" + REASONING_PROMPTS[prompt_version]
    metadata = build_metadata(row, config, traj=traj, success_record=success_record)
    return prompt, metadata


def extract_reasoning(raw: str) -> str:
    cleaned = _THINK_RE.sub("", raw).strip()
    match = _REASONING_RE.search(cleaned)
    return match.group(1).strip() if match else cleaned


async def generate_one(
    row: dict[str, str],
    prompt: str,
    metadata: dict[str, Any],
    sem: asyncio.Semaphore,
    config: ExperimentConfig,
) -> dict | None:
    async with sem:
        extra_body = model_extra_body(config.reasoning_model)
        model_kwargs: dict[str, Any] = {}
        if extra_body is not None:
            model_kwargs["extra_body"] = extra_body
        try:
            response = await litellm.acompletion(
                model=config.reasoning_model,
                messages=[
                    {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                api_key=os.environ.get("OPENROUTER_API_KEY"),
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                timeout=config.timeout,
                drop_params=True,
                **model_kwargs,
            )
        except Exception as exc:
            LOGGER.warning("LLM call failed for %s: %s", row.get("trajectory_id", ""), exc)
            return None

    message = response.choices[0].message
    content = message.content or ""
    thinking = (
        getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
        or ""
    )
    reasoning = extract_reasoning(content)
    if len(reasoning) < config.min_reasoning_chars:
        LOGGER.warning(
            "Dropped short reasoning for %s (%d chars)",
            row.get("trajectory_id", ""),
            len(reasoning),
        )
        return None
    if len(reasoning) > config.max_reasoning_chars:
        LOGGER.warning(
            "Dropped long reasoning for %s (%d chars)",
            row.get("trajectory_id", ""),
            len(reasoning),
        )
        return None

    metadata = dict(metadata)
    metadata.update(
        {
            "reasoning_len": len(reasoning),
            "reasoning_chars": len(reasoning),
            "response_model": getattr(response, "model", None),
            "request_extra_body": extra_body,
        }
    )
    return {
        "system_prompt": DISTILL_SYSTEM_PROMPT,
        "input": prompt,
        "reasoning": reasoning,
        "thinking": thinking,
        "metadata": metadata,
    }


def jsonable_config(config: ExperimentConfig) -> dict:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def default_provenance_path(config: ExperimentConfig) -> Path:
    if config.provenance_json is not None:
        return config.provenance_json
    return config.output_jsonl.with_name(f"{config.output_jsonl.name}.provenance.json")


def write_provenance(config: ExperimentConfig, stats: dict) -> None:
    path = default_provenance_path(config)
    payload = {
        "run_name": config.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "config": jsonable_config(config),
        "environment": {
            "openrouter_api_key_set": bool(os.environ.get("OPENROUTER_API_KEY")),
        },
        "model_extra_body": model_extra_body(config.reasoning_model),
        "reasoning_prompt_versions": sorted(REASONING_PROMPTS),
        "stats": stats,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("wrote provenance=%s", path)


async def run(config: ExperimentConfig) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set")

    rows = read_csv_rows(config.pairs_csv)
    if config.limit is not None:
        rows = rows[: config.limit]

    completed = set() if config.overwrite else load_completed_keys(config.output_jsonl)
    pending = [row for row in rows if pair_key(row) not in completed]
    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if config.overwrite else "a"

    LOGGER.info("rows=%d completed=%d pending=%d", len(rows), len(completed), len(pending))
    sem = asyncio.Semaphore(config.max_concurrent)
    lock = asyncio.Lock()
    written = 0

    async def process(row: dict[str, str], idx: int) -> None:
        nonlocal written
        try:
            prompt, metadata = build_prompt_and_metadata(
                row,
                prompt_version=config.prompt_version,
                config=config,
            )
        except OSError as exc:
            LOGGER.warning("Skipping row %d: %s", idx + 1, exc)
            return
        record = await generate_one(row, prompt, metadata, sem, config)
        if record is None:
            return
        async with lock:
            with config.output_jsonl.open("a") as f:
                f.write(json.dumps(record) + "\n")
            written += 1
            LOGGER.info(
                "[%d/%d] wrote %s (%d chars)",
                idx + 1,
                len(pending),
                row.get("trajectory_id", ""),
                len(record["reasoning"]),
            )

    if mode == "w":
        config.output_jsonl.write_text("")
    await asyncio.gather(*(process(row, idx) for idx, row in enumerate(pending)))
    stats = {
        "rows": len(rows),
        "completed_before_run": len(completed),
        "pending": len(pending),
        "written": written,
        "output_jsonl": str(config.output_jsonl),
        "provenance_json": str(default_provenance_path(config)),
    }
    write_provenance(config, stats)
    LOGGER.info("done: wrote=%d output=%s", written, config.output_jsonl)


def main() -> None:
    config = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
