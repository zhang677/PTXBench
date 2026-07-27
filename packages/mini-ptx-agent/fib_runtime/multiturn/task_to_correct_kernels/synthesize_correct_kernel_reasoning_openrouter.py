#!/usr/bin/env python3
"""Synthesize reasoning for correct kernels.

Input is the CSV emitted by enrich_correct_kernels_for_reasoning.py. Each LLM
request includes the latest prompt-tag base knowledge, the original first user
task message, and one correct kernel.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import csv
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import litellm
import yaml

LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(
    os.environ.get("MINI_PTX_AGENT_ROOT", Path(__file__).resolve().parents[3])
).expanduser().resolve()
MULTITURN_DIR = REPO_ROOT / "fib_runtime" / "multiturn"
PROMPT_CONFIG_DIR = MULTITURN_DIR / "prompt_configs"
HUB_PATH = PROMPT_CONFIG_DIR / "hub.json"
TVM_FFI_EXAMPLE_PATH = (
    REPO_ROOT / "fib_runtime" / "mini_swe_agent_docker" / "envs" / "example.cu"
)

DEFAULT_MODEL = "openrouter/moonshotai/kimi-k2.7-code"
MODEL_PROVIDER_CONFIGS = {
    "openrouter/z-ai/glm-5.1": {"provider": {"only": ["z-ai/fp8", "fireworks"]}},
    "openrouter/z-ai/glm-5.2": {"provider": {"only": ["z-ai/fp8", "fireworks"]}},
    # "openrouter/moonshotai/kimi-k2.6": {
    #     "provider": {"only": ["moonshotai/int4", "moonshotai/highspeed"]},
    # },
    # "openrouter/moonshotai/kimi-k2.7-code": {
    #     "provider": {"only": ["moonshotai/int4", "moonshotai/highspeed"]},
    # },
}

DISTILL_SYSTEM_PROMPT = """\
You are a CUDA kernel optimization expert.
Your task is to produce high-quality internal monologue that explains how an expert would arrive at a correct optimized CUDA kernel for the task.

Requirements:
- Think at maximum depth and effort.
- Be concrete: reference concrete PTX features, indexing choices, memory layout, synchronization, instructions and math.
- Do not include a full copy of the kernel in the reasoning. You may include short snippets, pseudocode, or partial calculations if useful.
- Output the synthesized internal monologue inside <my_reasoning>...</my_reasoning> tags."""

REASONING_PROMPTS = {
    "v1_correct_kernel_adopted": """## Your Task
Above you can see the Base Knowledge that was available to an expert CUDA developer ("the Expert"). The Expert's Kernel is what the Expert wrote next.

Your job: WRITE the reasoning trace as if you ARE the Expert, in first person, BEFORE you wrote the Expert's Kernel. Wrapped between `<my_reasoning>` and `</my_reasoning>`,
this is the internal monologue the Expert would have gone through -- what they noticed, what they considered, what they decided, and why.

This trace will be used as training data for a model that should output correct kernels, so it must be detailed and authentic.

### Hard requirements for the internal monologue:
- **First person** -- "I need to use instructions this way. I choose this memory layout because..."
- **NEVER refer to the Expert's Kernel** - The internal monologue should lead to the expert's kernel but MUST NOT refer to the kernel because the kernel doesn't exist yet at the reasoning time! The internal monologue MUST NOT contain phrases like "looking at the expert kernel" or "in the expert kenel" which could pollute the reasoning.
- **Meticulous and comprehensive** -- target 40,000 characters (don't mention this in the internal monologue). Capture the Expert's full thought process, not a summary. Be exhaustive.
- **Concrete** -- reference specific tile sizes, tensor shapes, indexing formulas, memory layouts, synchronization points, PTX/CUDA instructions, architecture constraints, and code structure from the knowledge and task.
- **Forward-flowing** -- start from the task and available hardware knowledge, show the decision chain, and end with a concrete implementation strategy about to be written.
- **Include alternatives considered and rejected** -- "I could try A, but that would cause B, so instead I'll do C"
- **Show calculations** -- e.g. shared memory footprint, register pressure tradeoffs, tile coverage, vector widths, or loop trip counts when relevant.
- **Walk through small code patterns inline** -- Feel free to write kernel code snippets in the thinking process just as an expert put some scratch on paper.

### Output format requirements (CRITICAL):
- Output ONLY the <my_reasoning>...</my_reasoning> block, nothing before, nothing after.
- DO NOT add a summary or conclusion outside the block.

Begin now the internal monologue with `<my_reasoning>` and end with `</my_reasoning>`."""
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_REASONING_RE = re.compile(r"<my_reasoning>(.*?)</my_reasoning>", re.DOTALL)


@dataclass
class ExperimentConfig:
    input_csv: Path
    output_jsonl: Path
    config_yaml: Path | None = None
    provenance_json: Path | None = None
    name: str = "correct_kernel_reasoning_openrouter"
    description: str = ""
    reasoning_model: str = DEFAULT_MODEL
    prompt_version: str = "v1_correct_kernel_adopted"
    max_tokens: int = 196000
    min_reasoning_chars: int = 200
    max_reasoning_chars: int = 190000
    temperature: float = 1.0
    top_p: float = 0.95
    max_concurrent: int = 8
    timeout: float = 600.0
    limit: int | None = None
    overwrite: bool = False


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        required = {"correct_kernel_path", "prompt_tag"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {', '.join(sorted(missing))}")
        return rows


def resolve_config_path(value: str | Path | None, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path
    return base_dir / path


def load_config(path: Path) -> ExperimentConfig:
    path = path.expanduser()
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping")

    fields = set(ExperimentConfig.__dataclass_fields__)
    unknown = set(raw).difference(fields)
    if unknown:
        raise ValueError(f"{path}: unknown config keys: {', '.join(sorted(unknown))}")
    missing = {"input_csv", "output_jsonl"}.difference(raw)
    if missing:
        raise ValueError(f"{path}: missing required keys: {', '.join(sorted(missing))}")

    base_dir = path.resolve().parent
    raw["input_csv"] = resolve_config_path(raw["input_csv"], base_dir=base_dir)
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


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_yaml", type=Path)
    return load_config(parser.parse_args().config_yaml)


def load_common_system_instructions() -> str:
    common_path = MULTITURN_DIR / "common.py"
    tree = ast.parse(common_path.read_text(), filename=str(common_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "SYSTEM_INSTRUCTIONS":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, str):
                        raise TypeError("SYSTEM_INSTRUCTIONS is not a string")
                    return value
    raise ValueError(f"SYSTEM_INSTRUCTIONS not found in {common_path}")


def assemble_base_prompt_from_hub(
    prompt_tag: str,
    hub: dict[str, list[str]],
    *,
    seen: set[str] | None = None,
) -> tuple[str, list[str]]:
    if seen is None:
        seen = set()
    if prompt_tag in seen:
        cycle = " -> ".join([*seen, prompt_tag])
        raise ValueError(f"cycle in prompt hub references: {cycle}")
    if prompt_tag not in hub:
        base_prompt_path = PROMPT_CONFIG_DIR / f"{prompt_tag}.md"
        if base_prompt_path.is_file():
            return base_prompt_path.read_text(), [str(base_prompt_path)]
        raise KeyError(f"prompt_tag {prompt_tag!r} not found in {HUB_PATH}")

    seen = {*seen, prompt_tag}
    parts: list[str] = []
    sources: list[str] = []
    for partial_doc_path in hub[prompt_tag]:
        if "/" not in partial_doc_path:
            content, nested_sources = assemble_base_prompt_from_hub(
                partial_doc_path,
                hub,
                seen=seen,
            )
            parts.append(content)
            sources.extend(nested_sources)
        else:
            doc_path = REPO_ROOT / "fib_runtime" / partial_doc_path
            if not doc_path.is_file():
                raise FileNotFoundError(
                    f"Doc fragment {doc_path} referenced by hub[{prompt_tag!r}] does not exist"
                )
            parts.append(doc_path.read_text())
            sources.append(str(doc_path))
        parts.append("\n\n")
    return "".join(parts), sources


def infer_gpu_arch(prompt_tag: str, row_arch: str) -> str:
    tag = prompt_tag.lower()
    if "b200" in tag or "blackwell" in tag or "sm100" in tag:
        return "blackwell"
    if row_arch.strip():
        return row_arch.strip().lower()
    return "hopper"


def build_tagged_system_prompt(
    prompt_tag: str,
    *,
    row_arch: str,
    hub: dict[str, list[str]],
) -> dict[str, Any]:
    gpu_arch = infer_gpu_arch(prompt_tag, row_arch)
    base_prompt, sources = assemble_base_prompt_from_hub(prompt_tag, hub)
    base_prompt += f"""
Here is an example of how to use TVM-FFI. You should use TVM-FFI to wrap you kernel.
```cpp
{TVM_FFI_EXAMPLE_PATH.read_text()}
```

"""
    if gpu_arch == "hopper":
        base_prompt += (
            "\n\n You are targeting NVIDIA Hopper architecture GPUs. Use the provided "
            "structural docs to understand the hardware features and how to optimize "
            "for them. \n\n"
        )
    elif gpu_arch == "blackwell":
        base_prompt += (
            "\n\n You are targeting NVIDIA Blackwell architecture GPUs. Use the provided "
            "structural docs to understand the hardware features and how to optimize "
            "for them. \n\n"
        )
    else:
        raise ValueError(f"Unsupported GPU architecture: {gpu_arch}")

    content = "{% raw %}" + load_common_system_instructions() + base_prompt + "{% endraw %}"
    return {
        "content": content,
        "prompt_tag": prompt_tag,
        "gpu_arch": gpu_arch,
        "sha256": sha256_text(content),
        "sources": sources + [str(TVM_FFI_EXAMPLE_PATH), str(MULTITURN_DIR / "common.py")],
    }


def prompt_cache_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("prompt_tag", ""), row.get("arch", ""))


def read_text(path_text: str, label: str) -> str:
    path = Path(os.path.expandvars(path_text)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path.read_text(errors="replace")


def first_user_task_message(row: dict[str, str]) -> str:
    raw_path = row.get("trajectory_path", "").strip()
    if not raw_path:
        return ""
    path = Path(os.path.expandvars(raw_path)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"missing trajectory: {path}")
    data = read_json(path)
    messages = data.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{path}: missing messages list")
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def source_id(row: dict[str, str]) -> str:
    parts = [
        row.get("exp_dir", ""),
        row.get("trajectory_id", ""),
        row.get("turn", ""),
        row.get("correct_kernel_path", ""),
        row.get("prompt_tag", ""),
    ]
    return "\n".join(parts)


def load_completed_source_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = record.get("metadata") or {}
            value = metadata.get("source_id")
            if isinstance(value, str) and value:
                done.add(value)
    return done


def build_prompt_and_metadata(
    row: dict[str, str],
    *,
    tagged_prompt: dict[str, Any],
    prompt_version: str,
) -> tuple[str, dict[str, Any]]:
    kernel_code = read_text(row["correct_kernel_path"], "correct kernel")
    task_message = first_user_task_message(row)
    user_prompt = f"""\
# Base Knowledge
{tagged_prompt['content']}

# Task
{task_message}

# Expert's Kernel
```cpp
{kernel_code}
```

{REASONING_PROMPTS[prompt_version]}
"""
    metadata = {
        "source_id": source_id(row),
        "run_id": Path(row.get("exp_dir", "")).name,
        "exp_dir": row.get("exp_dir", ""),
        "trajectory_id": row.get("trajectory_id", ""),
        "turn": int(row["turn"]),
        "correct_turn": int(row["turn"]),
        "definition_name": row.get("definition", ""),
        "workload": row.get("workload", ""),
        "model": row.get("model", ""),
        "arch": row.get("arch", ""),
        "arch_tag": row.get("arch_tag", ""),
        "prompt_tag": row.get("prompt_tag", ""),
        "speedup": float(row["speedup"]) if row.get("speedup") else None,
        "correct_kernel_path": row.get("correct_kernel_path", ""),
        "trajectory_path": row.get("trajectory_path", ""),
        "original_task_source": "trajectory.first_user_message",
        "turn_csv": row.get("turn_csv", ""),
        "prompt_version": prompt_version,
        "base_prompt_sha256": tagged_prompt["sha256"],
        "base_prompt_sources": tagged_prompt["sources"],
        "base_prompt_gpu_arch": tagged_prompt["gpu_arch"],
    }
    return user_prompt, metadata


def normalize_openrouter_model(model: str) -> str:
    return model.rsplit(":", 1)[0] if model.startswith("openrouter/") else model


def model_extra_body(model: str) -> dict[str, Any] | None:
    return MODEL_PROVIDER_CONFIGS.get(normalize_openrouter_model(model))


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
) -> dict[str, Any] | None:
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
            LOGGER.warning(
                "LLM call failed for %s turn=%s: %s",
                row.get("trajectory_id", ""),
                row.get("turn", ""),
                exc,
            )
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
            "Dropped short reasoning for %s turn=%s (%d chars)",
            row.get("trajectory_id", ""),
            row.get("turn", ""),
            len(reasoning),
        )
        return None
    if len(reasoning) > config.max_reasoning_chars:
        LOGGER.warning(
            "Dropped long reasoning for %s turn=%s (%d chars)",
            row.get("trajectory_id", ""),
            row.get("turn", ""),
            len(reasoning),
        )
        return None

    metadata = dict(metadata)
    metadata.update(
        {
            "reasoning_model": config.reasoning_model,
            "response_model": getattr(response, "model", None),
            "request_extra_body": extra_body,
            "reasoning_len": len(reasoning),
            "reasoning_chars": len(reasoning),
        }
    )
    return {
        "system_prompt": DISTILL_SYSTEM_PROMPT,
        "input": prompt,
        "reasoning": reasoning,
        "thinking": thinking,
        "metadata": metadata,
    }


def repo_provenance(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    try:
        sha = git("rev-parse", "HEAD").strip()
        dirty = [
            line
            for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
            if line
        ]
    except Exception as exc:
        return {"error": str(exc)}
    return {"sha": sha, "dirty_files": dirty}


def jsonable_config(config: ExperimentConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def default_provenance_path(config: ExperimentConfig) -> Path:
    if config.provenance_json is not None:
        return config.provenance_json
    return config.output_jsonl.with_name(f"{config.output_jsonl.name}.provenance.json")


def write_provenance(config: ExperimentConfig, stats: dict[str, Any]) -> None:
    path = default_provenance_path(config)
    payload = {
        "run_name": config.name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": str(Path(__file__).resolve()),
        "cwd": str(Path.cwd()),
        "argv": sys.argv,
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "config": jsonable_config(config),
        "environment": {
            "openrouter_api_key_set": bool(os.environ.get("OPENROUTER_API_KEY")),
        },
        "repo_provenance": repo_provenance(REPO_ROOT),
        "model_extra_body": model_extra_body(config.reasoning_model),
        "stats": stats,
    }
    if config.input_csv.is_file():
        payload["input_csv_sha256"] = file_sha256(config.input_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    LOGGER.info("wrote provenance=%s", path)


async def run(config: ExperimentConfig) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set")
    rows = read_csv_rows(config.input_csv)
    if config.limit is not None:
        rows = rows[: config.limit]

    with HUB_PATH.open() as f:
        hub = json.load(f)
    if not isinstance(hub, dict):
        raise ValueError(f"{HUB_PATH}: expected JSON object")

    prompt_cache: dict[tuple[str, str], dict[str, Any]] = {}
    completed = set() if config.overwrite else load_completed_source_ids(config.output_jsonl)
    pending = [row for row in rows if source_id(row) not in completed]

    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if config.overwrite:
        config.output_jsonl.write_text("")

    LOGGER.info("rows=%d completed=%d pending=%d", len(rows), len(completed), len(pending))
    sem = asyncio.Semaphore(config.max_concurrent)
    lock = asyncio.Lock()
    written = 0

    async def process(row: dict[str, str], idx: int) -> None:
        nonlocal written
        try:
            key = prompt_cache_key(row)
            if key not in prompt_cache:
                prompt_cache[key] = build_tagged_system_prompt(
                    row["prompt_tag"],
                    row_arch=row.get("arch", ""),
                    hub=hub,
                )
            prompt, metadata = build_prompt_and_metadata(
                row,
                tagged_prompt=prompt_cache[key],
                prompt_version=config.prompt_version,
            )
        except Exception as exc:
            LOGGER.warning("Skipping row %d before LLM call: %s", idx + 1, exc)
            return

        record = await generate_one(row, prompt, metadata, sem, config)
        if record is None:
            return
        async with lock:
            with config.output_jsonl.open("a") as f:
                f.write(json.dumps(record) + "\n")
            written += 1
            LOGGER.info(
                "[%d/%d] wrote %s turn=%s (%d chars)",
                idx + 1,
                len(pending),
                row.get("trajectory_id", ""),
                row.get("turn", ""),
                len(record["reasoning"]),
            )

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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
