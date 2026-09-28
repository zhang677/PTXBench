from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from .eval_exchange import EvalExchangeBroker
from .feedback import feedback_style_from_config
from .models import WorkloadManifest, normalized_usage, validate_language_contract
from .registry import Registry, TERMINAL_STATES
from .task_view import agent_task_payload, task_view_from_config, validate_task_view
from .util import append_jsonl, atomic_write_json, json_digest, read_json, utc_now


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("experiment config must be a JSON object")
    value["_config_dir"] = str(path.resolve().parent)
    return value


def _resolve(config: dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(config["_config_dir"]) / path


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _required_config_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _disable_web(config: dict[str, Any]) -> bool:
    value = config.get("disable_web", False)
    if not isinstance(value, bool):
        raise ValueError("disable_web must be true or false")
    return value


def _codex_service_tier(config: dict[str, Any]) -> str | None:
    value = config.get("serviceTier")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("serviceTier must be a non-empty string")
    return value.strip()


_WORKLOAD_SELECTOR_KEYS = ("task_id", "definition", "workload_uuid")
_AGY_PTXBENCH_MCP_TIMEOUT_SECONDS = 960


def _workload_config_for_run(
    config: dict[str, Any], run: dict[str, Any]
) -> dict[str, Any]:
    """Return config with one run's workload selectors applied.

    Workload selectors may be supplied either together on the run or together
    at the experiment top level for compatibility with existing configs.
    """

    present = [key for key in _WORKLOAD_SELECTOR_KEYS if key in run]
    if present and len(present) != len(_WORKLOAD_SELECTOR_KEYS):
        missing = [key for key in _WORKLOAD_SELECTOR_KEYS if key not in run]
        run_id = run.get("run_id", "<unknown>")
        raise ValueError(
            f"run {run_id} has incomplete workload selectors; missing: "
            + ", ".join(missing)
        )
    if not present:
        return config
    merged = dict(config)
    for key in _WORKLOAD_SELECTOR_KEYS:
        merged[key] = run[key]
    return merged


def _profile_get_json(base_url: str, path: str) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        response = requests.get(url, timeout=(3, 10))
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, TypeError, ValueError) as exc:
        raise ValueError(f"profiling request failed for {url}: {exc}") from exc


def _service_workload_manifest(config: dict[str, Any]) -> WorkloadManifest:
    """Resolve one immutable workload manifest from the profiling service."""

    base_url = _required_config_string(config, "profile_base_url").rstrip("/")
    definition_name = _required_config_string(config, "definition")
    workload_uuid = _required_config_string(config, "workload_uuid")

    health = _profile_get_json(base_url, "/health")
    if not isinstance(health, dict):
        raise ValueError("profiling-service health response is not an object")
    backends = health.get("backends") or health.get("workers") or []
    healthy = (
        health.get("status") == "ok"
        and isinstance(backends, list)
        and bool(backends)
        and all(
            isinstance(backend, dict) and backend.get("healthy") is True
            for backend in backends
        )
        and int(health.get("queue_size") or 0) == 0
    )
    if not healthy:
        raise ValueError(
            "profiling service is not ready: "
            + json.dumps(health, sort_keys=True)
        )

    encoded_definition = quote(definition_name, safe="")
    definition_path = f"/definitions/{encoded_definition}"
    workloads_path = f"{definition_path}/workloads"
    definition_payload = _profile_get_json(base_url, definition_path)
    if not isinstance(definition_payload, dict):
        raise ValueError(
            f"profiling definition {definition_name} is not an object"
        )
    if definition_payload.get("name") != definition_name:
        raise ValueError(
            "profiling service returned the wrong definition: "
            f"expected {definition_name}, got {definition_payload.get('name')}"
        )

    workloads = _profile_get_json(base_url, workloads_path)
    if not isinstance(workloads, list):
        raise ValueError(
            f"profiling workloads for {definition_name} are not a list"
        )
    matches = [
        workload
        for workload in workloads
        if isinstance(workload, dict)
        and (workload.get("uuid") or workload.get("workload_uuid"))
        == workload_uuid
    ]
    if len(matches) != 1:
        raise ValueError(
            f"profiling service has {len(matches)} matches for workload "
            f"{workload_uuid} under definition {definition_name}"
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task_id": _required_config_string(config, "task_id"),
        "definition": definition_name,
        "workload_uuid": workload_uuid,
        "language": _required_config_string(config, "language"),
        "target_hardware": config.get("target_hardware"),
        "definition_payload": definition_payload,
        "workload_payload": matches[0],
        "profile_source": {
            "base_url": base_url,
            "definition_path": definition_path,
            "workloads_path": workloads_path,
        },
    }
    validate_language_contract(config)
    if config["language"] == "cuda":
        manifest["binding"] = config["binding"]
        manifest["nvcc_gencode"] = config["nvcc_gencode"]
    if "target_speedup" in config:
        manifest["target_speedup"] = config["target_speedup"]
    return WorkloadManifest.parse(manifest)


def _workload_manifest(config: dict[str, Any]) -> WorkloadManifest:
    if "workload_manifest" in config:
        raise ValueError(
            "workload_manifest is forbidden; select definition and workload_uuid "
            "from profile_base_url"
        )
    return _service_workload_manifest(config)


def _load_prompt_hub(path: Path) -> dict[str, list[str]]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read prompt-tag hub from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"prompt-tag hub must be a JSON object: {path}")
    hub: dict[str, list[str]] = {}
    for tag, items in value.items():
        if (
            not isinstance(tag, str)
            or not isinstance(items, list)
            or not all(isinstance(item, str) for item in items)
        ):
            raise ValueError(
                f"invalid prompt-tag hub entry for {tag!r}; expected a list of strings"
            )
        hub[tag] = items
    return hub


def _prompt_tag_files(
    prompt_tag: str,
    hub: dict[str, list[str]],
    *,
    stack: tuple[str, ...] = (),
) -> list[str]:
    if prompt_tag in stack:
        cycle = " -> ".join((*stack, prompt_tag))
        raise ValueError(f"cycle in prompt-tag hub: {cycle}")
    if prompt_tag not in hub:
        raise ValueError(f"prompt_tag is not present in the prompt-tag hub: {prompt_tag}")

    files: list[str] = []
    for item in hub[prompt_tag]:
        if item in hub:
            files.extend(
                _prompt_tag_files(item, hub, stack=(*stack, prompt_tag))
            )
            continue
        if Path(item).suffix.lower() not in {".md", ".h"}:
            raise ValueError(
                f"prompt tag {prompt_tag!r} references unsupported context file: {item}"
            )
        files.append(item)
    return files


def _prompt_context_sources(
    prompt_tag: str,
    hub: dict[str, list[str]],
    root: Path,
) -> list[tuple[str, Path]]:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"prompt_files_root is not a directory: {root}")
    sources: list[tuple[str, Path]] = []
    for relative_value in _prompt_tag_files(prompt_tag, hub):
        relative = Path(relative_value)
        if relative.is_absolute():
            raise ValueError(f"prompt-tag context path must be relative: {relative_value}")
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"prompt-tag context path escapes prompt_files_root: {relative_value}"
            ) from exc
        if not source.is_file():
            raise ValueError(f"prompt-tag context file does not exist: {source}")
        sources.append((relative.as_posix(), source))
    return sources


def _materialize_prompt_context(
    workspace: Path,
    sources: list[tuple[str, Path]],
    example_source_path: Path | None,
) -> dict[str, Any]:
    markdown_sources = [item for item in sources if item[1].suffix.lower() == ".md"]
    header_sources = [item for item in sources if item[1].suffix.lower() == ".h"]
    markdown = "".join(path.read_text() + "\n\n" for _, path in markdown_sources)
    knowledge = markdown.rstrip() + "\n"
    if example_source_path is not None:
        example_intro = (
            "\n\n## Required TVM-FFI wrapper example\n\n"
            "Use this exact registration and destination-passing pattern.\n\n```cpp\n"
        )
        knowledge = markdown.rstrip() + (
            example_intro
            + example_source_path.read_text().rstrip()
            + "\n```\n"
        )
    (workspace / "knowledge_pack.md").write_text(knowledge)

    header_names: dict[str, Path] = {}
    for _, source in header_sources:
        previous = header_names.get(source.name)
        if previous is not None and previous != source:
            raise ValueError(
                f"prompt-tag headers have the same basename: {previous} and {source}"
            )
        header_names[source.name] = source
        shutil.copyfile(source, workspace / source.name)

    def provenance(relative: str, source: Path, workspace_path: str) -> dict[str, str]:
        content = source.read_text()
        return {
            "path": relative,
            "source": str(source),
            "workspace_path": workspace_path,
            "sha256": _sha256_text(content),
        }

    result = {
        "knowledge_pack_sha256": _sha256_text(knowledge),
        "knowledge_pack_sources": [
            provenance(relative, source, "knowledge_pack.md")
            for relative, source in markdown_sources
        ],
        "header_files": [
            provenance(relative, source, source.name)
            for relative, source in header_sources
        ],
    }
    if example_source_path is not None:
        result.update({
            "tvm_ffi_example_sha256": _sha256_text(example_source_path.read_text()),
            "tvm_ffi_example_source": str(example_source_path),
        })
    return result


def _agent_kind(config: dict[str, Any]) -> str:
    value = str(config.get("agent", "agy")).strip().lower()
    if value == "antigravity":
        value = "agy"
    if value not in {"agy", "codex"}:
        raise ValueError(f"unsupported coding agent: {value}")
    return value


def _add_goal_command(config: dict[str, Any], agent: str) -> bool:
    value = config.get("add_goal_command", agent == "agy")
    if not isinstance(value, bool):
        raise ValueError("add_goal_command must be true or false")
    return value


def _codex_use_mcp(config: dict[str, Any]) -> bool:
    value = config.get("codex_use_mcp", False)
    if not isinstance(value, bool):
        raise ValueError("codex_use_mcp must be true or false")
    return value


def _agent_harness(agent: str) -> str:
    return "antigravity" if agent == "agy" else "codex"


_AGY_NO_WEB_AGENT = "ptxbench-no-web"


def _agy_no_web_agent_definition() -> str:
    return """---
name: ptxbench-no-web
description: Implements and evaluates one PTXBench CUDA kernel without web or shell access.
tools:
  - list_dir
  - view_file
  - grep_search
  - write_to_file
  - replace_file_content
  - multi_replace_file_content
mainAgent: true
subagent: false
inheritMcp: true
commandExecutionPolicy: off
---

# PTXBench no-web agent

Work only with the files under /workspace. Do not use terminal, command,
browser, URL-reading, web-search, subagent, or external-network tools.

Create and revise /workspace/kernel.cu with the file tools. Evaluate it only by
calling the `ptxbench-eval` MCP server's `evaluate_kernel` tool. That tool
takes no arguments, evaluates the fixed kernel and workload paths, and waits
for complete feedback. Never try to invoke `ptxbench-eval` through a shell.
"""


def _agent_home(registry: Registry, run_id: str, agent: str) -> Path:
    directory = "antigravity_home" if agent == "agy" else "codex_home"
    return registry.root / directory / run_id


def _agent_credential_env(agent: str) -> str | None:
    return "GEMINI_API_KEY" if agent == "agy" else None


def _codex_plan_auth(config: dict[str, Any]) -> str:
    value = config.get("codex_auth_file")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("codex_auth_file is required for Codex plan authentication")
    path = _resolve(config, value)
    try:
        payload = path.read_text()
        auth = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read Codex plan authentication from {path}") from exc
    if not isinstance(auth, dict) or auth.get("auth_mode") != "chatgpt":
        raise ValueError(f"Codex auth file is not a ChatGPT plan login: {path}")
    # Pass only the ChatGPT-plan credentials to the container. Codex-managed
    # auth files can contain an OPENAI_API_KEY placeholder even in ChatGPT mode;
    # do not propagate that field, regardless of whether it is null or populated.
    auth.pop("OPENAI_API_KEY", None)
    return json.dumps(auth, separators=(",", ":"))


def _agent_argv(config: dict[str, Any], prompt: str) -> list[str]:
    agent = _agent_kind(config)
    disable_web = _disable_web(config)
    if agent == "agy":
        command = ["agy", "-p", prompt, "--model", config["model"]]
        effort = config.get("effort")
        if effort:
            command.extend(["--effort", effort])
        command.extend(
            [
                "--output-format",
                "stream-json",
                *(
                    ["--agent", _AGY_NO_WEB_AGENT]
                    if disable_web
                    else []
                ),
                *([] if disable_web else ["--dangerously-skip-permissions"]),
                "--print-timeout",
                config.get("print_timeout", "24h"),
            ]
        )
        return command

    command = [
        "ptxbench-codex",
        "--prompt",
        prompt,
        "--model",
        config["model"],
    ]
    service_tier = _codex_service_tier(config)
    if service_tier is not None:
        command.extend(["--service-tier", service_tier])
    effort = config.get("effort")
    if effort:
        command.extend(["--effort", effort])
    if disable_web:
        command.append("--disable-web")
    if _codex_use_mcp(config):
        command.append("--evaluation-via-mcp")
    if _add_goal_command(config, agent):
        command.append("--goal")
    command.extend(["--print-timeout", config.get("print_timeout", "24h")])
    return command


def _goal_instructions(target_speedup: float | int | None) -> str:
    completion = "passes all correctness tests"
    if target_speedup is not None:
        completion += f" and reports `min_speedup >= {target_speedup}`"
    return f"""The goal is complete only when the latest `ptxbench-eval` result {completion}.
Compilation errors, numerical failures, timeouts, placeholder kernels,
scaffolds, or conclusions that the task is too complex are not completion.
Continue revising and evaluating until the goal is complete. Believe in yourself, you have succeeded before.
"""


def _prompt_for_agent(
    agent: str,
    base_prompt: str,
    *,
    add_goal_command: bool = False,
    disable_web: bool = False,
    evaluation_via_mcp: bool = False,
) -> str:
    prompt = base_prompt
    if disable_web:
        if agent == "agy":
            policy = (
                "Don't use web search. Only refer to local files.\n"
                "Do not use browser, URL-reading, terminal, or command tools. The only "
                "permitted external operation is calling the "
                "`ptxbench-eval` MCP server's `evaluate_kernel` tool."
            )
        elif evaluation_via_mcp:
            policy = (
                "Don't use web search. Only refer to local files.\n"
                "Do not use browser or URL-reading tools. Submit kernels only by calling "
                "the `ptxbench-eval` MCP server's `evaluate_kernel` tool."
            )
        else:
            policy = (
                "Don't use web search. Only refer to local files.\n"
                "Do not use browser or URL-reading tools. The only permitted external "
                "operation is submitting kernels with the provided `ptxbench-eval` "
                "command."
            )
        prompt = f"{prompt.rstrip()}\n\n{policy}\n"
    if add_goal_command and agent == "agy":
        return f"/goal {prompt}"
    return prompt


def _agent_prompt(
    *,
    target_speedup: float | int | None,
    evaluation_via_mcp: bool = False,
    task_view: str = "full",
) -> str:
    task_view = validate_task_view(task_view)
    goal_instructions = _goal_instructions(target_speedup)
    if evaluation_via_mcp:
        evaluation_instructions = """Read and edit /workspace/kernel.cu. Submit every candidate by calling the
`ptxbench-eval` MCP server's `evaluate_kernel` tool. It accepts no arguments
and always evaluates /workspace/kernel.cu against /opt/ptxbench/workload.json.

The evaluation tool will return feedback eventually although it might sometimes take minutes. Only submit complete kernels. It can only provide feedback for kernels that implrment the workload.
While the profiling is running, you may also reflect and improve /workspace/kernel.cu.
You must not submit another candidate until the current feedback arrives because the gateway rejects concurrent submissions for the same run with HTTP 409. If you receive HTTP 409, wait for the in-flight request's feedback instead of retrying immediately."""
    else:
        evaluation_instructions = """Read and edit /workspace/kernel.cu. Submit every candidate using exactly:

   `ptxbench-eval /workspace/kernel.cu /opt/ptxbench/workload.json`

`ptxbench-eval` will return feedback eventually although it might sometimes take minutes. Only submit complete kernels. `ptxbench-eval` can only provide feedback for kernels that implrment the workload.
While the profiling is running, you may also reflect and improve /workspace/kernel.cu.
You must not submit another candidate until the current feedback arrives because the gateway rejects concurrent submissions for the same run with HTTP 409. If you receive HTTP 409, wait for the in-flight request's feedback instead of retrying immediately."""
    task_description = (
        "The file is the authoritative task definition; use it to determine the reference behavior,\n"
        "input and output contract, symbolic shapes, dtypes, and constraints."
        if task_view == "multiturn"
        else "The file is the authoritative task specification; use it to determine the reference behavior,\n"
        "input and output contract, shapes, dtypes, constraints, and target hardware."
    )
    implementation_instructions = """Your task is to implement an optimized CUDA-PTX kernel with a host `run`
function that implements the reference specified in that file.

Create and work in /workspace/kernel.cu. The GPU architecture knowledge and the required TVM-FFI wrapper example are merged into /workspace/knowledge_pack.md. Helper headers are listed as individual .h files in /workspace.
The headers and wrapper example are reference material. Keep kernel.cu self-contained and not rely on these local helper files because only kernel.cu is submitted and the evaluation environment doesn't have those local heeader files in /workspace.
You must optimize the CUDA kernels with architecure-specific PTX instructions. Naive for-loop kernels are not acceptable."""
    environment_instructions = "The current environment doesn't have any CUDA related software."
    return f"""{goal_instructions}

Do not write or edit any code before reading `/opt/ptxbench/workload.json` completely.
{task_description}

{implementation_instructions}

{evaluation_instructions}
Do not use unbounded `tail -f`, `watch`, or infinite polling loops. Use bounded inspection such as `tail -n`, a finite retry loop, or `timeout` so control always returns to you.
{environment_instructions}
Do not call the profiling service directly or create an alternative evaluator or benchmark.
Use the returned feedback to revise the implementation. You can also store positive and negative experiences in `/workspace` to augment yourself.
Keep optimizing after correctness.
"""


def prepare(config: dict[str, Any]) -> Registry:
    validate_language_contract(config)
    feedback_style = feedback_style_from_config(config)
    task_view = task_view_from_config(config)
    agent = _agent_kind(config)
    add_goal_command = _add_goal_command(config, agent)
    codex_use_mcp = _codex_use_mcp(config)
    if codex_use_mcp and agent != "codex":
        raise ValueError("codex_use_mcp is only supported for the Codex agent")
    disable_web = _disable_web(config)
    if agent == "codex":
        _codex_service_tier(config)
    harness = _agent_harness(agent)
    root = Path(config["experiment_root"]).expanduser().resolve()
    registry = Registry(root)
    hub_value = config.get("prompt_hub")
    if not isinstance(hub_value, str) or not hub_value.strip():
        raise ValueError("prompt_hub is required")
    hub_path = _resolve(config, hub_value).resolve()
    hub = _load_prompt_hub(hub_path)
    prompt_root_value = config.get("prompt_files_root")
    if not isinstance(prompt_root_value, str) or not prompt_root_value.strip():
        raise ValueError("prompt_files_root is required")
    prompt_root = _resolve(config, prompt_root_value).resolve()
    example_value = _required_config_string(config, "tvm_ffi_example")
    example_source_path = _resolve(config, example_value).resolve()
    example_source = example_source_path.read_text()
    manifests: dict[tuple[str, str, str], WorkloadManifest] = {}
    for run in config["runs"]:
        workload_config = _workload_config_for_run(config, run)
        selector = tuple(
            _required_config_string(workload_config, key)
            for key in _WORKLOAD_SELECTOR_KEYS
        )
        manifest = manifests.get(selector)
        if manifest is None:
            manifest = _workload_manifest(workload_config)
            manifests[selector] = manifest
        prompt = _prompt_for_agent(
            agent,
            _agent_prompt(
                target_speedup=manifest.data.get("target_speedup"),
                evaluation_via_mcp=(codex_use_mcp if agent == "codex" else disable_web),
                task_view=task_view,
            ),
            add_goal_command=add_goal_command,
            disable_web=disable_web,
            evaluation_via_mcp=codex_use_mcp,
        )
        extra_user_prompt_tag = run.get("extra_user_prompt_tag")
        extra_user_prompt_sources: list[tuple[str, Path]] = []
        if extra_user_prompt_tag is not None:
            if (
                not isinstance(extra_user_prompt_tag, str)
                or not extra_user_prompt_tag.strip()
            ):
                raise ValueError("extra_user_prompt_tag must be a non-empty string")
            extra_user_prompt_tag = extra_user_prompt_tag.strip()
            extra_user_prompt_sources = _prompt_context_sources(
                extra_user_prompt_tag, hub, prompt_root
            )
            extra_user_prompt = "\n\n".join(
                source.read_text().rstrip()
                for _, source in extra_user_prompt_sources
            )
            if extra_user_prompt:
                prompt = f"{prompt.rstrip()}\n\n{extra_user_prompt}\n"
        run_id = run["run_id"]
        prompt_tag = run["prompt_tag"]
        context_sources = _prompt_context_sources(prompt_tag, hub, prompt_root)
        if registry.run_path(run_id).exists():
            existing = registry.get_run(run_id)
            if existing.get("task_view", "full") != task_view:
                raise ValueError(f"existing run {run_id} has a different task_view; use a new experiment_root")
            if existing.get("feedback_style", "default") != feedback_style:
                raise ValueError(f"existing run {run_id} has a different feedback_style")
            if existing["manifest_digest"] != manifest.digest:
                raise ValueError(f"existing run {run_id} has a different workload")
            _validate_agent_task(registry, run_id, manifest, task_view)
            if existing.get("agent_prompt_sha256") != _sha256_text(prompt):
                raise ValueError(f"existing run {run_id} has a different agent prompt")
            if existing.get("harness") != harness:
                raise ValueError(f"existing run {run_id} has a different coding agent")
            if existing.get("model") != config["model"]:
                raise ValueError(f"existing run {run_id} has a different model")
            existing_provenance = existing.get("artifact_provenance") or {}
            expected_context_files = [
                {
                    "path": relative,
                    "sha256": _sha256_text(source.read_text()),
                }
                for relative, source in context_sources
            ]
            if existing_provenance.get("prompt_context_files") != expected_context_files:
                raise ValueError(f"existing run {run_id} has different prompt-tag files")
            if extra_user_prompt_tag is not None:
                expected_extra_user_prompt_files = [
                    {
                        "path": relative,
                        "sha256": _sha256_text(source.read_text()),
                    }
                    for relative, source in extra_user_prompt_sources
                ]
                if (
                    existing_provenance.get("extra_user_prompt_tag")
                    != extra_user_prompt_tag
                    or existing_provenance.get("extra_user_prompt_files")
                    != expected_extra_user_prompt_files
                ):
                    raise ValueError(
                        f"existing run {run_id} has different extra user-prompt files"
                    )
            if existing_provenance.get("tvm_ffi_example_sha256") != _sha256_text(
                example_source
            ):
                raise ValueError(f"existing run {run_id} has a different TVM-FFI example")
            token_path = registry.root / "secrets" / f"{run_id}.token"
            env_path = registry.root / "secrets" / f"{run_id}.env"
            if token_path.exists() and not env_path.exists():
                env_path.write_text(
                    f"PTXBENCH_RUN_TOKEN={token_path.read_text().strip()}\n"
                )
                env_path.chmod(0o600)
            continue
        workspace = registry.root / "workspaces" / run_id
        workspace.mkdir(parents=True, exist_ok=True)
        context_provenance = _materialize_prompt_context(
            workspace, context_sources, example_source_path
        )
        context_provenance.update(
            {
                "prompt_hub_sha256": _sha256_text(hub_path.read_text()),
                "prompt_hub_source": str(hub_path),
                "prompt_context_files": [
                    {
                        "path": relative,
                        "sha256": _sha256_text(source.read_text()),
                    }
                    for relative, source in context_sources
                ],
            }
        )
        if extra_user_prompt_tag is not None:
            context_provenance.update(
                {
                    "extra_user_prompt_tag": extra_user_prompt_tag,
                    "extra_user_prompt_files": [
                        {
                            "path": relative,
                            "sha256": _sha256_text(source.read_text()),
                        }
                        for relative, source in extra_user_prompt_sources
                    ],
                }
            )
        atomic_write_json(registry.manifest_path(run_id), manifest.data)
        registry.manifest_path(run_id).chmod(0o444)
        if task_view == "multiturn":
            atomic_write_json(registry.task_path(run_id), agent_task_payload(manifest, task_view))
            registry.task_path(run_id).chmod(0o444)
        registry.prompt_path(run_id).write_text(prompt)
        agent_home = _agent_home(registry, run_id, agent)
        agent_home.mkdir(parents=True, exist_ok=True)
        if agent == "agy":
            settings = agent_home / ".gemini" / "antigravity-cli" / "settings.json"
            settings.parent.mkdir(parents=True, exist_ok=True)
            settings_payload: dict[str, Any] = {"modelProvider": "gemini"}
            if disable_web:
                settings_payload.update(
                    {
                        "enableTerminalSandbox": False,
                        "internetPolicy": "deny",
                        "permissions": {
                            "allow": [
                                "read_file(/workspace)",
                                "read_file(/opt)",
                                "write_file(/workspace)",
                                "mcp(ptxbench-eval/evaluate_kernel)",
                            ],
                            "deny": [
                                "command(*)",
                                "unsandboxed(*)",
                                "read_url(*)",
                                "execute_url(*)",
                            ],
                        },
                    }
                )
            atomic_write_json(settings, settings_payload)
            if disable_web:
                customization_root = agent_home / ".gemini" / "config"
                atomic_write_json(
                    customization_root / "mcp_config.json",
                    {
                        "mcpServers": {
                            "ptxbench-eval": {
                                "command": "/usr/local/bin/ptxbench-eval-mcp",
                                "args": [],
                                "cwd": "/workspace",
                                # Exceed the adapter's 930-second timeout so AGY
                                # receives its result instead of abandoning a
                                # gateway action that is still running.
                                "timeoutSeconds": _AGY_PTXBENCH_MCP_TIMEOUT_SECONDS,
                            }
                        }
                    },
                )
                agent_definition = (
                    customization_root
                    / "agents"
                    / _AGY_NO_WEB_AGENT
                    / "agent.md"
                )
                agent_definition.parent.mkdir(parents=True, exist_ok=True)
                agent_definition.write_text(_agy_no_web_agent_definition())
        token = registry.create_run(
            run_id=run_id,
            manifest=manifest.data,
            agent_prompt=prompt,
            token_budget=int(run.get("token_budget", config["token_budget"])),
            max_turns=run.get("max_turns", config.get("max_turns")),
            harness=harness,
            model=config["model"],
            prompt_tag=prompt_tag,
            artifact_provenance=context_provenance,
            feedback_style=feedback_style,
            task_view=task_view,
        )
        token_path = registry.root / "secrets" / f"{run_id}.token"
        token_path.write_text(token + "\n")
        token_path.chmod(0o600)
        env_path = registry.root / "secrets" / f"{run_id}.env"
        env_path.write_text(f"PTXBENCH_RUN_TOKEN={token}\n")
        env_path.chmod(0o600)
    experiment_record = registry.root / "experiment.json"
    if not experiment_record.exists():
        atomic_write_json(
            experiment_record,
            {key: value for key, value in config.items() if key != "_config_dir"},
        )
    return registry


def _validate_agent_task(
    registry: Registry, run_id: str, manifest: WorkloadManifest, task_view: str
) -> None:
    if task_view != "multiturn":
        return
    path = registry.task_path(run_id)
    if not path.is_file():
        raise ValueError(f"prepared experiment is missing agent task for {run_id}")
    if json_digest(read_json(path)) != json_digest(agent_task_payload(manifest, task_view)):
        raise ValueError(f"prepared experiment agent task changed for {run_id}")


def _registry_for_launch(config: dict[str, Any]) -> Registry:
    """Reuse a prepared, config-identical experiment without re-querying it."""

    validate_language_contract(config)
    feedback_style = feedback_style_from_config(config)
    task_view = task_view_from_config(config)
    root = Path(config["experiment_root"]).expanduser().resolve()
    experiment_record = root / "experiment.json"
    if not experiment_record.exists():
        return prepare(config)

    recorded = read_json(experiment_record)
    expected = {key: value for key, value in config.items() if key != "_config_dir"}
    if recorded != expected:
        raise ValueError(
            f"prepared experiment config differs from the launch config: {root}"
        )
    registry = Registry(root)
    agent = _agent_kind(config)
    codex_use_mcp = _codex_use_mcp(config)
    for run in config["runs"]:
        run_id = run["run_id"]
        if not registry.run_path(run_id).is_file():
            raise ValueError(f"prepared experiment is missing run {run_id}: {root}")
        if not registry.manifest_path(run_id).is_file():
            raise ValueError(
                f"prepared experiment is missing manifest for {run_id}: {root}"
            )
        if not registry.prompt_path(run_id).is_file():
            raise ValueError(
                f"prepared experiment is missing prompt for {run_id}: {root}"
            )
        snapshot = WorkloadManifest.parse(read_json(registry.manifest_path(run_id)))
        run_record = registry.get_run(run_id)
        if run_record.get("task_view", "full") != task_view:
            raise ValueError(f"prepared run {run_id} has a different task_view")
        if run_record.get("feedback_style", "default") != feedback_style:
            raise ValueError(f"prepared run {run_id} has a different feedback_style")
        if snapshot.digest != run_record.get("manifest_digest"):
            raise ValueError(
                f"prepared experiment manifest digest changed for {run_id}: {root}"
            )
        _validate_agent_task(registry, run_id, snapshot, task_view)
        prompt = registry.prompt_path(run_id).read_text()
        if _sha256_text(prompt) != run_record.get("agent_prompt_sha256"):
            raise ValueError(
                f"prepared experiment agent prompt changed for {run_id}: {root}"
            )
        if agent == "codex":
            expected_instruction = (
                "`ptxbench-eval` MCP server's `evaluate_kernel` tool"
                if codex_use_mcp
                else "`ptxbench-eval /workspace/kernel.cu /opt/ptxbench/workload.json`"
            )
            if expected_instruction not in prompt:
                mode = "MCP" if codex_use_mcp else "native command-line"
                raise ValueError(
                    f"prepared experiment prompt for {run_id} does not use the "
                    f"configured Codex {mode} evaluator; use a new experiment_root"
                )
    return registry


def _reported_usage(event: dict[str, Any], usage: dict[str, Any]) -> dict[str, int]:
    result = normalized_usage(usage)
    if event.get("source") != "codex_app_server":
        # AGY reports uncached input and cache reads as disjoint counts. Expose
        # the conventional provider semantics instead: cached input is a
        # subset of input, and total is all processed input plus output.
        result["input_tokens"] += result["cache_read_tokens"]
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


def _usage_from_event(
    event: dict[str, Any], step_usage: dict[int, dict[str, int]]
) -> dict[str, int] | None:
    if event.get("event") == "step_update":
        step = event.get("step_update") or {}
        usage = step.get("usage")
        index = step.get("step_index")
        if isinstance(usage, dict) and isinstance(index, int):
            step_usage[index] = _reported_usage(event, usage)
            totals = {key: 0 for key in normalized_usage({})}
            for item in step_usage.values():
                for key in totals:
                    totals[key] += item[key]
            return totals
    if event.get("event") == "result":
        usage = (event.get("result") or {}).get("usage")
        if isinstance(usage, dict):
            return _reported_usage(event, usage)
    return None


def _agy_denied_actions(event: dict[str, Any] | None) -> list[Any]:
    if not isinstance(event, dict) or event.get("event") != "result":
        return []
    result = event.get("result")
    if not isinstance(result, dict):
        return []
    denied_actions = result.get("denied_actions")
    return list(denied_actions) if isinstance(denied_actions, list) else []


def _agy_permission_denial_notice(event: dict[str, Any]) -> str | None:
    if event.get("event") != "non_json_output":
        return None
    value = event.get("text")
    if not isinstance(value, str):
        return None
    lowered = value.lower()
    if "jetski: no output produced" in lowered and "permission" in lowered:
        return value
    return None


def _codex_goal_warning_notice(event: dict[str, Any]) -> str | None:
    if event.get("event") != "codex_goal_warning":
        return None
    message = event.get("message")
    if not isinstance(message, str) or not message.strip():
        message = "Codex goal setup failed; aborting the experiment"
    details: list[str] = []
    objective_characters = event.get("objective_characters")
    if isinstance(objective_characters, int) and not isinstance(
        objective_characters, bool
    ):
        details.append(f"objective_characters={objective_characters}")
    upstream_error = event.get("upstream_error")
    if isinstance(upstream_error, str) and upstream_error.strip():
        details.append(f"upstream_error={upstream_error.strip()}")
    return f"{message}: {'; '.join(details)}" if details else message


def _agy_completion_validation(
    registry: Registry,
    run_id: str,
    *,
    result_event: dict[str, Any] | None,
    permission_denial_notice: str | None,
) -> dict[str, Any]:
    canonical_turns = registry.canonical_turn_count(run_id)
    result = result_event.get("result") if isinstance(result_event, dict) else None
    result = result if isinstance(result, dict) else {}
    denied_actions = _agy_denied_actions(result_event)
    base = {
        "canonical_turns": canonical_turns,
        "result_status": result.get("status"),
    }
    if permission_denial_notice is not None:
        return {
            **base,
            "valid": False,
            "reason": "headless_permission_denied",
            "notice": permission_denial_notice,
        }
    if denied_actions:
        return {
            **base,
            "valid": False,
            "reason": "headless_permission_denied",
            "denied_actions": denied_actions,
        }
    if not result:
        return {**base, "valid": False, "reason": "missing_agent_result"}
    if result.get("status") != "SUCCESS":
        return {**base, "valid": False, "reason": "non_success_agent_result"}

    trajectory = read_json(registry.trajectory_path(run_id), {})
    latest_evaluation: dict[str, Any] | None = None
    for message in reversed(trajectory.get("messages", [])):
        extra = message.get("extra") or {}
        if message.get("role") == "user" and extra.get("raw_output") is not None:
            latest_evaluation = extra
            break
    if latest_evaluation is None:
        response = result.get("response")
        reason = (
            "empty_agent_response"
            if not isinstance(response, str) or not response.strip()
            else "missing_evaluator_result"
        )
        return {**base, "valid": False, "reason": reason}

    all_passed = latest_evaluation.get("all_passed") is True
    min_speedup = latest_evaluation.get("min_speedup")
    record = registry.get_run(run_id)
    target_speedup = (record.get("manifest") or {}).get("target_speedup")
    evaluation = {
        **base,
        "all_passed": all_passed,
        "min_speedup": min_speedup,
        "target_speedup": target_speedup,
    }
    if not all_passed:
        return {
            **evaluation,
            "valid": False,
            "reason": "latest_evaluator_result_incorrect",
        }
    if target_speedup is not None:
        try:
            target = float(target_speedup)
        except (TypeError, ValueError):
            return {
                **evaluation,
                "valid": False,
                "reason": "invalid_target_speedup",
            }
        if (
            isinstance(min_speedup, bool)
            or not isinstance(min_speedup, (int, float))
            or min_speedup < target
        ):
            return {
                **evaluation,
                "valid": False,
                "reason": "target_speedup_not_met",
            }
    return {**evaluation, "valid": True, "reason": "goal_met"}


def _container_name(run_id: str) -> str:
    return f"ptxbench-{run_id}".replace("_", "-")


def _agent_container_security_args(
    agent: str, *, disable_web: bool = False
) -> list[str]:
    if agent != "codex":
        return []
    return [
        "--security-opt",
        "seccomp=unconfined",
        "--security-opt",
        "apparmor=unconfined",
    ]


def _agent_container_tmpfs_args(agent: str) -> list[str]:
    if agent != "codex":
        return []
    return [
        "--tmpfs",
        "/var/ptxbench-codex-home:rw,exec,nosuid,size=64m",
    ]


_FORBIDDEN_WEB_TOOLS = {
    "open_browser_url",
    "read_url_content",
    "search_web",
}


def _forbidden_web_tool(event: dict[str, Any]) -> str | None:
    if event.get("event") == "step_update":
        step = event.get("step_update")
        if isinstance(step, dict) and step.get("state") == "ACTIVE":
            tool_name = step.get("tool_name")
            if isinstance(tool_name, str) and (
                tool_name in _FORBIDDEN_WEB_TOOLS
                or "browser" in tool_name
            ):
                return tool_name
    if event.get("event") == "codex_app_server":
        payload = event.get("payload")
        if isinstance(payload, dict) and payload.get("method") == "item/started":
            params = payload.get("params")
            item = params.get("item") if isinstance(params, dict) else None
            if isinstance(item, dict) and item.get("type") == "webSearch":
                return "webSearch"
    return None


def _start_eval_exchange(
    config: dict[str, Any], registry: Registry, run_id: str, workspace: Path
) -> tuple[threading.Event, threading.Thread]:
    exchange_dir = workspace / ".ptxbench-eval-exchange"
    token_path = registry.root / "secrets" / f"{run_id}.token"
    token = token_path.read_text().strip()
    if not token:
        raise ValueError(f"empty evaluator token for {run_id}")
    broker = EvalExchangeBroker(
        exchange_dir,
        gateway_url=config["gateway_url"],
        run_id=run_id,
        run_token=token,
    )
    stop = threading.Event()
    thread = threading.Thread(
        target=broker.serve,
        args=(stop,),
        name=f"ptxbench-eval-exchange-{run_id}",
        daemon=True,
    )
    thread.start()
    return stop, thread


def _uses_eval_exchange(config: dict[str, Any]) -> bool:
    return _disable_web(config) or (
        _agent_kind(config) == "codex" and _codex_use_mcp(config)
    )


def _evaluator_container_args(
    config: dict[str, Any], run_id: str, secret_env: Path
) -> list[str]:
    if _uses_eval_exchange(config):
        return [
            "-e",
            f"PTXBENCH_RUN_ID={run_id}",
            "-e",
            "PTXBENCH_EVAL_EXCHANGE_DIR=/workspace/.ptxbench-eval-exchange",
        ]
    return [
        "-e",
        f"PTXBENCH_GATEWAY_URL={config['gateway_url']}",
        "-e",
        f"PTXBENCH_RUN_ID={run_id}",
        "--env-file",
        str(secret_env),
    ]


def _stop_agent_containers(runs: list[dict[str, Any]]) -> None:
    names = [_container_name(run["run_id"]) for run in runs]
    if not names:
        return
    subprocess.run(
        ["docker", "stop", "-t", "10", *names],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _finish_token_watcher(
    watcher: subprocess.Popen, *, launch_error: BaseException | None
) -> None:
    if launch_error is None:
        watcher.wait()
        return
    watcher.terminate()
    try:
        watcher.wait(timeout=10)
    except subprocess.TimeoutExpired:
        watcher.kill()
        watcher.wait(timeout=10)


def _launch_one(config: dict[str, Any], registry: Registry, run: dict[str, Any]) -> int:
    run_id = run["run_id"]
    initial_record = registry.get_run(run_id)
    stop_request = initial_record.get("stop_request")
    if isinstance(stop_request, dict):
        reason = str(stop_request.get("reason") or "stop_requested")
        registry.update_run(
            run_id,
            status=(
                "infra_failed" if reason == "profile_recovery_failed" else "failed"
            ),
            failure={
                "kind": reason,
                "scope": "prelaunch",
                "observed_at": utc_now(),
                "request": stop_request,
            },
        )
        return 1
    agent = _agent_kind(config)
    disable_web = _disable_web(config)
    codex_auth = _codex_plan_auth(config) if agent == "codex" else None
    workspace = registry.root / "workspaces" / run_id
    task_view = validate_task_view(initial_record.get("task_view", "full"))
    workload = (
        registry.task_path(run_id) if task_view == "multiturn"
        else registry.manifest_path(run_id)
    )
    _validate_agent_task(
        registry, run_id, WorkloadManifest.parse(initial_record["manifest"]), task_view
    )
    agent_home = _agent_home(registry, run_id, agent)
    secret_env = registry.root / "secrets" / f"{run_id}.env"
    prompt = registry.prompt_path(run_id).read_text()
    container_name = _container_name(run_id)
    credential_env = _agent_credential_env(agent)
    agent_runtime_args = ["-e", credential_env] if credential_env else ["-i"]
    evaluator_args = _evaluator_container_args(config, run_id, secret_env)
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "host",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,exec,nosuid,size=512m",
        *_agent_container_tmpfs_args(agent),
        *_agent_container_security_args(agent, disable_web=disable_web),
        *agent_runtime_args,
        *evaluator_args,
        "-v",
        f"{workspace}:/workspace",
        "-v",
        f"{workload}:/opt/ptxbench/workload.json:ro",
        "-v",
        f"{agent_home}:/home/agent",
        "-w",
        "/workspace",
        config["agent_image"],
    ]
    command.extend(_agent_argv(config, prompt))
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE if codex_auth is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    registry.set_process(run_id, pid=process.pid, container_name=container_name)
    exchange_stop: threading.Event | None = None
    exchange_thread: threading.Thread | None = None
    if _uses_eval_exchange(config):
        exchange_stop, exchange_thread = _start_eval_exchange(
            config, registry, run_id, workspace
        )
    if codex_auth is not None:
        assert process.stdin is not None
        process.stdin.write(codex_auth)
        process.stdin.close()
    step_usage: dict[int, dict[str, int]] = {}
    policy_violation: dict[str, Any] | None = None
    agy_result_event: dict[str, Any] | None = None
    agy_permission_denial: str | None = None
    codex_goal_warning: dict[str, Any] | None = None
    codex_goal_warning_notice: str | None = None
    assert process.stdout is not None
    for line_number, line in enumerate(process.stdout, start=1):
        line = line.rstrip("\n")
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = {"event": "non_json_output", "text": line}
        append_jsonl(
            registry.harness_events_path(run_id),
            {
                "schema_version": 1,
                "timestamp": utc_now(),
                "run_id": run_id,
                "line_number": line_number,
                "payload": event,
            },
        )
        if isinstance(event, dict):
            if agent == "codex":
                goal_notice = _codex_goal_warning_notice(event)
                if goal_notice is not None:
                    codex_goal_warning = event
                    codex_goal_warning_notice = goal_notice
                    print(
                        f"WARNING [{run_id}]: {goal_notice}",
                        file=sys.stderr,
                        flush=True,
                    )
            if agent == "agy":
                if event.get("event") == "result":
                    agy_result_event = event
                permission_notice = _agy_permission_denial_notice(event)
                if permission_notice is not None:
                    agy_permission_denial = permission_notice
            forbidden_tool = (
                _forbidden_web_tool(event) if disable_web else None
            )
            if forbidden_tool is not None and policy_violation is None:
                policy_violation = {
                    "observed_at": utc_now(),
                    "tool": forbidden_tool,
                    "source_event": line_number,
                }
                append_jsonl(
                    registry.harness_events_path(run_id),
                    {
                        "schema_version": 1,
                        "timestamp": policy_violation["observed_at"],
                        "run_id": run_id,
                        "line_number": line_number,
                        "payload": {
                            "event": "disable_web_policy_violation",
                            "tool": forbidden_tool,
                        },
                    },
                )
                subprocess.run(
                    ["docker", "stop", "-t", "1", container_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            usage = _usage_from_event(event, step_usage)
            if usage is not None:
                registry.update_usage(run_id, usage, source_event=line_number)
    returncode = process.wait()
    if exchange_stop is not None:
        exchange_stop.set()
    if exchange_thread is not None:
        exchange_thread.join(timeout=1)
    current = registry.get_run(run_id)
    if codex_goal_warning is not None:
        registry.update_run(
            run_id,
            status="failed",
            codex_goal_warning=codex_goal_warning,
            process={
                **current.get("process", {}),
                "returncode": returncode,
                "ended_at": utc_now(),
            },
        )
        raise RuntimeError(
            codex_goal_warning_notice
            or f"Codex goal setup failed for {run_id}; aborting the experiment"
        )
    if policy_violation is not None:
        registry.update_run(
            run_id,
            status="failed",
            disable_web_policy_violation=policy_violation,
            process={
                **current.get("process", {}),
                "returncode": returncode,
                "ended_at": utc_now(),
            },
        )
        return returncode or 1
    if current["status"] not in TERMINAL_STATES:
        usage = normalized_usage(registry.usage_snapshot(run_id).get("usage"))
        exhausted = usage["total_tokens"] >= int(current["token_budget"])
        crossing = current.get("budget_crossing")
        if exhausted and not crossing:
            crossing = {
                "observed_at": utc_now(),
                "total_tokens": usage["total_tokens"],
                "budget": current["token_budget"],
                "overshoot": usage["total_tokens"] - int(current["token_budget"]),
                "canonical_turns": registry.canonical_turn_count(run_id),
                "process_ended_before_watcher_stop": True,
            }
        if agent == "agy" and not exhausted and returncode == 0:
            completion_validation = _agy_completion_validation(
                registry,
                run_id,
                result_event=agy_result_event,
                permission_denial_notice=agy_permission_denial,
            )
            registry.update_run(
                run_id,
                status=(
                    "completed" if completion_validation["valid"] else "failed"
                ),
                completion_validation=completion_validation,
                budget_crossing=crossing,
                process={
                    **current.get("process", {}),
                    "returncode": returncode,
                    "ended_at": utc_now(),
                },
            )
        else:
            # Preserve the existing completion behavior for Codex and for
            # non-zero or budget-exhausted AGY processes.
            registry.update_run(
                run_id,
                status=(
                    "budget_exhausted"
                    if exhausted
                    else "completed" if returncode == 0 else "failed"
                ),
                budget_crossing=crossing,
                process={
                    **current.get("process", {}),
                    "returncode": returncode,
                    "ended_at": utc_now(),
                },
            )
    else:
        registry.update_run(
            run_id,
            process={
                **current.get("process", {}),
                "returncode": returncode,
                "ended_at": utc_now(),
            },
        )
    return returncode


def launch(config: dict[str, Any]) -> int:
    registry = _registry_for_launch(config)
    runnable = [
        run
        for run in config["runs"]
        if registry.get_run(run["run_id"])["status"] not in TERMINAL_STATES
    ]
    if not runnable:
        return 0
    watcher = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ptxbench_eval.token_watcher",
            "--root",
            str(registry.root),
            "--interval",
            str(config.get("watch_interval", 0.2)),
            "--tool-start-grace",
            str(config.get("tool_start_grace", 30.0)),
            "--stream-stall-timeout",
            str(config.get("stream_stall_timeout", 600.0)),
        ]
    )
    launch_error: BaseException | None = None
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(config.get("max_parallel", len(runnable)))
        ) as pool:
            futures = [pool.submit(_launch_one, config, registry, run) for run in runnable]
            try:
                pending = set(futures)
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=0.5,
                        return_when=concurrent.futures.FIRST_EXCEPTION,
                    )
                    for future in done:
                        future.result()
                    if pending and watcher.poll() is not None:
                        raise RuntimeError(
                            "token watcher exited while agent containers were active: "
                            f"returncode={watcher.returncode}"
                        )
            except BaseException:
                _stop_agent_containers(runnable)
                raise
    except BaseException as exc:
        launch_error = exc
        _stop_agent_containers(runnable)
    finally:
        _finish_token_watcher(watcher, launch_error=launch_error)
    if launch_error is not None:
        raise launch_error
    if watcher.returncode != 0:
        raise RuntimeError(f"token watcher failed: returncode={watcher.returncode}")
    final_states = [registry.get_run(run["run_id"])["status"] for run in runnable]
    return 1 if any(
        state in {"failed", "infra_failed", "cancelled"} for state in final_states
    ) else 0


def print_status(root: Path) -> None:
    registry = Registry(root)
    for path in sorted((registry.root / "runs").glob("*.json")):
        record = read_json(path)
        usage = registry.usage_snapshot(record["run_id"])["usage"]
        print(
            f"{record['run_id']}\t{record['status']}\t"
            f"{usage['total_tokens']}/{record['token_budget']}\t"
            f"turns={registry.canonical_turn_count(record['run_id'])}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ptxbench-run")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "launch"):
        item = subparsers.add_parser(command)
        item.add_argument("config", type=Path)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "status":
        print_status(args.root)
        return 0
    config = _load_config(args.config)
    if args.command == "prepare":
        registry = prepare(config)
        print(registry.root)
        return 0
    return launch(config)


if __name__ == "__main__":
    raise SystemExit(main())
