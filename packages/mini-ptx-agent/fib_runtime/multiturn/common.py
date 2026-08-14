#!/usr/bin/env python3
"""Shared classes, constants, and utilities for multiturn kernel generation.

Used by run_v2.py, which selects a prompt configuration and delegates to
run_main() for the agent loop.
"""

import argparse
import ast
import fcntl
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable

import litellm
import requests
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded, Submitted
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parents[1]))  # AccRL root
from accrl.utils.code_utils import extract_code_block
from analyze_pattern import strip_comments
from collect_notes.note_feedback import append_note_feedback

logger = logging.getLogger(__name__)

# Detects cuBLAS/cuDNN usage in kernel source
_CUBLAS_PATTERN = re.compile(
    r"""
    (?:^\s*\#\s*include\s*[<"]\s*cublas(?:_v2)?\.h\s*[>"])
    |
    (?:\bcublas[A-Za-z0-9_]*\s*\()
    |
    (?:\bcublasHandle_t\b)
    """,
    re.MULTILINE | re.IGNORECASE | re.VERBOSE,
)
_CUDNN_PATTERN = re.compile(
    r"cudnn|cudnnCreate|cudnn\.h|cudnnConvolution|cudnnSetTensor", re.MULTILINE | re.IGNORECASE
)
_LIBRARY_BANNED_MSG = (
    "ERROR: Your kernel uses {library} library calls instead of a hand-written kernel. "
    "Please implement the kernel using CUDA directly — "
    "cuBLAS, cuDNN, and other library shortcuts are not allowed."
)

DEFAULT_SERVICE_URL = "http://localhost:10000"
DOCKER_IMAGE = os.environ.get("PTXBENCH_EVAL_IMAGE", "ptxbench-eval:dev")
LLM_CONTEXT_POLICIES = ("full", "latest-pair", "single-user")

SYSTEM_INSTRUCTIONS = """\
You are an expert CUDA kernel developer. Your task is to write an optimized CUDA kernel using TVM-FFI binding.

## Response Format

Each response, provide your complete kernel implementation inside a single ```cpp code block.
You will receive evaluation feedback showing compilation errors, correctness results, or performance measurements. Use this feedback to iteratively improve your kernel.

## Requirements

- Use TVM-FFI binding: include `TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, ...)`
- Implement a `__global__` CUDA kernel function
- Provide a host-side `void run(...)` function using `tvm::ffi::TensorView` parameters
- Single self-contained `.cu` file
- No external library calls (e.g. cuBLAS, cuDNN) — implement everything in CUDA directly

## Reference Material

"""

TRITON_SYSTEM_INSTRUCTIONS = """\
You are an expert Triton kernel developer. Your task is to write an optimized NVIDIA GPU kernel using Triton.

## Response Format

Each response must provide the complete `kernel.py` implementation inside a single ```python code block.
You will receive evaluation feedback showing import or compilation errors, correctness results, or performance measurements. Use this feedback to iteratively improve your kernel.

## Requirements

- Use Triton (`triton` and `triton.language`) for the GPU computation and define a destination-passing `run(...)` entry point.
- The reference function describes the required semantics only; do not copy its signature or output-allocation behavior. The public `run(...)` entry point receives every input tensor in definition order, followed by every preallocated output tensor in definition order. Write each result into the supplied output tensor; allocating or replacing outputs is not allowed.
- Use one or more real `@triton.jit` kernels and launch them from `run(...)` on the current CUDA stream.
- Do not use CuTeDSL, CUDA extensions, inline CUDA C++, cuBLAS, cuDNN, or PyTorch compute operations as a fallback.
- PyTorch may be used for tensor metadata and CUDA runtime/device operations. Do not call `torch.empty`, `torch.zeros`, or related allocation functions except inside a function installed with `triton.set_allocator` solely for device-created tensor-descriptor storage.
- Keep runtime-varying sizes and scalar values as ordinary kernel arguments unless specialization is intentional. Use `tl.constexpr` for tile shapes, feature flags, dtypes, axes, and other compile-time decisions.
- Provide a single self-contained Python file. The available kernel dependencies are `torch` and `triton`.

## Reference Material

"""

OBSERVATION_TEMPLATE = """\
{% if output.exception_info -%}
<exception>{{output.exception_info}}</exception>
{% endif -%}
<returncode>{{output.returncode}}</returncode>
<output>
{{ output.output -}}
</output>
"""


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def assemble_prompt(user_template_path: str, definition_name: str, service_url: str) -> str:
    """Fetch definition from the profiling service and fill the user template."""
    try:
        resp = requests.get(f"{service_url}/definitions/{definition_name}", timeout=(3, 10))
        resp.raise_for_status()
    except requests.exceptions.Timeout as e:
        print(f"INFRA_TIMEOUT: {service_url}/definitions/{definition_name} timed out: {e}", file=sys.stderr)
        sys.exit(2)
    definition = resp.json()
    template = Path(user_template_path).read_text()
    # Remove the "tags" field from definition
    definition.pop("tags", None)
    return template.replace("{task_content}", json.dumps(definition, indent=2))


GPU_ARCH_NVCC = {
    "hopper": "arch=compute_90a,code=sm_90a",
    "blackwell": "arch=compute_100a,code=sm_100a",
}

GPU_ARCH_TRITON = {
    "hopper": "hopper",
    "blackwell": "blackwell",
}

PYTHON_KERNEL_LANGUAGES = frozenset({"triton"})

_TRITON_TORCH_COMPUTE_METHODS = frozenset({
    "addbmm",
    "addmm",
    "baddbmm",
    "bmm",
    "conv1d",
    "conv2d",
    "conv3d",
    "einsum",
    "linear",
    "matmul",
    "mm",
    "scaled_dot_product_attention",
    "softmax",
})


def _ast_qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _ast_qualified_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
    return None


def validate_triton_source(source: str) -> str | None:
    """Return a rejection message for an unsupported Triton submission."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"Triton source is not valid Python: {exc}"

    aliases: dict[str, str] = {}
    has_triton_import = False
    has_triton_jit = False
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                module = imported.name
                if module == "cutlass" or module.startswith("cutlass."):
                    return "CuTeDSL/CUTLASS imports are not allowed in Triton kernels"
                if module == "torch" and imported.asname not in (None, "torch"):
                    return "Import torch without an alias so torch usage can be validated"
                bound_name = imported.asname or module.split(".", 1)[0]
                aliases[bound_name] = module if imported.asname else bound_name
                if module == "triton" or module.startswith("triton."):
                    has_triton_import = True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "cutlass" or module.startswith("cutlass."):
                return "CuTeDSL/CUTLASS imports are not allowed in Triton kernels"
            if module == "torch" or module.startswith("torch."):
                return "`from torch ... import ...` is not allowed in Triton kernels"
            for imported in node.names:
                bound_name = imported.asname or imported.name
                aliases[bound_name] = f"{module}.{imported.name}" if module else imported.name
                if module == "triton" or module.startswith("triton."):
                    has_triton_import = True

    def resolve_name(name: str | None) -> str | None:
        if not name:
            return None
        head, dot, tail = name.partition(".")
        resolved_head = aliases.get(head, head)
        return resolved_head + (dot + tail if dot else "")

    descriptor_allocator_functions: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if resolve_name(_ast_qualified_name(node.func)) != "triton.set_allocator":
            continue
        allocator_node = node.args[0] if node.args else None
        if isinstance(allocator_node, ast.Name):
            descriptor_allocator_functions.add(allocator_node.id)

    run_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
    ]
    if not run_functions:
        return "Triton source must define a `run(...)` entry point"
    if not has_triton_import:
        return "Triton source must import `triton`"

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in {"torch", "triton"}:
            return f"Rebinding the `{node.id}` name is not allowed in Triton kernels"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return "The matrix multiplication operator `@` is not allowed in Triton kernels"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                decorator_target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if resolve_name(_ast_qualified_name(decorator_target)) == "triton.jit":
                    has_triton_jit = True
        if not isinstance(node, ast.Call):
            continue

        call_name = resolve_name(_ast_qualified_name(node.func))
        if call_name in {"__import__", "importlib.import_module"}:
            return "Dynamic imports are not allowed in Triton kernels"
        if call_name == "getattr" and node.args:
            first_arg = resolve_name(_ast_qualified_name(node.args[0]))
            if first_arg == "torch" or (first_arg and first_arg.startswith("torch.")):
                return "Dynamic torch attribute access is not allowed in Triton kernels"

        if call_name == "torch" or (call_name and call_name.startswith("torch.")):
            allowed_torch_call = call_name.startswith("torch.cuda.")
            if call_name == "torch.empty":
                parent = parents.get(node)
                while parent is not None and not isinstance(
                    parent, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    parent = parents.get(parent)
                allowed_torch_call = (
                    isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and parent.name in descriptor_allocator_functions
                )
            if not allowed_torch_call:
                return (
                    f"PyTorch call `{call_name}(...)` is not allowed in Triton kernels; "
                    "use Triton for GPU computation"
                )

        method_name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if method_name in _TRITON_TORCH_COMPUTE_METHODS:
            return f"Tensor/PyTorch compute method `.{method_name}(...)` is not allowed in Triton kernels"

    if not has_triton_jit:
        return "Triton source must define at least one real `@triton.jit` kernel"
    return None


# ---------------------------------------------------------------------------
# Feedback formatting
# ---------------------------------------------------------------------------

def format_traces_feedback(traces: list) -> str:
    """Convert evaluation result to human-readable feedback for the LLM."""
    lines = []
    any_passed = False
    for i, trace in enumerate(traces):
        ev = trace.get("evaluation", {})
        status = ev.get("status", "UNKNOWN")
        wl = trace.get("workload", {})

        axes = wl.get("axes", {}) if isinstance(wl, dict) else {}
        axes_str = ", ".join(f"{k}={v}" for k, v in axes.items()) if axes else f"workload {i}"

        if status == "PASSED":
            any_passed = True
            perf = ev.get("performance", {})
            speedup = perf.get("speedup_factor", 0)
            latency = perf.get("latency_ms", 0)
            ref_latency = perf.get("reference_latency_ms", 0)
            lines.append(
                f"[{axes_str}] PASSED — speedup: {speedup:.3f}x "
                f"(kernel: {latency:.4f}ms, ref: {ref_latency:.4f}ms)"
            )
        else:
            lines.append(f"[{axes_str}] {status}")
            corr = ev.get("correctness") or {}
            if corr:
                abs_err = corr.get("max_absolute_error")
                rel_err = corr.get("max_relative_error")
                # None means NaN/Infinity was lost during JSON serialization
                abs_str = "NaN/Inf" if abs_err is None else abs_err
                rel_str = "NaN/Inf" if rel_err is None else rel_err
                lines.append(
                    f"  max_abs_error={abs_str}, max_rel_error={rel_str}"
                )
        
        log = ev.get("log", "")
        if log:
            lines.append(log)

    header = "Evaluation results:\n" if any_passed else "Evaluation FAILED:\n"
    return header + "\n".join(lines)


# ---------------------------------------------------------------------------
# KernelModel — LLM wrapper that skips bash-command action parsing
# ---------------------------------------------------------------------------

class KernelModel(LitellmTextbasedModel):
    """LitellmTextbasedModel that returns no parsed actions.

    The KernelAgent handles code extraction from the response content
    rather than relying on action parsing (which expects bash commands).
    """

    def _query(self, messages: list[dict[str, str]], **kwargs):
        model_kwargs = self.config.model_kwargs | kwargs
        output_config = model_kwargs.pop("_anthropic_output_config", None)
        thinking = model_kwargs.pop("_anthropic_thinking", None)
        api_base = model_kwargs.pop("_anthropic_api_base", None)
        if output_config is not None or api_base is not None:
            return self._query_anthropic_messages_api(
                messages=messages,
                model_kwargs=model_kwargs,
                output_config=output_config,
                thinking=thinking,
                api_base=api_base,
            )
        return super()._query(messages, **kwargs)

    def _calculate_cost(self, response) -> dict[str, float]:
        if self.config.model_name.startswith("openrouter/"):
            cost = self._openrouter_response_cost(response)
            if cost is not None:
                return {"cost": cost}
            if self.config.cost_tracking != "ignore_errors":
                raise RuntimeError(
                    f"OpenRouter response for {self.config.model_name} did not include usage.cost"
                )
            logger.warning("OpenRouter response for %s did not include usage.cost", self.config.model_name)
            return {"cost": 0.0}
        return super()._calculate_cost(response)

    @staticmethod
    def _openrouter_response_cost(response) -> float | None:
        usage = getattr(response, "usage", None)
        cost = None
        if isinstance(usage, dict):
            cost = usage.get("cost")
        elif usage is not None:
            cost = getattr(usage, "cost", None)

        if cost is None:
            hidden_params = getattr(response, "_hidden_params", {}) or {}
            additional_headers = hidden_params.get("additional_headers", {})
            cost = additional_headers.get("llm_provider-x-litellm-response-cost")

        if cost is None:
            return None
        return float(cost)

    def _query_anthropic_messages_api(
        self,
        *,
        messages: list[dict[str, str]],
        model_kwargs: dict,
        output_config: dict | None,
        thinking: dict | None,
        api_base: str | None,
    ):
        api_key = model_kwargs.pop("api_key", "")
        model_kwargs.pop("drop_params", None)

        system_parts = []
        anthropic_messages = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                anthropic_messages.append({
                    "role": role if role in {"user", "assistant"} else "user",
                    "content": content,
                })

        payload = {
            "model": self.config.model_name.removeprefix("anthropic/"),
            "max_tokens": model_kwargs.pop("max_tokens", None)
            or MODEL_REGISTRY[self.config.model_name]["max_tokens"],
            "messages": anthropic_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if output_config is not None:
            payload["output_config"] = output_config
        if thinking is not None:
            payload["thinking"] = thinking

        allowed_passthrough = {"stop_sequences", "top_p", "top_k", "metadata"}
        payload.update({k: v for k, v in model_kwargs.items() if k in allowed_passthrough})

        request_timeout = model_kwargs.pop("timeout", None)
        if request_timeout is None:
            request_timeout = (30, 1800)
        elif isinstance(request_timeout, (int, float)):
            request_timeout = (30, float(request_timeout))

        response = requests.post(
            f"{(api_base or 'https://api.anthropic.com').rstrip('/')}/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
            timeout=request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        reasoning_content = "\n".join(
            block.get("thinking", "")
            for block in data.get("content", [])
            if block.get("type") == "thinking" and block.get("thinking")
        )
        return _AnthropicMessagesResponse(data, text, reasoning_content)

    def _parse_actions(self, response) -> list[dict]:
        return []


class _AnthropicMessagesMessage:
    def __init__(self, content: str, reasoning_content: str = ""):
        self.content = content
        self.reasoning_content = reasoning_content

    def model_dump(self):
        message = {"role": "assistant", "content": self.content}
        if self.reasoning_content:
            message["reasoning_content"] = self.reasoning_content
        return message


class _AnthropicMessagesChoice:
    def __init__(self, content: str, reasoning_content: str = ""):
        self.message = _AnthropicMessagesMessage(content, reasoning_content)


class _AnthropicMessagesResponse:
    def __init__(self, data: dict, content: str, reasoning_content: str = ""):
        self._data = data
        self.model = data.get("model")
        self.choices = [_AnthropicMessagesChoice(content, reasoning_content)]

    def model_dump(self):
        return {
            "id": self._data.get("id"),
            "model": self.model,
            "choices": [{"message": self.choices[0].message.model_dump()}],
            "usage": self._data.get("usage", {}),
            "response": self._data,
        }


# ---------------------------------------------------------------------------
# KernelDockerEnvironment — Docker env for kernel compile + profile
# ---------------------------------------------------------------------------

def _profile_stage_state(state_file: Path) -> dict:
    try:
        state = json.loads(state_file.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    return {
        "next_ticket": int(state.get("next_ticket", 0)),
        "admit_ticket": int(state.get("admit_ticket", 0)),
        "waiters": dict(state.get("waiters") or {}),
        "holders": dict(state.get("holders") or {}),
    }


def _save_profile_stage_state(state_file: Path, state: dict) -> None:
    tmp = state_file.with_name(f"{state_file.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, sort_keys=True))
    tmp.replace(state_file)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _reap_profile_stage_state(state: dict) -> None:
    for key in list(state["waiters"]):
        pid = int((state["waiters"].get(key) or {}).get("pid", -1))
        if pid <= 0 or not _pid_alive(pid):
            state["waiters"].pop(key, None)
    for key in list(state["holders"]):
        pid = int((state["holders"].get(key) or {}).get("pid", -1))
        if pid <= 0 or not _pid_alive(pid):
            state["holders"].pop(key, None)

    # If the next ticket holder died before admission, advance to the next
    # registered waiter. This keeps killed children from wedging the queue.
    while state["admit_ticket"] < state["next_ticket"]:
        key = str(state["admit_ticket"])
        if key in state["waiters"]:
            break
        state["admit_ticket"] += 1


@contextmanager
def profile_stage_slot():
    """Limit concurrent compile/profile test.py executions across run_v2 processes.

    run_parallel_v2.py opts in by setting ACCRL_PROFILE_STAGE_TICKET_LOCK=1,
    ACCRL_PROFILE_SLOT_DIR, and ACCRL_MAX_PROFILES. Each child process takes a
    FIFO ticket immediately before running test.py; at most ACCRL_MAX_PROFILES
    tickets are admitted at once. If unset, this is a no-op for standalone
    run_v2.py usage and for legacy launchers.
    """
    if os.environ.get("ACCRL_PROFILE_STAGE_TICKET_LOCK") != "1":
        yield
        return
    slot_dir = os.environ.get("ACCRL_PROFILE_SLOT_DIR")
    max_profiles_raw = os.environ.get("ACCRL_MAX_PROFILES")
    if not slot_dir or not max_profiles_raw:
        yield
        return

    try:
        max_profiles = int(max_profiles_raw)
    except ValueError:
        raise ValueError(f"ACCRL_MAX_PROFILES must be an integer, got {max_profiles_raw!r}")
    if max_profiles <= 0:
        yield
        return

    path = Path(slot_dir)
    path.mkdir(parents=True, exist_ok=True)
    state_file = path / "ticket_state.json"
    lock_file = path / "ticket_state.lock"
    pid = os.getpid()
    ticket = None
    admitted = False

    with open(lock_file, "a+") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        state = _profile_stage_state(state_file)
        _reap_profile_stage_state(state)
        ticket = state["next_ticket"]
        state["next_ticket"] += 1
        state["waiters"][str(ticket)] = {"pid": pid, "created_at": time.time()}
        _save_profile_stage_state(state_file, state)
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    try:
        while not admitted:
            with open(lock_file, "a+") as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                state = _profile_stage_state(state_file)
                _reap_profile_stage_state(state)
                key = str(ticket)
                if (
                    state["admit_ticket"] == ticket
                    and len(state["holders"]) < max_profiles
                    and key in state["waiters"]
                ):
                    state["waiters"].pop(key, None)
                    state["holders"][key] = {"pid": pid, "admitted_at": time.time()}
                    state["admit_ticket"] += 1
                    admitted = True
                _save_profile_stage_state(state_file, state)
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            if not admitted:
                time.sleep(5)

        yield
    finally:
        if ticket is not None:
            with open(lock_file, "a+") as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                state = _profile_stage_state(state_file)
                key = str(ticket)
                if admitted:
                    state["holders"].pop(key, None)
                else:
                    state["waiters"].pop(key, None)
                _reap_profile_stage_state(state)
                _save_profile_stage_state(state_file, state)
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


class KernelDockerEnvironment(DockerEnvironment):
    """Docker environment that evaluates kernel code via compile + test.py.

    Mirrors CheckedDockerEnvironmentCUDA's success tracking and traces reading,
    but orchestrates a fixed write→compile→test pipeline per turn instead of
    executing arbitrary bash commands.
    """

    def __init__(
        self,
        *,
        success_dir: str | None = None,
        target_speedup: float = 1.0,
        definition: str | None = None,
        language: str = "cuda",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._success_dir = Path(success_dir) if success_dir else None
        self._success_version: int = 0
        self._target_speedup: float = target_speedup
        self._definition = definition
        self._language = language
        self._last_traces: list[dict] | None = None
        self._best_speedup: float = 0.0
        # Tracks assistant turns in the trajectory. It must advance even when a
        # response has no extractable kernel, otherwise success/record.json turn
        # values drift from trajectory assistant-turn indices.
        self._turn: int = -1
        # Write container ID for external cleanup (e.g., parallel launcher timeout)
        ws = self._get_host_workspace()
        if ws and self.container_id:
            try:
                (ws / ".container_id").write_text(self.container_id)
            except OSError:
                pass

    def _check_finished(self, output: dict):
        """No-op: we handle submission logic in evaluate_kernel, not via
        COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT sentinel."""
        pass

    def mark_skipped_turn(self) -> None:
        """Advance the trajectory turn counter for a response without code."""
        self._turn += 1

    def evaluate_kernel(self, kernel_source: str) -> dict:
        """Write the language-specific candidate, run test.py, and return feedback.

        Returns dict compatible with observation_template: {output, returncode, exception_info}.
        Also sets result["extra"] with traces and metadata for the agent.
        """

        self._turn += 1

        # --- Step 0: Reject language-specific fallbacks ---
        if self._language == "triton":
            validation_error = validate_triton_source(kernel_source)
            if validation_error:
                return {
                    "output": f"ERROR: {validation_error}",
                    "returncode": 1,
                    "exception_info": "",
                    "extra": {"event": "triton_validation_failed"},
                }
        else:
            kernel_source_without_comments = strip_comments(kernel_source)
            if _CUBLAS_PATTERN.search(kernel_source_without_comments):
                return {
                    "output": _LIBRARY_BANNED_MSG.format(library="cuBLAS"),
                    "returncode": 1,
                    "exception_info": "",
                    "extra": {"event": "cublas_banned"},
                }
            if _CUDNN_PATTERN.search(kernel_source_without_comments):
                return {
                    "output": _LIBRARY_BANNED_MSG.format(library="cuDNN"),
                    "returncode": 1,
                    "exception_info": "",
                    "extra": {"event": "cudnn_banned"},
                }

        # --- Step 1: Write candidate via host filesystem (volume-mounted) ---
        kernel_filename = "kernel.py" if self._language in PYTHON_KERNEL_LANGUAGES else "kernel.cu"
        ws = self._get_host_workspace()
        if ws:
            (ws / kernel_filename).write_text(kernel_source)
        else:
            # Fallback: write via docker exec
            escaped = kernel_source.replace("\\", "\\\\").replace("'", "'\\''")
            write_result = super().execute(
                {"command": f"printf '%s' '{escaped}' > {kernel_filename}"}, timeout=10
            )
            if write_result["returncode"] != 0:
                return {
                    "output": f"Failed to write {kernel_filename}:\n{write_result['output']}",
                    "returncode": 1,
                    "exception_info": "",
                    "extra": {"event": "write_failed"},
                }

        # --- Step 2: Run test.py in container (compile + local check + remote profile) ---
        # Inner timeout kills hung GPU processes; outer timeout (config.timeout) bounds the turn.
        inner_timeout = max(self.config.timeout - 10, 10)
        test_cmd = f"timeout --signal=KILL {inner_timeout} python test.py"
        with profile_stage_slot():
            test_result = super().execute({"command": test_cmd})
        test_output = test_result.get("output", "")
        test_rc = test_result.get("returncode", -1)

        # --- Step 3: Read traces.json from host workspace ---
        self._last_traces = self._read_traces_file()

        # --- Step 4: Build feedback ---
        if test_rc != 0 and not self._last_traces:
            # test.py failed before producing traces (compile error, local check fail, etc.)
            feedback = f"test.py failed (returncode {test_rc}):\n{test_output}"
            feedback = append_note_feedback(
                feedback,
                kernel_source,
                definition=self._definition,
            )
            return {
                "output": feedback,
                "returncode": 1,
                "exception_info": test_result.get("exception_info", ""),
                "extra": {"event": "test_failed", "traces": None},
            }

        if self._last_traces:
            feedback = format_traces_feedback(self._last_traces)
        else:
            feedback = f"test.py exited with code {test_rc} but no traces.json found.\n{test_output}"
            feedback = append_note_feedback(
                feedback,
                kernel_source,
                definition=self._definition,
            )
            return {
                "output": feedback,
                "returncode": 1,
                "exception_info": "",
                "extra": {"event": "no_traces", "traces": None},
            }

        # --- Step 5: Check pass/fail and save success ---
        all_passed = all(
            t.get("evaluation", {}).get("status") == "PASSED"
            for t in self._last_traces
        )
        min_speedup = 0.0
        target_met = False

        if all_passed:
            speedups = [
                t["evaluation"]["performance"]["speedup_factor"]
                for t in self._last_traces
                if t.get("evaluation", {}).get("performance", {}).get("speedup_factor") is not None
            ]
            min_speedup = min(speedups) if speedups else 0.0
            if min_speedup > self._best_speedup:
                self._best_speedup = min_speedup
            self._save_success(kernel_source, self._last_traces)

            if self._target_speedup > 0 and min_speedup >= self._target_speedup:
                target_met = True
                feedback += (
                    f"\n\nTarget speedup {self._target_speedup:.3f}x achieved! "
                    f"Min speedup: {min_speedup:.3f}x."
                )
            else:
                feedback += (
                    f"\n\nKernel is correct. Min speedup: {min_speedup:.3f}x"
                    + (f" (target: {self._target_speedup:.3f}x). Keep optimizing."
                       if self._target_speedup > 0 else ".")
                )
        else:
            feedback += "\n\nPlease fix the issues and provide an updated kernel."
            feedback = append_note_feedback(
                feedback,
                kernel_source,
                definition=self._definition,
            )
        
        
        return {
            "output": feedback,
            "returncode": 0 if all_passed else 1,
            "exception_info": "",
            "extra": {
                "event": "evaluation",
                "traces": self._last_traces,
                "all_passed": all_passed,
                "min_speedup": min_speedup,
                "target_met": target_met,
            },
        }

    def _get_host_workspace(self) -> Path | None:
        """Extract host workspace path from Docker volume mount args."""
        run_args = self.config.run_args
        for i, arg in enumerate(run_args):
            if arg == "-v" and i + 1 < len(run_args):
                mount = run_args[i + 1]
                if ":/workspace" in mount and not mount.endswith(":ro"):
                    return Path(mount.split(":/workspace")[0])
        return None

    def _read_traces_file(self) -> list[dict] | None:
        """Read and consume traces.json written by test.py."""
        ws = self._get_host_workspace()
        if ws is None:
            return None
        traces_path = ws / "traces.json"
        try:
            if not traces_path.exists():
                return None
            traces = json.loads(traces_path.read_text())
            traces_path.unlink()
            return traces
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read traces.json", exc_info=True)
            return None

    def _save_success(self, kernel_source: str, traces: list[dict]) -> None:
        """Save passing kernel version and metadata to disk."""
        if self._success_dir is None:
            return
        try:
            self._success_dir.mkdir(parents=True, exist_ok=True)
            v = self._success_version
            suffix = ".py" if self._language in PYTHON_KERNEL_LANGUAGES else ".cu"
            (self._success_dir / f"kernel_v{v}{suffix}").write_text(kernel_source)

            record_path = self._success_dir / "record.json"
            records = json.loads(record_path.read_text()) if record_path.exists() else []
            records.append({
                "version": v,
                "turn": self._turn,
                "timestamp": datetime.now().isoformat(),
                "traces": traces,
            })
            record_path.write_text(json.dumps(records, indent=2))
            self._success_version += 1
            logger.info(f"Saved passing kernel v{v} to {self._success_dir}")
        except Exception:
            logger.warning("Failed to save success kernel", exc_info=True)


# ---------------------------------------------------------------------------
# KernelAgent — DefaultAgent subclass for kernel code output
# ---------------------------------------------------------------------------

class KernelAgent(DefaultAgent):
    """Agent where the LLM outputs a complete language-specific kernel each turn.

    Overrides step() to extract kernel code from the LLM response and
    evaluate it via KernelDockerEnvironment, rather than parsing bash commands.
    """

    def __init__(
        self,
        *args,
        llm_context_policy: str = "full",
        language: str = "cuda",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if llm_context_policy not in LLM_CONTEXT_POLICIES:
            raise ValueError(f"Unsupported LLM context policy: {llm_context_policy}")
        self.llm_context_policy = llm_context_policy
        self.language = language

    def _messages_for_llm(self) -> list[dict]:
        if self.llm_context_policy == "full":
            return self.messages
        if self.llm_context_policy == "single-user":
            return self._single_user_messages_for_llm()
        if len(self.messages) <= 2:
            return self.messages

        latest_assistant = None
        latest_feedback = None
        for i in range(len(self.messages) - 2, 1, -1):
            if self.messages[i].get("role") != "assistant":
                continue
            feedback = self.messages[i + 1]
            if feedback.get("role") != "user":
                continue
            latest_assistant = self.messages[i]
            latest_feedback = feedback
            break

        if latest_assistant is None or latest_feedback is None:
            return self.messages[:2]

        return [self.messages[0], self.messages[1], latest_assistant, latest_feedback]

    @staticmethod
    def _message_content(message: dict) -> str:
        content = message.get("content", "")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    def _single_user_messages_for_llm(self) -> list[dict]:
        system_messages = []
        chunks = []
        for message in self.messages:
            role = message.get("role", "user")
            if role == "system":
                system_messages.append(message)
                continue
            chunks.append(self._message_content(message))

        if not chunks:
            return system_messages
        return [*system_messages, {"role": "user", "content": "\n\n".join(chunks)}]

    def query(self) -> dict:
        """Query the model, optionally filtering only the API-visible context."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        message = self.model.query(self._messages_for_llm())
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def step(self) -> list[dict]:
        # Query LLM (handles cost/step limits, adds assistant message)
        message = self.query()
        content = message.get("content", "") or ""

        # Extract the language-specific complete kernel block.
        code_block_language = "python" if self.language in PYTHON_KERNEL_LANGUAGES else "cpp"
        kernel_code = extract_code_block(
            content,
            languages=[code_block_language],
            keep_separators=False,
        )

        if not kernel_code:
            if hasattr(self.env, "mark_skipped_turn"):
                self.env.mark_skipped_turn()
            if self.language in PYTHON_KERNEL_LANGUAGES:
                no_code_message = (
                    "Could not extract a ```python code block from your response. "
                    "Please provide your complete Triton kernel.py in a single ```python code block."
                )
            else:
                no_code_message = (
                    "Could not extract a ```cpp code block from your response. "
                    "Please provide your complete CUDA kernel in a single ```cpp code block."
                )
            output = {
                "output": no_code_message,
                "returncode": 1,
                "exception_info": "",
                "extra": {"event": "no_code"},
            }
        else:
            # Evaluate kernel in Docker container
            output = self.env.evaluate_kernel(kernel_code)

        # Format observation using the model's observation template
        obs = self.model.format_observation_messages(
            message, [output], self.get_template_vars()
        )
        self.add_messages(*obs)

        # If target speedup met, raise Submitted to stop the loop
        extra = output.get("extra", {})
        if extra.get("target_met"):
            raise Submitted({
                "role": "exit",
                "content": "Target speedup achieved.",
                "extra": {
                    "exit_status": "Submitted",
                    "submission": kernel_code,
                    "min_speedup": extra.get("min_speedup", 0),
                },
            })

        return obs

    def resume(self, task: str, prior_messages: list[dict], n_calls_so_far: int) -> dict:
        """Like run(), but skip message reset/seeding and use pre-populated history.

        `prior_messages` should contain [system, user, then N assistant/observation pairs].
        `n_calls_so_far` is the number of LLM queries already made (i.e. N).
        """
        self.extra_template_vars |= {"task": task}
        self.messages = list(prior_messages)
        self.n_calls = n_calls_so_far
        while True:
            try:
                self.step()
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

QWEN3_SETTINGS = {
    "max_tokens": 81920,
    "temperature": 1.0,
    "top_p": 0.95,
    "presence_penalty": 1.5,
    "extra_body": {
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
    },
}
# QWEN3_SETTINGS = {
#     "max_tokens": 81920,
#     "temperature": 0.6,
#     "top_p": 0.95,
#     "presence_penalty": 0.0,
#     "extra_body": {
#         "top_k": 20,
#         "min_p": 0.0,
#         "repetition_penalty": 1.0,
#     },
# }

MODEL_REGISTRY = {
    "openai/Qwen/Qwen3-Coder-Next": {
        "input_cost_per_token": 0.5 / 1e6,
        "output_cost_per_token": 1.2 / 1e6,
    },
    "openai/Qwen/Qwen3.5-35B-A3B": {
        "input_cost_per_token": 0.4 / 1e6,
        "output_cost_per_token": 0.8 / 1e6,
    },
    "openai/Qwen/Qwen3.5-9B": {
        "input_cost_per_token": 0.4 / 1e6,
        "output_cost_per_token": 0.8 / 1e6,
    },
    "openai/Qwen/Qwen3.6-35B-A3B": {
        "input_cost_per_token": 0.4 / 1e6,
        "output_cost_per_token": 0.8 / 1e6,
    },
    "openai/Qwen/Qwen3.6-27B": {
        "input_cost_per_token": 0.4 / 1e6,
        "output_cost_per_token": 0.8 / 1e6,
    },
    "anthropic/claude-opus-4-8": {
        "max_tokens": 128000,
        "input_cost_per_token": 5.0 / 1e6,
        "output_cost_per_token": 25.0 / 1e6,
    },
    "anthropic/thinkingmachines/Inkling": {
        "max_tokens": 65536,
        "input_cost_per_token": 1.87 / 1e6,
        "output_cost_per_token": 4.68 / 1e6,
    },
    "gemini/gemini-3-flash-preview": {
        "max_tokens": 65536,
        "input_cost_per_token": 0.5 / 1e6,
        "output_cost_per_token": 3.0 / 1e6,
    },
    "gemini/gemini-3.1-pro-preview": {
        "max_tokens": 65536,
        "input_cost_per_token": 2.0 / 1e6,
        "output_cost_per_token": 12.0 / 1e6,
    },
    "gemini/gemini-3.1-flash-lite": {
        "max_tokens": 65536,
    },
    "together_ai/Qwen/Qwen3.5-397B-A17B": {
        "input_cost_per_token": 0.6 / 1e6,
        "output_cost_per_token": 3.6 / 1e6,
    },
    "openai/Qwen/Qwen3.5-397B-A17B-FP8": {
        "input_cost_per_token": 0.6 / 1e6,
        "output_cost_per_token": 3.6 / 1e6,
    },
    "openrouter/qwen/qwen3.6-plus": {},
    "openrouter/z-ai/glm-5.1": {
        "max_tokens": 131072,
    },
    "openrouter/z-ai/glm-5.2": {
        "max_tokens": 131072,
    },
    "openrouter/moonshotai/kimi-k2.7-code": {
        "max_tokens": 131072,
    },
    "openrouter/moonshotai/kimi-k2.6": {
        "max_tokens": 131072,
    },
    "openrouter/deepseek/deepseek-v4-pro": {},
    "openai/gpt-5.4": {
        "input_cost_per_token": 2.5 / 1e6,
        "output_cost_per_token": 15.0 / 1e6,
    }
}


TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
TINKER_ANTHROPIC_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/anthropic/api"


def _local_qwen_api_base() -> str:
    return f"http://{os.environ.get('ACCRL_MODEL_HOST', 'localhost:30002')}/v1"


def _resolve_tinker_model_path() -> str:
    """Resolve a Tinker ``tinker://...`` sampler path from environment variables.

    Either ``TINKER_MODEL_PATH`` is set directly, or
    ``TINKER_CHECKPOINTS_JSONL`` points at a ``checkpoints.jsonl`` and we look
    up ``TINKER_CHECKPOINT_NAME`` (default: ``final``) within it.
    """
    direct = os.environ.get("TINKER_MODEL_PATH")
    if direct:
        return direct

    jsonl_path = os.environ.get("TINKER_CHECKPOINTS_JSONL")
    if jsonl_path:
        name = os.environ.get("TINKER_CHECKPOINT_NAME", "final")
        import json as _json
        with open(jsonl_path) as f:
            entries = [_json.loads(line) for line in f if line.strip()]
        for entry in entries:
            if entry.get("name") == name:
                return entry["sampler_path"]
        available = ", ".join(e.get("name", "?") for e in entries)
        raise RuntimeError(
            f"Tinker checkpoint {name!r} not found in {jsonl_path}. Available: {available}"
        )

    raise RuntimeError(
        "Tinker model selected but no checkpoint configured. Set TINKER_MODEL_PATH "
        "(a tinker:// URI) or TINKER_CHECKPOINTS_JSONL (path to checkpoints.jsonl, "
        "optionally with TINKER_CHECKPOINT_NAME, default 'final')."
    )


def make_model(model_name: str) -> KernelModel:
    """Create KernelModel for the given model name."""
    litellm.register_model(MODEL_REGISTRY)

    model_map = {

        "Qwen3.5-35B-A3B": ("openai/Qwen/Qwen3.5-35B-A3B", {
            "api_base": _local_qwen_api_base(), 
            "api_key": "dummy",
            **QWEN3_SETTINGS
        }),
        "Qwen3.5-9B": ("openai/Qwen/Qwen3.5-9B", {
            "api_base": _local_qwen_api_base(),
            "api_key": "dummy",
            **QWEN3_SETTINGS
        }),
        "Qwen3.6-35B-A3B": ("openai/Qwen/Qwen3.6-35B-A3B", {
            "api_base": _local_qwen_api_base(),
            "api_key": "dummy",
            **QWEN3_SETTINGS
        }),
        "Qwen3.6-27B": ("openai/Qwen/Qwen3.6-27B", {
            "api_base": _local_qwen_api_base(), 
            "api_key": "dummy",
            **QWEN3_SETTINGS
        }),
        "claude-opus-4.8-xhigh": ("anthropic/claude-opus-4-8", {
            "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "_anthropic_thinking": {"type": "adaptive"},
            "_anthropic_output_config": {"effort": "xhigh"},
        }),
        "gemini-3-flash-preview": ("gemini/gemini-3-flash-preview", {
            "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""),
            "max_tokens": MODEL_REGISTRY["gemini/gemini-3-flash-preview"]["max_tokens"],
        }),
        "gemini-3.1-pro-preview": ("gemini/gemini-3.1-pro-preview", {
            "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""),
            "max_tokens": MODEL_REGISTRY["gemini/gemini-3.1-pro-preview"]["max_tokens"],
        }),
        "gemini-3.1-flash-lite": ("gemini/gemini-3.1-flash-lite", {
            "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""),
            "max_tokens": MODEL_REGISTRY["gemini/gemini-3.1-flash-lite"]["max_tokens"],
        }),
        # Gemini 3.1 Pro cannot fully disable thinking (API rejects thinkingBudget=0 with
        # "This model only works in thinking mode"). The minimum is thinkingLevel="low" +
        # includeThoughts=False, which LiteLLM exposes via reasoning_effort="none".
        "gemini-3.1-pro-no-reasoning": ("gemini/gemini-3.1-pro-preview", {
            "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""),
            "max_tokens": MODEL_REGISTRY["gemini/gemini-3.1-pro-preview"]["max_tokens"],
            "reasoning_effort": "none",
        }),
        "Qwen3.5-397B-A17B": ("together_ai/Qwen/Qwen3.5-397B-A17B", {"api_key": os.environ.get("TOGETHER_API_KEY", "")}),
        "Qwen3.5-397B-A17B-FP8": ("openai/Qwen/Qwen3.5-397B-A17B-FP8", {"api_base": _local_qwen_api_base(), "api_key": "dummy"}),
        "Qwen3.6-plus": ("openrouter/qwen/qwen3.6-plus", {"api_key": os.environ.get("OPENROUTER_API_KEY", "")}),
        "GLM-5.1": ("openrouter/z-ai/glm-5.1", {
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "max_tokens": MODEL_REGISTRY["openrouter/z-ai/glm-5.1"]["max_tokens"],
            "extra_body": {"provider": {"only": ["z-ai/fp8", "fireworks"]}},
        }),
        "GLM-5.2": ("openrouter/z-ai/glm-5.2", {
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "max_tokens": MODEL_REGISTRY["openrouter/z-ai/glm-5.2"]["max_tokens"],
            "extra_body": {"provider": {"only": ["z-ai/fp8", "fireworks"]}},
        }),
        "Kimi-K2.6": ("openrouter/moonshotai/kimi-k2.6", {
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "max_tokens": MODEL_REGISTRY["openrouter/moonshotai/kimi-k2.6"]["max_tokens"],
            "temperature": 1.0,
            "top_p": 1.0,
            "extra_body": {"provider": {"only": ["moonshotai/int4", "moonshotai/highspeed"]}},
        }),
        "Kimi-K2.7-Code": ("openrouter/moonshotai/kimi-k2.7-code", {
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "max_tokens": MODEL_REGISTRY["openrouter/moonshotai/kimi-k2.7-code"]["max_tokens"],
            "temperature": 1.0,
            "top_p": 0.95,
            "extra_body": {"provider": {"only": ["moonshotai/int4", "moonshotai/highspeed"]}},
        }),
        "DeepSeek-V4:pro": ("openrouter/deepseek/deepseek-v4-pro", {
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "temperature": 1.0,
            "top_p": 1.0,
            "extra_body": {"provider": {"only": ["deepseek"]}},
        }),
        "GPT-5.4": ("openai/gpt-5.4", {"api_key": os.environ.get("OPENAI_API_KEY", "")})
    }

    if model_name == "inkling":
        api_key = os.environ.get("TINKER_API_KEY", "")
        if not api_key:
            raise RuntimeError("Inkling selected but TINKER_API_KEY is not set")
        model_map[model_name] = (
            "anthropic/thinkingmachines/Inkling",
            {
                "api_key": api_key,
                "_anthropic_api_base": TINKER_ANTHROPIC_BASE_URL,
            },
        )

    # Tinker-hosted model: resolve the sampler path lazily so users selecting
    # other models don't need TINKER_* env vars set.
    if model_name == "Qwen3.5-35B-A3B-tinker":
        tinker_path = _resolve_tinker_model_path()
        model_map[model_name] = (
            f"openai/{tinker_path}",
            {
                "api_base": TINKER_OAI_BASE_URL,
                "api_key": os.environ.get("TINKER_API_KEY", ""),
            },
        )
    

    if "qwen36-35b" in model_name:
        model_map[model_name] = (f"openai/{model_name}", {
            "api_base": _local_qwen_api_base(),
            "api_key": "dummy",
            **QWEN3_SETTINGS
        })
    elif "qwen35-35b" in model_name:
        model_map[model_name] = (f"openai/{model_name}", {
            "api_base": _local_qwen_api_base(), 
            "api_key": "dummy",
            **QWEN3_SETTINGS
        })
    elif "qwen36-27b" in model_name:
        model_map[model_name] = (f"openai/{model_name}", {
            "api_base": _local_qwen_api_base(),
            "api_key": "dummy",
            **QWEN3_SETTINGS
        })

    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}. Supported: {list(model_map.keys()) + ['Qwen3.5-35B-A3B-tinker']}")

    litellm_name, extra_kwargs = model_map[model_name]
    model_kwargs = {"drop_params": True, **extra_kwargs}
    llm_api_timeout = os.environ.get("LLM_API_TIMEOUT")
    if llm_api_timeout:
        try:
            model_kwargs["timeout"] = float(llm_api_timeout)
        except ValueError as exc:
            raise ValueError(f"LLM_API_TIMEOUT must be a number of seconds, got {llm_api_timeout!r}") from exc

    return KernelModel(
        model_name=litellm_name,
        model_kwargs=model_kwargs,
        observation_template=OBSERVATION_TEMPLATE,
        cost_tracking="ignore_errors",
    )


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Multiturn kernel generation with profiling feedback (Docker-based)",
    )
    parser.add_argument("--definition", required=True, help="Definition name (e.g. gemm_n6144_k4096)")
    parser.add_argument("--model", required=True,
                        help="Model name (e.g. inkling, claude-opus-4.8-xhigh, Qwen3.5-35B-A3B, Qwen3.5-35B-A3B-tinker). "
                             "For inkling, set TINKER_API_KEY. For Qwen3.5-35B-A3B-tinker, set TINKER_API_KEY plus either "
                             "TINKER_MODEL_PATH (a tinker:// URI) or TINKER_CHECKPOINTS_JSONL "
                             "(+ optional TINKER_CHECKPOINT_NAME, default 'final').")
    parser.add_argument(
        "--language",
        choices=("cuda", "triton"),
        default="cuda",
        help="Kernel language (default: cuda). Triton is an opt-in Python kernel path.",
    )
    parser.add_argument("--test-path", required=True, help="Path to the test.py for this definition")
    parser.add_argument("--log-path", required=True, help="Trajectory output path (.json)")
    parser.add_argument("--output-dir", default=None, help="Working directory (default: auto tmpdir)")
    parser.add_argument("--image", default=DOCKER_IMAGE, help=f"Docker image (default: {DOCKER_IMAGE})")
    parser.add_argument("--gpus", default="all", help="GPUs to expose (default: all)")
    parser.add_argument("--service-url", default=DEFAULT_SERVICE_URL, help="Profiling service URL")
    parser.add_argument("--user-template", default=None, help="User template path (default: user_template.txt)")
    parser.add_argument("--max-turns", type=int, default=5, help="Max turns (default: 5)")
    parser.add_argument("--target-speedup", type=float, default=1.0, help="Target speedup (default: 1.0)")
    parser.add_argument("--success-dir", default=None, help="Directory to save passing kernels")
    parser.add_argument("--turn-timeout", type=int, default=130, help="Per-turn timeout in seconds (default: 130)")
    parser.add_argument("--container-timeout", type=str, default="8h",
                        help="Docker container lifetime (sleep duration, e.g. '8h', '14400s'; default: 8h). "
                             "Must exceed the total wall time of the run or later turns will see "
                             "'No such container' after the sleep expires and --rm removes the container.")
    parser.add_argument("--gpu-arch", choices=list(GPU_ARCH_NVCC.keys()), default="hopper",
                        help="GPU architecture (default: hopper)")
    parser.add_argument("--without-local-gpu", action="store_true", help="Whether to run test.py with GPU access (default: False)")
    parser.add_argument("--prompt-tag", default=None,
                        help="Prompt tag used by run_v2.py to assemble the system prompt from hub.json")
    parser.add_argument("--resume-trajectory", default=None,
                        help="Path to an existing trajectory.json; when set with --resume-turn, "
                             "seed the agent's message history from it instead of starting fresh.")
    parser.add_argument("--resume-turn", type=int, default=None,
                        help="Turn index to resume from (the first turn the new run will execute).")
    parser.add_argument("--llm-context-policy", choices=LLM_CONTEXT_POLICIES, default="full",
                        help=(
                            "Which trajectory messages are visible to the LLM API. "
                            "'full' preserves existing behavior; 'latest-pair' keeps full "
                            "trajectory logging but sends only the original task and latest "
                            "assistant/feedback pair on later turns; 'single-user' keeps system "
                            "messages as system and concatenates the rest into one user message "
                            "without role labels."
                        ))
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser.parse_args()


def run_main(build_system_prompt_fn: Callable[[str], str]) -> None:
    """Shared main() logic, parameterized by system prompt builder.

    Args:
        build_system_prompt_fn: callable(gpu_arch: str) -> str
    """
    args = parse_args()
    _run_main_impl(args, build_system_prompt_fn(args.gpu_arch))


def run_main_v2(
    build_system_prompt_fn: Callable[[str, str], str],
    build_triton_system_prompt_fn: Callable[[str, str], str] | None = None,
) -> None:
    """Like run_main, but build_system_prompt_fn takes (prompt_tag, gpu_arch).

    Requires --prompt-tag on the command line.
    """
    args = parse_args()
    if not args.prompt_tag:
        raise ValueError("run_main_v2 requires --prompt-tag")
    if args.language == "triton":
        if build_triton_system_prompt_fn is None:
            raise ValueError("run_main_v2 has no Triton system prompt builder")
        system_prompt = build_triton_system_prompt_fn(args.prompt_tag, args.gpu_arch)
    else:
        system_prompt = build_system_prompt_fn(args.prompt_tag, args.gpu_arch)
    _run_main_impl(args, system_prompt)


def _run_main_impl(args, system_prompt: str) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # --- Workspace setup ---
    tmpdir = args.output_dir or tempfile.mkdtemp(prefix="kernel_eval_")
    os.makedirs(tmpdir, exist_ok=True)
    tmpdir = str(Path(tmpdir).resolve())
    print(f"Working directory: {tmpdir}")

    # Copy test.py into workspace
    test_src = Path(args.test_path).resolve()
    shutil.copy2(test_src, os.path.join(tmpdir, "test.py"))

    # --- Prompts ---
    default_user_template = {
        "cuda": "user_template.txt",
        "triton": "user_template_triton.txt",
    }[args.language]
    user_template_path = args.user_template or str(SCRIPT_DIR / default_user_template)
    task_prompt = assemble_prompt(user_template_path, args.definition, args.service_url)

    # --- Docker environment ---
    nvcc_gencode = GPU_ARCH_NVCC[args.gpu_arch]
    if not args.without_local_gpu:
        run_args=[
            "--rm",
            "--gpus", args.gpus,
            "--network=host",
            "-v", f"{tmpdir}:/workspace",
        ]
    else:
        run_args=[
            "--rm",
            "--network=host",
            "-v", f"{tmpdir}:/workspace",
        ]
    environment_variables = {
        "PAGER": "cat",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
        "PROFILE_BASE_URL": os.environ.get("PROFILE_BASE_URL", args.service_url),
        "NVCC_GENCODE": nvcc_gencode,
    }
    if args.language == "triton":
        environment_variables["TRITON_GPU_ARCH"] = GPU_ARCH_TRITON[args.gpu_arch]

    env = KernelDockerEnvironment(
        success_dir=args.success_dir,
        target_speedup=args.target_speedup,
        definition=args.definition,
        language=args.language,
        image=args.image,
        cwd="/workspace",
        timeout=args.turn_timeout,
        container_timeout=args.container_timeout,
        run_args=run_args,
        env=environment_variables,
    )

    # --- Model ---
    model = make_model(args.model)

    # --- Agent ---
    trajectory_path = Path(args.log_path)
    agent = KernelAgent(
        model,
        env,
        system_template=system_prompt,
        instance_template="{{task}}",
        step_limit=args.max_turns,
        cost_limit=0.0,
        output_path=trajectory_path,
        llm_context_policy=args.llm_context_policy,
        language=args.language,
    )

    print(f"Definition: {args.definition}")
    print(f"Model: {args.model}")
    print(f"GPU arch: {args.gpu_arch} ({nvcc_gencode})")
    if args.language != "cuda":
        print(f"Kernel language: {args.language}")
    print(f"Docker image: {args.image}")
    print(f"GPUs: {args.gpus}")
    print(f"Max turns: {args.max_turns}")
    print(f"Target speedup: {args.target_speedup}x")
    print(f"LLM context policy: {args.llm_context_policy}")

    # Resume branch: reload trajectory messages and prime env/agent state so the
    # new run picks up at `resume_turn` with prior context preserved.
    if (args.resume_trajectory is None) != (args.resume_turn is None):
        raise ValueError("--resume-trajectory and --resume-turn must be provided together")
    if args.resume_trajectory is not None:
        resume_turn = args.resume_turn
        if resume_turn < 0:
            raise ValueError(f"--resume-turn must be >= 0, got {resume_turn}")
        with open(args.resume_trajectory) as f:
            saved = json.load(f)
        full_messages = saved.get("messages", [])
        keep = 2 + 2 * resume_turn
        if len(full_messages) < keep:
            raise ValueError(
                f"Trajectory at {args.resume_trajectory} has {len(full_messages)} messages, "
                f"need at least {keep} to resume from turn {resume_turn}"
            )
        saved_messages = full_messages[:keep]
        env._turn = resume_turn - 1
        if args.success_dir:
            record_path = Path(args.success_dir) / "record.json"
            if record_path.exists():
                try:
                    env._success_version = len(json.loads(record_path.read_text()))
                except (json.JSONDecodeError, OSError):
                    logger.warning("Failed to read existing success record.json; starting versions at 0")
        print(f"Resuming from {args.resume_trajectory} at turn {resume_turn} "
              f"({len(saved_messages)} prior messages, success_version={env._success_version})")
        print("Starting agent...")
        agent.resume(task_prompt, saved_messages, n_calls_so_far=resume_turn)
    else:
        print("Starting agent...")
        agent.run(task_prompt)

    print(f"Agent finished. Workspace: {tmpdir}")
    print(f"Trajectory: {trajectory_path}")
    if args.success_dir:
        print(f"Success kernels: {args.success_dir}")
