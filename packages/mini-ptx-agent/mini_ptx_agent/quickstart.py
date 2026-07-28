"""Small, truthful first-run workflow for PTXBench."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from .paths import PTXBenchPaths

DEFINITION = "gemm_n7168_k5120"
TEST_RELATIVE_PATH = (
    "2026-0413-1611/"
    "gemm_n7168_k5120_94920358-01a8-4c5b-9209-3103fd490e94.py"
)
LOCAL_MODEL_NAMES = {
    "Qwen3.5-35B-A3B",
    "Qwen3.5-9B",
    "Qwen3.6-35B-A3B",
    "Qwen3.6-27B",
    "Qwen3.5-397B-A17B-FP8",
}
CLOUD_MODEL_KEYS = {
    "GPT-5.4": ("OPENAI_API_KEY",),
    "claude-opus-4.8-xhigh": ("ANTHROPIC_API_KEY",),
    "gemini-3-flash-preview": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gemini-3.1-pro-preview": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "gemini-3.1-pro-no-reasoning": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "Qwen3.5-397B-A17B": ("TOGETHER_API_KEY",),
    "Qwen3.6-plus": ("OPENROUTER_API_KEY",),
    "GLM-5.1": ("OPENROUTER_API_KEY",),
    "GLM-5.2": ("OPENROUTER_API_KEY",),
    "Kimi-K2.6": ("OPENROUTER_API_KEY",),
    "Kimi-K2.7-Code": ("OPENROUTER_API_KEY",),
    "DeepSeek-V4:pro": ("OPENROUTER_API_KEY",),
}


@dataclass(frozen=True)
class Check:
    label: str
    ok: bool
    detail: str


def _is_local_model(model: str) -> bool:
    lowered = model.lower()
    return model in LOCAL_MODEL_NAMES or any(
        marker in lowered
        for marker in ("qwen36-27b", "qwen36-35b", "qwen35-35b")
    )


def _get_json(url: str) -> Any:
    response = requests.get(url, timeout=(3, 10))
    response.raise_for_status()
    return response.json()


def _check_fibserve(service_url: str) -> list[Check]:
    base_url = service_url.rstrip("/")
    checks: list[Check] = []
    try:
        health = _get_json(f"{base_url}/health")
        status_ok = isinstance(health, dict) and health.get("status") == "ok"
        workers = health.get("workers", []) if isinstance(health, dict) else []
        backends = health.get("backends", []) if isinstance(health, dict) else []
        if workers:
            status_ok = status_ok and all(worker.get("healthy") for worker in workers)
        if backends:
            status_ok = status_ok and all(backend.get("healthy") for backend in backends)
        checks.append(
            Check(
                "FIBServe health",
                status_ok,
                f"{base_url}/health returned status={health.get('status')!r}"
                if isinstance(health, dict)
                else f"{base_url}/health returned a non-object",
            )
        )
    except (requests.RequestException, ValueError) as exc:
        checks.append(Check("FIBServe health", False, f"{base_url}/health: {exc}"))

    try:
        definition = _get_json(f"{base_url}/definitions/{DEFINITION}")
        found = isinstance(definition, dict) and definition.get("name") == DEFINITION
        checks.append(
            Check(
                "quickstart definition",
                found,
                f"{base_url}/definitions/{DEFINITION}",
            )
        )
    except (requests.RequestException, ValueError) as exc:
        checks.append(
            Check(
                "quickstart definition",
                False,
                f"{base_url}/definitions/{DEFINITION}: {exc}",
            )
        )
    return checks


def _check_model(model: str, model_host: str | None) -> Check:
    if not model:
        return Check(
            "model",
            False,
            "set MODEL_NAME or pass --model (for example Qwen3.6-27B or GPT-5.4)",
        )
    if _is_local_model(model):
        if not model_host:
            return Check(
                "model endpoint",
                False,
                "set ACCRL_MODEL_HOST to host:port for the OpenAI-compatible endpoint",
            )
        if "://" in model_host:
            return Check(
                "model endpoint",
                False,
                "ACCRL_MODEL_HOST must be host:port without http:// or /v1",
            )
        url = f"http://{model_host}/v1/models"
        try:
            payload = _get_json(url)
            data = payload.get("data", []) if isinstance(payload, dict) else []
            ids = [item.get("id") for item in data if isinstance(item, dict)]
            return Check(
                "model endpoint",
                model in ids,
                f"{url} model_ids={ids}",
            )
        except (requests.RequestException, ValueError) as exc:
            return Check("model endpoint", False, f"{url}: {exc}")

    keys = CLOUD_MODEL_KEYS.get(model)
    if keys is None:
        return Check(
            "model",
            False,
            f"{model!r} is not supported by the quickstart launcher",
        )
    configured = [key for key in keys if os.environ.get(key)]
    return Check(
        "model credentials",
        bool(configured),
        f"{configured[0]} is set" if configured else f"set one of: {', '.join(keys)}",
    )


def preflight(
    paths: PTXBenchPaths,
    *,
    model: str,
    model_host: str | None,
    service_url: str,
    eval_image: str,
) -> list[Check]:
    """Check every dependency needed before spending a model request."""
    runner = paths.multiturn_root / "run_parallel_v2.py"
    test_path = paths.multiturn_root / TEST_RELATIVE_PATH
    config = paths.config_root / "quickstart.json"
    checks = [
        Check("multiturn runner", runner.is_file(), str(runner)),
        Check("GEMM test", test_path.is_file(), str(test_path)),
        Check("quickstart config", config.is_file(), str(config)),
        Check("docker command", shutil.which("docker") is not None, shutil.which("docker") or "not found"),
    ]
    if shutil.which("docker"):
        image_check = subprocess.run(
            ["docker", "image", "inspect", eval_image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        checks.append(
            Check(
                "evaluator image",
                image_check.returncode == 0,
                eval_image
                if image_check.returncode == 0
                else (image_check.stderr.strip() or f"docker cannot inspect {eval_image}"),
            )
        )
    else:
        checks.append(Check("evaluator image", False, f"cannot inspect {eval_image} without docker"))
    checks.extend(_check_fibserve(service_url))
    checks.append(_check_model(model, model_host))
    return checks


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _feedback_excerpt(trajectory: Path, limit: int = 1200) -> str | None:
    data = _read_json(trajectory)
    messages = data.get("messages", []) if isinstance(data, dict) else []
    for message in reversed(messages[2:] if isinstance(messages, list) else []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip()
        output_match = re.search(r"<output>\s*(.*?)\s*</output>", content, re.DOTALL)
        if output_match:
            content = output_match.group(1).strip()
        if content:
            return content[:limit] + ("..." if len(content) > limit else "")
    return None


def _exit_status(trajectory: Path) -> str | None:
    data = _read_json(trajectory)
    messages = data.get("messages", []) if isinstance(data, dict) else []
    if not isinstance(messages, list) or not messages:
        return None
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "exit":
        return None
    extra = last.get("extra") or {}
    return str(extra.get("exit_status") or last.get("content") or "exit")


def build_result(output_root: Path) -> dict[str, Any]:
    """Describe useful artifacts without confusing process completion with correctness."""
    output_root = output_root.expanduser().resolve()
    summary = _read_json(output_root / "summary.json")
    summary = summary if isinstance(summary, dict) else {}
    summary_results = summary.get("results", [])
    result_by_name = {
        item["exp_name"]: item
        for item in summary_results
        if isinstance(item, dict) and isinstance(item.get("exp_name"), str)
    }

    plan_data = _read_json(output_root / "plan.json")
    plan = plan_data.get("plan", []) if isinstance(plan_data, dict) else []
    definition_by_name: dict[str, str] = {}
    for position, item in enumerate(plan if isinstance(plan, list) else []):
        if not isinstance(item, dict):
            continue
        index = item.get("exp_index", position)
        try:
            exp_name = f"exp_{int(index):03d}"
        except (TypeError, ValueError):
            continue
        definition_by_name[exp_name] = str(item.get("definition") or "")

    exp_names = set(result_by_name) | set(definition_by_name)
    exp_names.update(path.stem for path in (output_root / "trajectories").glob("exp_*.json"))
    exp_names.update(path.name for path in (output_root / "success").glob("exp_*") if path.is_dir())
    exp_names.update(path.name for path in output_root.glob("exp_*") if path.is_dir())

    experiments: list[dict[str, Any]] = []
    for exp_name in sorted(exp_names):
        candidate = output_root / exp_name / "kernel.cu"
        trajectory = output_root / "trajectories" / f"{exp_name}.json"
        correct_kernels = sorted((output_root / "success" / exp_name).glob("kernel_v*.cu"))
        summary_item = result_by_name.get(exp_name, {})
        experiments.append(
            {
                "exp_name": exp_name,
                "definition": summary_item.get("definition") or definition_by_name.get(exp_name),
                "launcher_status": summary_item.get("status"),
                "candidate_kernel": _relative(candidate, output_root) if candidate.is_file() else None,
                "trajectory": _relative(trajectory, output_root) if trajectory.is_file() else None,
                "correct_kernels": [_relative(path, output_root) for path in correct_kernels],
                "exit_status": _exit_status(trajectory) if trajectory.is_file() else None,
                "feedback_excerpt": _feedback_excerpt(trajectory) if trajectory.is_file() else None,
            }
        )

    generated_count = sum(bool(item["candidate_kernel"]) for item in experiments)
    correct_count = sum(len(item["correct_kernels"]) for item in experiments)
    target_count = sum(item["exit_status"] == "Submitted" for item in experiments)
    completed = int(summary.get("completed", summary.get("success", 0)) or 0)
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "output_root": str(output_root),
        "runner": {
            "planned": len(plan) if isinstance(plan, list) else 0,
            "completed_processes": completed,
            "infra_failed": int(summary.get("infra_failed", 0) or 0),
            "api_failed": int(summary.get("api_failed", 0) or 0),
            "timeout": int(summary.get("timeout", 0) or 0),
            "error": int(summary.get("error", 0) or 0),
        },
        "outcome": {
            "generated_candidate_count": generated_count,
            "correct_kernel_count": correct_count,
            "target_achieved_count": target_count,
        },
        "experiments": experiments,
    }


def write_result(output_root: Path) -> tuple[Path, dict[str, Any]]:
    result = build_result(output_root)
    result_path = output_root.expanduser().resolve() / "quickstart-result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result_path, result


def print_result(result_path: Path, result: dict[str, Any]) -> None:
    outcome = result["outcome"]
    print("\nPTXBench quickstart result")
    print(f"  output root:          {result['output_root']}")
    print(f"  generated candidates: {outcome['generated_candidate_count']}")
    print(f"  correct kernels:      {outcome['correct_kernel_count']}")
    print(f"  targets achieved:     {outcome['target_achieved_count']}")
    for experiment in result["experiments"]:
        print(f"  {experiment['exp_name']}:")
        if experiment["candidate_kernel"]:
            print(f"    latest candidate:   {experiment['candidate_kernel']}")
        if experiment["correct_kernels"]:
            print(f"    correct kernel:     {experiment['correct_kernels'][-1]}")
        if experiment["trajectory"]:
            print(f"    full trajectory:    {experiment['trajectory']}")
        if experiment["feedback_excerpt"]:
            first_line = experiment["feedback_excerpt"].splitlines()[0]
            print(f"    final feedback:     {first_line[:240]}")
    print(f"  machine-readable:     {result_path}")


def run(
    paths: PTXBenchPaths,
    *,
    model: str,
    service_url: str,
    eval_image: str,
    output_root: Path,
) -> int:
    """Run one three-turn GEMM trajectory and always leave a truthful report."""
    output_root = output_root.expanduser().resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(paths.multiturn_root / "run_parallel_v2.py"),
        "--config",
        str(paths.config_root / "quickstart.json"),
        "--definition",
        DEFINITION,
        "--test-path",
        str(paths.multiturn_root / TEST_RELATIVE_PATH),
        "--model",
        model,
        "--service-url",
        service_url,
        "--gpu-arch",
        "hopper",
        "--image",
        eval_image,
        "--without-local-gpu",
        "--max-parallel",
        "1",
        "--output-root",
        str(output_root),
        "--timeout",
        "7200",
        "--turn-timeout",
        "980",
    ]
    completed = subprocess.run(command, check=False)
    if output_root.is_dir():
        result_path, result = write_result(output_root)
        print_result(result_path, result)
    return completed.returncode
