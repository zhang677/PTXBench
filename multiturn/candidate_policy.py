"""Opt-in source policies for the fixed multiturn evaluator.

This is a conservative AST verifier, not a Python security sandbox or a proof
that arbitrary inline assembly cannot implement a timer. Unsupported dynamic
Python is reported separately from an observed banned API reference. Candidate
code is never imported, executed, or rewritten by this verifier.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any


AUTOTUNE_BANNED = "autotune-banned-v1"
TAG_SUFFIX = "-" + AUTOTUNE_BANNED
POLICIES = ("default", AUTOTUNE_BANNED)
VERIFIER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# These spellings are reserved throughout candidate source, including imports,
# assignments of API references, and dead branches. This avoids alias/data-flow
# loopholes: `bench = triton.testing.do_bench` is rejected at its origin.
_BANNED_MEMBERS = {
    "autotune": "AUTOTUNER_API",
    "Autotuner": "AUTOTUNER_API",
    "autotuner": "AUTOTUNER_API",
    "do_bench": "BENCHMARK_API",
    "do_bench_cudagraph": "BENCHMARK_API",
    "do_bench_proton": "BENCHMARK_API",
    "get_benchmarker": "BENCHMARK_API",
    "benchmarker": "BENCHMARK_API",
    "benchmark": "BENCHMARK_API",
    "Benchmark": "BENCHMARK_API",
    "perf_report": "BENCHMARK_API",
    "testing": "BENCHMARK_API",
    "Event": "TIMING_API",
    "elapsed_time": "TIMING_API",
    "time": "TIMING_API",
    "time_ns": "TIMING_API",
    "perf_counter": "TIMING_API",
    "perf_counter_ns": "TIMING_API",
    "monotonic": "TIMING_API",
    "monotonic_ns": "TIMING_API",
    "process_time": "TIMING_API",
    "thread_time": "TIMING_API",
    "clock": "TIMING_API",
    "clock64": "TIMING_API",
    "globaltimer": "TIMING_API",
}
_DYNAMIC_NAMES = {
    "__import__", "__builtins__", "eval", "exec", "compile", "getattr",
    "setattr", "delattr", "hasattr", "globals", "locals", "vars", "open",
    "breakpoint", "getattribute", "attrgetter", "methodcaller",
}
_HOST_MODULES = {"math", "functools", "typing", "collections", "itertools"}
_TRITON_MODULES = (
    "triton.language", "triton.tools.tensor_descriptor",
)


def validate_policy(policy: str, language: str) -> None:
    if policy not in POLICIES:
        raise ValueError(f"Unsupported kernel policy: {policy!r}")
    if policy != "default" and language != "triton":
        raise ValueError(f"{policy} requires --language triton")


def policy_metadata(policy: str) -> dict[str, str]:
    return {"id": policy, "verifier_sha256": VERIFIER_SHA256}


def validate_policy_tag(policy: str, prompt_tag: str | None) -> None:
    is_banned_tag = bool(prompt_tag and prompt_tag.endswith(TAG_SUFFIX))
    if (policy == AUTOTUNE_BANNED) != is_banned_tag:
        raise ValueError("Autotune-banned prompt tags and --kernel-policy autotune-banned-v1 must be selected together.")


def _module_supported(module: str) -> bool:
    return (
        module in {"torch", "triton"}
        or module.split(".")[0] in _HOST_MODULES
        or any(module == prefix or module.startswith(prefix + ".") for prefix in _TRITON_MODULES)
    )


def verify_candidate(source: str, policy: str, *, filename: str = "kernel.py") -> dict[str, Any]:
    """Return JSON evidence. Only status=passed admits a policy-enabled source."""
    validate_policy(policy, "triton")
    result: dict[str, Any] = {
        **policy_metadata(policy),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "stage": "source",
        "status": "not_applicable" if policy == "default" else "passed",
        "findings": [],
    }
    if policy == "default":
        return result

    def finding(node: ast.AST, code: str, message: str, *, uncertain: bool = False) -> None:
        result["findings"].append({
            "code": code, "file": filename, "line": getattr(node, "lineno", None),
            "column": getattr(node, "col_offset", 0) + 1, "message": message,
            "status": "unverifiable" if uncertain else "rejected",
        })

    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        result["status"] = "unverifiable"
        result["findings"] = [{
            "code": "INVALID_PYTHON", "file": filename, "line": exc.lineno,
            "column": exc.offset, "message": str(exc), "status": "unverifiable",
        }]
        return result

    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if isinstance(node, ast.ImportFrom) and node.level:
                finding(node, "UNSUPPORTED_IMPORT", "Relative imports are unsupported; submit a self-contained kernel.", uncertain=True)
            for module in modules:
                if module.split(".")[0] in {"time", "timeit"}:
                    finding(node, "TIMING_API", f"Candidate-side timer module {module!r} is prohibited.")
                elif any(part in _BANNED_MEMBERS for part in module.split(".")):
                    finding(node, "BANNED_IMPORT", f"Autotuning/benchmarking module {module!r} is prohibited.")
                elif not _module_supported(module):
                    finding(node, "UNSUPPORTED_IMPORT", f"Module {module!r} is outside the statically supported host imports.", uncertain=True)
            for alias in node.names:
                if isinstance(node, ast.ImportFrom) and alias.name in _BANNED_MEMBERS:
                    finding(node, _BANNED_MEMBERS[alias.name], f"Import of {alias.name!r} is prohibited, including aliases.")
                if alias.name == "*" or alias.name.startswith("_"):
                    finding(node, "DYNAMIC_ACCESS", "Wildcard/private imports are unsupported.", uncertain=True)
                if alias.name in _DYNAMIC_NAMES:
                    finding(node, "DYNAMIC_ACCESS", f"Import of reflective API {alias.name!r} is unsupported.", uncertain=True)
                if isinstance(node, ast.ImportFrom) and alias.name == "inline_asm_elementwise":
                    finding(node, "DYNAMIC_ASSEMBLY", "Use direct tl.inline_asm_elementwise calls with literal assembly.", uncertain=True)
        if isinstance(node, ast.Attribute):
            if node.attr in _BANNED_MEMBERS or node.attr.startswith("do_bench"):
                finding(node, _BANNED_MEMBERS.get(node.attr, "BENCHMARK_API"), f"Candidate reference to .{node.attr} is prohibited.")
            elif node.attr in _DYNAMIC_NAMES or node.attr.startswith("_"):
                finding(node, "DYNAMIC_ACCESS", f"Private/reflective attribute .{node.attr} is unsupported.", uncertain=True)
            if not isinstance(node.ctx, ast.Load):
                finding(node, "ATTRIBUTE_MUTATION", "Attribute mutation is unsupported; use local state and kernel launches.", uncertain=True)
            if node.attr == "inline_asm_elementwise":
                parent = parents.get(node)
                if not isinstance(parent, ast.Call) or parent.func is not node:
                    finding(node, "DYNAMIC_ASSEMBLY", "Passing or aliasing inline assembly emitters is unsupported; call tl.inline_asm_elementwise directly.", uncertain=True)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in _DYNAMIC_NAMES:
            finding(node, "DYNAMIC_ACCESS", f"Dynamic API {node.id!r} is unsupported, including assigned aliases.", uncertain=True)
        # Device timers are another obvious candidate-side timing route. This
        # detects literal PTX only; arbitrary generated assembly is not proven safe.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "inline_asm_elementwise":
            asm = node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg == "asm"), None)
            if isinstance(asm, ast.Constant) and isinstance(asm.value, str):
                if "%globaltimer" in asm.value or "%clock" in asm.value:
                    finding(node, "DEVICE_TIMING", "Device clock reads for timing/search are prohibited.")
            else:
                finding(node, "DYNAMIC_ASSEMBLY", "Use literal inline assembly so device timer references can be checked.", uncertain=True)

    result["findings"].sort(key=lambda item: (item["line"] or 0, item["column"], item["code"]))
    if any(item["status"] == "rejected" for item in result["findings"]):
        result["status"] = "rejected"
    elif result["findings"]:
        result["status"] = "unverifiable"
    return result


def rejection_feedback(result: dict[str, Any]) -> str:
    details = "\n".join(
        f"{item['file']}:{item['line']}:{item['column']}: {item['code']}: {item['message']}"
        for item in result["findings"][:8]
    )
    return (
        f"KERNEL_POLICY_REJECTED ({result['status']})\n{details}\n"
        "Use explicit launch parameters or deterministic metadata-based heuristics. "
        "The evaluator performs benchmarking; use its feedback between turns."
    )
