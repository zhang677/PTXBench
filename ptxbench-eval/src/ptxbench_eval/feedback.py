from __future__ import annotations

import re
from typing import Any


FEEDBACK_STYLES = ("default", "multiturn")


def validate_feedback_style(value: Any) -> str:
    if not isinstance(value, str) or value not in FEEDBACK_STYLES:
        raise ValueError("feedback_style must be 'default' or 'multiturn'")
    return value


def feedback_style_from_config(config: dict[str, Any]) -> str:
    style = validate_feedback_style(config.get("feedback_style", "default"))
    runs = config.get("runs")
    if isinstance(runs, list) and any(
        isinstance(run, dict) and "feedback_style" in run for run in runs
    ):
        raise ValueError("feedback_style is experiment-wide; per-run overrides are not supported")
    return style


CUBLAS_PATTERN = re.compile(
    r'(?:^\s*\#\s*include\s*[<"]\s*cublas(?:_v2)?\.h\s*[>"])|'
    r'(?:\bcublas[A-Za-z0-9_]*\s*\()|(?:\bcublasHandle_t\b)',
    re.MULTILINE | re.IGNORECASE,
)
CUDNN_PATTERN = re.compile(r"cudnn", re.MULTILINE | re.IGNORECASE)


def strip_cpp_comments(source: str) -> str:
    """Strip C/C++ comments without treating comment markers in literals as comments."""
    out: list[str] = []
    i, length = 0, len(source)
    while i < length:
        char = source[i]
        if char == '"':
            end = i + 1
            while end < length:
                if source[end] == "\\" and end + 1 < length:
                    end += 2
                    continue
                if source[end] == '"':
                    end += 1
                    break
                end += 1
            out.append(source[i:end])
            i = end
            continue
        if char == "'":
            end = i + 1
            while end < length:
                if source[end] == "\\" and end + 1 < length:
                    end += 2
                    continue
                if source[end] == "'":
                    end += 1
                    break
                end += 1
            out.append(source[i:end])
            i = end
            continue
        if char == "/" and i + 1 < length and source[i + 1] == "/":
            end = i + 2
            while end < length and source[end] != "\n":
                end += 1
            i = end
            continue
        if char == "/" and i + 1 < length and source[i + 1] == "*":
            end = i + 2
            while end + 1 < length and not (
                source[end] == "*" and source[end + 1] == "/"
            ):
                if source[end] == "\n":
                    out.append("\n")
                end += 1
            i = end + 2 if end + 1 < length else length
            continue
        out.append(char)
        i += 1
    return "".join(out)


def validation_error(source: str) -> str | None:
    clean = strip_cpp_comments(source)
    library = "cuBLAS" if CUBLAS_PATTERN.search(clean) else "cuDNN" if CUDNN_PATTERN.search(clean) else None
    if not library:
        return None
    return (
        f"ERROR: Your kernel uses {library} library calls instead of a hand-written kernel. "
        "Please implement the kernel using CUDA directly — "
        "cuBLAS, cuDNN, and other library shortcuts are not allowed."
    )


def prepend_log(traces: list[dict[str, Any]] | None, text: str) -> None:
    if not traces or not text:
        return
    for trace in traces:
        evaluation = trace.setdefault("evaluation", {})
        existing = evaluation.get("log", "") or ""
        evaluation["log"] = text + ("\n" + existing if existing else "")


def strip_driver_probe_noise(log: str) -> str:
    out: list[str] = []
    skipping = False
    for line in log.split("\n"):
        if "Program hit" in line and "cuGetProcAddress_v2" in line:
            skipping = True
            continue
        if skipping:
            if line.strip() == "=========":
                skipping = False
            continue
        out.append(line)
    return "\n".join(out)


def sanitizer_failure_status(traces: Any) -> str | None:
    if not isinstance(traces, list):
        return None
    for trace in traces:
        clean = strip_driver_probe_noise(str((trace or {}).get("log", "") or ""))
        if "timed out after" in clean.lower():
            return "TIMEOUT"
        if "Program hit" in clean:
            return "RUNTIME_ERROR"
    return None


NOISE_SUBSTRINGS = (
    "========= COMPUTE-SANITIZER",
    "========= Target application returned an error",
    "ERROR SUMMARY",
    "Sanitizer checks complete",
    "WARNING:",
    "passed successfully",
)


def clean_sanitizer_log(log: str) -> str:
    kept: list[str] = []
    for line in log.split("\n"):
        stripped = line.strip()
        if "Host Frame" in line or "Saved host backtrace up to driver entry point" in line:
            continue
        if stripped and set(stripped) == {"="}:
            continue
        if stripped.startswith("Running ") and stripped.split()[-1:] == ["MEMCHECK"]:
            continue
        if line.startswith("STDOUT:") or line.startswith("Return code:"):
            continue
        if any(noise in line for noise in NOISE_SUBSTRINGS):
            continue
        if not stripped and kept and not kept[-1].strip():
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def sanitizer_traces(raw: Any, status: str) -> list[dict[str, Any]]:
    traces = raw if isinstance(raw, list) else []
    for trace in traces:
        if isinstance(trace.get("log"), str):
            trace["log"] = clean_sanitizer_log(trace["log"])
        evaluation = trace.setdefault("evaluation", {})
        evaluation["status"] = status
        if "log" not in evaluation and isinstance(trace.get("log"), str):
            evaluation["log"] = trace["log"]
    return traces


def render_debug_metadata(metadata: Any) -> str:
    if not isinstance(metadata, dict):
        return ""
    rendered: list[str] = []
    for title, section in metadata.items():
        if title == "FlashInfer CUDA debug report":
            continue
        if isinstance(section, dict) and not section.get("exist"):
            continue
        rendered.extend([str(title), "-" * 80])
        if isinstance(section, dict) and section.get("msg"):
            rendered.append(str(section["msg"]))
        rendered.append("")
    report = "\n".join(rendered).rstrip()
    return report + "\n" if report else ""


def replace_runtime_logs(traces: Any, report: str) -> None:
    if not report or not isinstance(traces, list):
        return
    for trace in traces:
        evaluation = trace.setdefault("evaluation", {})
        if evaluation.get("status") in {"RUNTIME_ERROR", "TIMEOUT"}:
            evaluation["log"] = report


def has_runtime_failure(traces: Any) -> bool:
    return isinstance(traces, list) and any(
        (trace.get("evaluation") or {}).get("status") in {"RUNTIME_ERROR", "TIMEOUT"}
        for trace in traces
        if isinstance(trace, dict)
    )


def _append_per_output_correctness(lines: list[str], correctness: dict[str, Any]) -> None:
    extra = correctness.get("extra")
    if not isinstance(extra, dict):
        return
    per_output = extra.get("per_output")
    if not isinstance(per_output, dict) or not per_output:
        return

    rendered: list[str] = []
    for output_name, metrics in per_output.items():
        if not isinstance(metrics, dict):
            continue
        passed = metrics.get("passed")
        output_status = (
            "PASSED" if passed is True else "FAILED" if passed is False else "UNKNOWN"
        )
        absolute = metrics.get("max_absolute_error")
        relative = metrics.get("max_relative_error")
        matched_ratio = metrics.get("min_matched_ratio")
        rendered.append(
            f"    {output_name}: {output_status}, "
            f"max_abs_error={'NaN/Inf' if absolute is None else absolute}, "
            f"max_rel_error={'NaN/Inf' if relative is None else relative}, "
            f"min_matched_ratio={'unknown' if matched_ratio is None else matched_ratio}"
        )
    if rendered:
        lines.append("  per-output correctness:")
        lines.extend(rendered)


def format_traces_feedback(
    traces: list[dict[str, Any]], *, feedback_style: str = "default"
) -> str:
    feedback_style = validate_feedback_style(feedback_style)
    lines: list[str] = []
    any_passed = False
    for index, trace in enumerate(traces):
        evaluation = trace.get("evaluation", {})
        status = evaluation.get("status", "UNKNOWN")
        workload = trace.get("workload", {})
        axes = workload.get("axes", {}) if isinstance(workload, dict) else {}
        axes_text = ", ".join(f"{key}={value}" for key, value in axes.items()) if axes else f"workload {index}"
        if status == "PASSED":
            any_passed = True
            performance = evaluation.get("performance", {})
            speedup = float(performance.get("speedup_factor", 0) or 0)
            latency = float(performance.get("latency_ms", 0) or 0)
            reference = float(performance.get("reference_latency_ms", 0) or 0)
            lines.append(
                f"[{axes_text}] PASSED — speedup: {speedup:.3f}x "
                f"(kernel: {latency:.4f}ms, ref: {reference:.4f}ms)"
            )
        else:
            lines.append(f"[{axes_text}] {status}")
            correctness = evaluation.get("correctness") or {}
            if correctness:
                absolute = correctness.get("max_absolute_error")
                relative = correctness.get("max_relative_error")
                lines.append(
                    "  max_abs_error="
                    f"{'NaN/Inf' if absolute is None else absolute}, "
                    f"max_rel_error={'NaN/Inf' if relative is None else relative}"
                )
                if feedback_style == "default":
                    _append_per_output_correctness(lines, correctness)
        log = evaluation.get("log", "")
        if log:
            lines.append(str(log))
    header = "Evaluation results:\n" if any_passed else "Evaluation FAILED:\n"
    return header + "\n".join(lines)


def finish_feedback(
    traces: list[dict[str, Any]], *, target_speedup: float = 0.0,
    feedback_style: str = "default",
) -> tuple[str, bool, float]:
    feedback = format_traces_feedback(traces, feedback_style=feedback_style)
    all_passed = bool(traces) and all(
        (trace.get("evaluation") or {}).get("status") == "PASSED" for trace in traces
    )
    speedups = []
    if all_passed:
        speedups = [
            float(((trace.get("evaluation") or {}).get("performance") or {}).get("speedup_factor"))
            for trace in traces
            if ((trace.get("evaluation") or {}).get("performance") or {}).get("speedup_factor")
            is not None
        ]
    min_speedup = min(speedups) if all_passed and speedups else 0.0
    if all_passed:
        if target_speedup > 0 and min_speedup >= target_speedup:
            feedback += (
                f"\n\nTarget speedup {target_speedup:.3f}x achieved! "
                f"Min speedup: {min_speedup:.3f}x."
            )
        else:
            feedback += f"\n\nKernel is correct. Min speedup: {min_speedup:.3f}x"
            feedback += (
                f" (target: {target_speedup:.3f}x). Keep optimizing."
                if target_speedup > 0
                else "."
            )
    else:
        feedback += "\n\nPlease fix the issues and provide an updated kernel."
    return feedback, all_passed, min_speedup
