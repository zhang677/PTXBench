#!/usr/bin/env python3
"""Compare two success-diffs.md reports and validate kernel file differences."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_LEFT = Path("/home/ubuntu/AccRL-exps/eval_runs/2026-0609-0100/success-diffs.md")
DEFAULT_RIGHT = Path("/home/ubuntu/AccRL-exps/eval_runs/2026-0608-1500/success-diffs.md")
SECTION_RE = re.compile(r"^## (exp_(\d+)) / ([^\n]+)$", re.MULTILINE)
BACKTICK_VALUE_RE = r"`([^`]+)`"


@dataclass(frozen=True)
class DiffStats:
    differs: bool | None
    added_lines: int | None
    removed_lines: int | None
    hunks: int | None
    reason: str | None = None


@dataclass(frozen=True)
class Entry:
    exp_name: str
    exp_index: int
    kernel_name: str
    success_kernel: Path
    original_kernel: Path
    definition: str | None
    workload_axes: str | None
    speedup: float | None
    latency_ms: float | None
    reference_latency_ms: float | None
    markdown_has_diff: bool
    success_sha256: str | None
    original_sha256: str | None
    file_diff: DiffStats


@dataclass(frozen=True)
class Report:
    path: Path
    title: str | None
    declared_success_count: int | None
    entries: list[Entry]

    @property
    def by_exp(self) -> dict[int, Entry]:
        return {entry.exp_index: entry for entry in self.entries}

    @property
    def original_paths(self) -> set[Path]:
        return {entry.original_kernel for entry in self.entries}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Investigate differences between two success-diffs.md reports. "
            "The script compares successful exp ids, original kernels, success "
            "kernel hashes, and whether each success kernel actually differs "
            "from its original .cu file."
        )
    )
    parser.add_argument(
        "left",
        nargs="?",
        type=Path,
        default=DEFAULT_LEFT,
        help=f"First success-diffs.md report. Defaults to {DEFAULT_LEFT}.",
    )
    parser.add_argument(
        "right",
        nargs="?",
        type=Path,
        default=DEFAULT_RIGHT,
        help=f"Second success-diffs.md report. Defaults to {DEFAULT_RIGHT}.",
    )
    parser.add_argument(
        "--left-label",
        default=None,
        help="Display label for the first report. Defaults to the parent directory name.",
    )
    parser.add_argument(
        "--right-label",
        default=None,
        help="Display label for the second report. Defaults to the parent directory name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write markdown output to this path instead of stdout.",
    )
    return parser.parse_args()


def extract_backtick_line(block: str, label: str) -> str | None:
    match = re.search(rf"^- {re.escape(label)}: {BACKTICK_VALUE_RE}", block, re.MULTILINE)
    return match.group(1) if match else None


def extract_speedup(block: str) -> float | None:
    value = extract_backtick_line(block, "Speedup")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def extract_latency(block: str) -> tuple[float | None, float | None]:
    match = re.search(
        rf"^- Latency: {BACKTICK_VALUE_RE} ms vs reference {BACKTICK_VALUE_RE} ms",
        block,
        re.MULTILINE,
    )
    if not match:
        return None, None
    try:
        latency = float(match.group(1))
    except ValueError:
        latency = None
    try:
        reference = float(match.group(2))
    except ValueError:
        reference = None
    return latency, reference


def markdown_diff_has_changes(block: str) -> bool:
    match = re.search(r"```diff\n(.*?)\n```", block, re.DOTALL)
    if not match:
        return False
    diff_text = match.group(1).strip()
    return bool(diff_text and diff_text != "(no changes)")


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def file_diff_stats(original: Path, success: Path) -> DiffStats:
    try:
        original_bytes = original.read_bytes()
        success_bytes = success.read_bytes()
    except OSError as exc:
        return DiffStats(None, None, None, None, str(exc))

    if original_bytes == success_bytes:
        return DiffStats(False, 0, 0, 0)

    original_lines = original_bytes.decode(errors="replace").splitlines(keepends=True)
    success_lines = success_bytes.decode(errors="replace").splitlines(keepends=True)
    added = 0
    removed = 0
    hunks = 0
    for line in difflib.unified_diff(original_lines, success_lines, n=0):
        if line.startswith("@@"):
            hunks += 1
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return DiffStats(True, added, removed, hunks)


def parse_report(path: Path) -> Report:
    text = path.read_text(errors="replace")
    title_match = re.search(r"^# Success Kernel Diff Report: `([^`]+)`$", text, re.MULTILINE)
    count_match = re.search(r"^- Success kernels: `(\d+)`$", text, re.MULTILINE)
    declared_count = int(count_match.group(1)) if count_match else None

    entries: list[Entry] = []
    matches = list(SECTION_RE.finditer(text))
    for index, match in enumerate(matches):
        block_start = match.end()
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[block_start:block_end]

        exp_name = match.group(1)
        exp_index = int(match.group(2))
        kernel_name = match.group(3).strip()
        success_text = extract_backtick_line(block, "Success kernel")
        original_text = extract_backtick_line(block, "Original kernel")
        if success_text is None or original_text is None:
            raise ValueError(f"{path}: {exp_name} is missing success or original kernel metadata")

        success_kernel = Path(success_text)
        original_kernel = Path(original_text)
        latency_ms, reference_latency_ms = extract_latency(block)
        success_sha = sha256_file(success_kernel)
        original_sha = sha256_file(original_kernel)
        diff_stats = file_diff_stats(original_kernel, success_kernel)
        entries.append(
            Entry(
                exp_name=exp_name,
                exp_index=exp_index,
                kernel_name=kernel_name,
                success_kernel=success_kernel,
                original_kernel=original_kernel,
                definition=extract_backtick_line(block, "Definition"),
                workload_axes=extract_backtick_line(block, "Workload axes"),
                speedup=extract_speedup(block),
                latency_ms=latency_ms,
                reference_latency_ms=reference_latency_ms,
                markdown_has_diff=markdown_diff_has_changes(block),
                success_sha256=success_sha,
                original_sha256=original_sha,
                file_diff=diff_stats,
            )
        )

    return Report(
        path=path,
        title=title_match.group(1) if title_match else None,
        declared_success_count=declared_count,
        entries=entries,
    )


def short_hash(value: str | None) -> str:
    return value[:12] if value else "missing"


def yes_no(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def exp_list(values: set[int]) -> str:
    return ", ".join(f"exp_{value:03d}" for value in sorted(values)) or "(none)"


def path_list(values: set[Path]) -> list[str]:
    return [str(path) for path in sorted(values, key=str)]


def diff_summary(entry: Entry) -> str:
    stats = entry.file_diff
    if stats.differs is None:
        return f"unknown ({stats.reason})"
    if not stats.differs:
        return "no"
    return f"yes (+{stats.added_lines}/-{stats.removed_lines}, {stats.hunks} hunks)"


def speedup_text(entry: Entry) -> str:
    return f"{entry.speedup:.6g}" if entry.speedup is not None else "n/a"


def run_label(path: Path, override: str | None) -> str:
    if override:
        return override
    if path.name == "success-diffs.md":
        return path.parent.name
    return path.stem


def render_original_group(report: Report) -> dict[Path, list[Entry]]:
    grouped: dict[Path, list[Entry]] = {}
    for entry in report.entries:
        grouped.setdefault(entry.original_kernel, []).append(entry)
    return grouped


def entry_names(entries: list[Entry]) -> str:
    return ", ".join(entry.exp_name for entry in entries) if entries else "-"


def changed_count(entries: list[Entry]) -> str:
    if not entries:
        return "-"
    changed = sum(1 for entry in entries if entry.file_diff.differs is True)
    unchanged = sum(1 for entry in entries if entry.file_diff.differs is False)
    unknown = len(entries) - changed - unchanged
    if unknown:
        return f"{changed}/{len(entries)} changed, {unknown} unknown"
    return f"{changed}/{len(entries)} changed"


def render_report(left: Report, right: Report, left_label: str, right_label: str) -> str:
    left_exps = set(left.by_exp)
    right_exps = set(right.by_exp)
    common_exps = left_exps & right_exps
    left_only_exps = left_exps - right_exps
    right_only_exps = right_exps - left_exps
    common_originals = left.original_paths & right.original_paths
    left_only_originals = left.original_paths - right.original_paths
    right_only_originals = right.original_paths - left.original_paths

    left_changed = sum(1 for entry in left.entries if entry.file_diff.differs is True)
    right_changed = sum(1 for entry in right.entries if entry.file_diff.differs is True)
    left_unchanged = sum(1 for entry in left.entries if entry.file_diff.differs is False)
    right_unchanged = sum(1 for entry in right.entries if entry.file_diff.differs is False)
    left_unknown = len(left.entries) - left_changed - left_unchanged
    right_unknown = len(right.entries) - right_changed - right_unchanged

    lines = [
        "# Success Diff Investigation",
        "",
        f"- {left_label}: `{left.path}`",
        f"- {right_label}: `{right.path}`",
        f"- Parsed success sections: {left_label}={len(left.entries)}, {right_label}={len(right.entries)}",
        f"- Declared success counts: {left_label}={left.declared_success_count}, {right_label}={right.declared_success_count}",
        f"- Success kernels differing from original files: {left_label}={left_changed} yes / {left_unchanged} no / {left_unknown} unknown; {right_label}={right_changed} yes / {right_unchanged} no / {right_unknown} unknown",
        "",
        "## Successful Exp Ids",
        "",
        f"- Common: {len(common_exps)} ({exp_list(common_exps)})",
        f"- Only in {left_label}: {len(left_only_exps)} ({exp_list(left_only_exps)})",
        f"- Only in {right_label}: {len(right_only_exps)} ({exp_list(right_only_exps)})",
        "",
        "## Original Kernels",
        "",
        f"- Common original kernel paths: {len(common_originals)}",
        f"- Only in {left_label}: {len(left_only_originals)}",
        f"- Only in {right_label}: {len(right_only_originals)}",
        "",
    ]

    if left_only_originals:
        lines.append(f"### Original kernels only in {left_label}")
        lines.extend(f"- `{path}`" for path in path_list(left_only_originals))
        lines.append("")
    if right_only_originals:
        lines.append(f"### Original kernels only in {right_label}")
        lines.extend(f"- `{path}`" for path in path_list(right_only_originals))
        lines.append("")

    if left_only_exps:
        lines.append(f"## Successes Only In {left_label}")
        lines.append("")
        lines.append("| exp | original kernel | differs from original | speedup | success hash |")
        lines.append("| --- | --- | --- | ---: | --- |")
        for exp_index in sorted(left_only_exps):
            entry = left.by_exp[exp_index]
            lines.append(
                f"| {entry.exp_name} | `{entry.original_kernel}` | {diff_summary(entry)} | "
                f"{speedup_text(entry)} | `{short_hash(entry.success_sha256)}` |"
            )
        lines.append("")

    if right_only_exps:
        lines.append(f"## Successes Only In {right_label}")
        lines.append("")
        lines.append("| exp | original kernel | differs from original | speedup | success hash |")
        lines.append("| --- | --- | --- | ---: | --- |")
        for exp_index in sorted(right_only_exps):
            entry = right.by_exp[exp_index]
            lines.append(
                f"| {entry.exp_name} | `{entry.original_kernel}` | {diff_summary(entry)} | "
                f"{speedup_text(entry)} | `{short_hash(entry.success_sha256)}` |"
            )
        lines.append("")

    common_rows: list[tuple[Entry, Entry]] = []
    for exp_index in sorted(common_exps):
        left_entry = left.by_exp[exp_index]
        right_entry = right.by_exp[exp_index]
        common_rows.append((left_entry, right_entry))

    if common_rows:
        lines.append("## Common Exp Comparison")
        lines.append("")
        lines.append(
            "| exp | same original path | same success file content | "
            f"{left_label} differs from original | {right_label} differs from original | "
            f"{left_label} speedup | {right_label} speedup |"
        )
        lines.append("| --- | --- | --- | --- | --- | ---: | ---: |")
        for left_entry, right_entry in common_rows:
            lines.append(
                f"| {left_entry.exp_name} | {yes_no(left_entry.original_kernel == right_entry.original_kernel)} | "
                f"{yes_no(left_entry.success_sha256 == right_entry.success_sha256)} | "
                f"{diff_summary(left_entry)} | {diff_summary(right_entry)} | "
                f"{speedup_text(left_entry)} | {speedup_text(right_entry)} |"
            )
        lines.append("")

    left_by_original = render_original_group(left)
    right_by_original = render_original_group(right)
    lines.append("## Original Kernel Groups")
    lines.append("")
    lines.append(
        "| original kernel | "
        "reports | "
        f"{left_label} exps | {right_label} exps | "
        f"{left_label} diff count | {right_label} diff count | "
        f"{left_label} unique success hashes | {right_label} unique success hashes | shared success hashes |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |")
    for original in sorted(left.original_paths | right.original_paths, key=str):
        left_entries = left_by_original.get(original, [])
        right_entries = right_by_original.get(original, [])
        left_hashes = {entry.success_sha256 for entry in left_entries if entry.success_sha256}
        right_hashes = {entry.success_sha256 for entry in right_entries if entry.success_sha256}
        reports = []
        if left_entries:
            reports.append(left_label)
        if right_entries:
            reports.append(right_label)
        lines.append(
            f"| `{original}` | "
            f"{', '.join(reports)} | "
            f"{entry_names(left_entries)} | "
            f"{entry_names(right_entries)} | "
            f"{changed_count(left_entries)} | {changed_count(right_entries)} | "
            f"{len(left_hashes)} | {len(right_hashes)} | {len(left_hashes & right_hashes)} |"
        )
    lines.append("")

    mismatched_markdown = [
        (left_label, entry)
        for entry in left.entries
        if entry.file_diff.differs is not None and entry.markdown_has_diff != entry.file_diff.differs
    ] + [
        (right_label, entry)
        for entry in right.entries
        if entry.file_diff.differs is not None and entry.markdown_has_diff != entry.file_diff.differs
    ]
    if mismatched_markdown:
        lines.append("## Markdown/File Diff Mismatches")
        lines.append("")
        lines.append("| report | exp | markdown has diff | file differs |")
        lines.append("| --- | --- | --- | --- |")
        for label, entry in mismatched_markdown:
            lines.append(
                f"| {label} | {entry.exp_name} | {yes_no(entry.markdown_has_diff)} | "
                f"{yes_no(entry.file_diff.differs)} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    try:
        left_path = args.left.expanduser().resolve(strict=False)
        right_path = args.right.expanduser().resolve(strict=False)
        if not left_path.is_file():
            raise ValueError(f"left report does not exist: {left_path}")
        if not right_path.is_file():
            raise ValueError(f"right report does not exist: {right_path}")

        left = parse_report(left_path)
        right = parse_report(right_path)
        output = render_report(
            left,
            right,
            run_label(left_path, args.left_label),
            run_label(right_path, args.right_label),
        )
        if args.output:
            output_path = args.output.expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output)
            print(f"wrote investigation report to {output_path}")
        else:
            print(output, end="")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
