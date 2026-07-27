#!/usr/bin/env python3
"""Check that accrl-training solutions match flashinfer-trace counterparts."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TARGET_ROOT = Path("/home/ubuntu/accrl-training")
DEFAULT_SOURCE_ROOT = Path("/home/ubuntu/flashinfer-trace")


@dataclass(frozen=True)
class CheckResult:
    status: str
    rel_path: Path
    target_path: Path
    candidate_paths: tuple[Path, ...]
    matched_path: Path | None = None
    detail: str = ""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def normalized_solution_bytes(path: Path, rel_path: Path) -> bytes:
    data = read_json(path)
    if len(rel_path.parts) < 3:
        raise ValueError(f"{path} is not under family/definition/name.json")

    expected_definition = rel_path.parts[-2]
    expected_name = rel_path.stem
    source_definition = data.get("definition")
    allowed_alias = f"{expected_definition}_flashinfer"
    if source_definition not in (expected_definition, allowed_alias):
        raise ValueError(
            f"{path} has definition={source_definition!r}, expected {expected_definition!r}"
        )
    if data.get("name") != expected_name:
        raise ValueError(f"{path} has name={data.get('name')!r}, expected {expected_name!r}")

    normalized = dict(data)
    normalized["definition"] = expected_definition
    spec = normalized.get("spec")
    if isinstance(spec, dict) and spec.get("language") == "python":
        spec = dict(spec)
        if spec.get("destination_passing_style") is not False:
            spec["destination_passing_style"] = False
        normalized["spec"] = spec

    return json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")


def candidate_paths(source_root: Path, rel_path: Path) -> tuple[Path, ...]:
    seen: set[Path] = set()
    candidates = [
        source_root / "solutions" / rel_path,
        source_root / "solutions" / "baseline" / rel_path,
    ]
    unique = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return tuple(unique)


def iter_target_solutions(
    target_root: Path,
    families: set[str] | None,
    definitions: set[str] | None,
    solutions: set[str] | None,
) -> list[tuple[Path, Path]]:
    solutions_root = target_root / "solutions"
    if not solutions_root.is_dir():
        raise FileNotFoundError(f"target solutions directory does not exist: {solutions_root}")

    selected: list[tuple[Path, Path]] = []
    for path in sorted(solutions_root.rglob("*.json")):
        rel_path = path.relative_to(solutions_root)
        family = rel_path.parts[0] if rel_path.parts else ""
        definition = rel_path.parts[-2] if len(rel_path.parts) >= 2 else ""
        solution = rel_path.stem
        if families and family not in families:
            continue
        if definitions and definition not in definitions:
            continue
        if solutions and solution not in solutions:
            continue
        selected.append((rel_path, path))
    return selected


def compare_one(
    rel_path: Path,
    target_path: Path,
    source_root: Path,
    normalized: bool,
) -> CheckResult:
    candidates = candidate_paths(source_root, rel_path)
    existing_candidates = tuple(path for path in candidates if path.exists())
    if not existing_candidates:
        return CheckResult("missing", rel_path, target_path, candidates)

    try:
        target_bytes = (
            normalized_solution_bytes(target_path, rel_path)
            if normalized
            else target_path.read_bytes()
        )
    except Exception as exc:
        return CheckResult("error", rel_path, target_path, existing_candidates, detail=str(exc))

    candidate_hashes = []
    for candidate in existing_candidates:
        try:
            source_bytes = (
                normalized_solution_bytes(candidate, rel_path)
                if normalized
                else candidate.read_bytes()
            )
        except Exception as exc:
            return CheckResult(
                "error",
                rel_path,
                target_path,
                existing_candidates,
                detail=f"{candidate}: {exc}",
            )
        if source_bytes == target_bytes:
            return CheckResult("match", rel_path, target_path, existing_candidates, candidate)
        candidate_hashes.append(f"{candidate} sha256={sha256_bytes(source_bytes)}")

    detail = f"target sha256={sha256_bytes(target_bytes)}; " + "; ".join(candidate_hashes)
    return CheckResult("different", rel_path, target_path, existing_candidates, detail=detail)


def unified_diff(target_path: Path, source_path: Path, max_lines: int) -> list[str]:
    target_lines = target_path.read_text(encoding="utf-8", errors="replace").splitlines()
    source_lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
    diff = list(
        difflib.unified_diff(
            source_lines,
            target_lines,
            fromfile=str(source_path),
            tofile=str(target_path),
            lineterm="",
        )
    )
    if max_lines > 0 and len(diff) > max_lines:
        return diff[:max_lines] + [f"... diff truncated after {max_lines} lines"]
    return diff


def print_result(result: CheckResult, show_diff: bool, diff_lines: int) -> None:
    rel = result.rel_path.as_posix()
    if result.status == "missing":
        print(f"MISSING   {rel}")
        for path in result.candidate_paths:
            print(f"          tried {path}")
        return
    if result.status == "different":
        print(f"DIFFERENT {rel}")
        print(f"          {result.detail}")
        if show_diff and result.candidate_paths:
            for line in unified_diff(result.target_path, result.candidate_paths[0], diff_lines):
                print(f"          {line}")
        return
    if result.status == "error":
        print(f"ERROR     {rel}")
        print(f"          {result.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--family", action="append", help="limit to one family; repeatable")
    parser.add_argument("--definition", action="append", help="limit to one definition; repeatable")
    parser.add_argument("--solution", action="append", help="limit to one solution name; repeatable")
    parser.add_argument(
        "--normalized",
        action="store_true",
        help="compare canonical JSON after applying the migration script's expected solution normalization",
    )
    parser.add_argument(
        "--show-matches",
        action="store_true",
        help="print matching files as well as failures",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=50,
        help="maximum failures to print; use 0 to suppress per-file failures",
    )
    parser.add_argument("--diff", action="store_true", help="print a unified diff for divergent files")
    parser.add_argument("--diff-lines", type=int, default=80, help="maximum diff lines per file")
    args = parser.parse_args()

    if args.max_failures < 0:
        raise ValueError("--max-failures must be non-negative")
    if args.diff_lines < 0:
        raise ValueError("--diff-lines must be non-negative")

    selected = iter_target_solutions(
        args.target_root,
        set(args.family) if args.family else None,
        set(args.definition) if args.definition else None,
        set(args.solution) if args.solution else None,
    )
    if not selected:
        raise ValueError("selection is empty")

    results = [
        compare_one(rel_path, target_path, args.source_root, args.normalized)
        for rel_path, target_path in selected
    ]

    counts = {status: 0 for status in ("match", "missing", "different", "error")}
    for result in results:
        counts[result.status] += 1

    mode = "normalized JSON" if args.normalized else "byte-for-byte"
    print(f"mode: {mode}")
    print(f"target: {args.target_root / 'solutions'}")
    print(f"source: {args.source_root / 'solutions'}")
    print(
        "checked={checked} matches={match} missing={missing} different={different} errors={error}".format(
            checked=len(results),
            **counts,
        )
    )

    printed_failures = 0
    for result in results:
        if result.status == "match":
            if args.show_matches:
                print(f"MATCH     {result.rel_path.as_posix()} -> {result.matched_path}")
            continue
        if printed_failures < args.max_failures:
            print_result(result, args.diff, args.diff_lines)
        printed_failures += 1

    if printed_failures > args.max_failures:
        print(f"... suppressed {printed_failures - args.max_failures} additional failures")

    return 0 if counts["missing"] == counts["different"] == counts["error"] == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        sys.exit(1)
