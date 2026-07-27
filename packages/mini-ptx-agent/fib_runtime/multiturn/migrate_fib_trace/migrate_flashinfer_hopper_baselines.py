#!/usr/bin/env python3
"""Migrate FlashInfer Hopper-baseline trace artifacts into accrl-training.

The migration set is read from the markdown artifact produced by the clean-data
task. By default this script is a dry-run. It skips definitions with zero
workload rows and refuses to overwrite divergent existing target files unless
the requested operation is a reference rewrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MANIFEST_MD = Path(
    "/home/ubuntu/AccRL-exps/tasks/clean_data/artifacts/"
    "flashinfer_hopper_baseline_definitions.md"
)
DEFAULT_SOURCE_ROOT = Path("/home/ubuntu/flashinfer-trace")
DEFAULT_TARGET_ROOT = Path("/home/ubuntu/accrl-training")
ARCHIVE_MARKER = "## Original Python Reference"


@dataclass(frozen=True)
class Entry:
    family: str
    definition: str
    workload_rows: int
    solution_name: str


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_manifest(path: Path) -> list[Entry]:
    family: str | None = None
    entries: list[Entry] = []
    row_re = re.compile(
        r"^\| `(?P<definition>[^`]+)` \|\s*(?P<rows>\d+)\s*\|"
        r"\s*\d+\s*\| `(?P<solution>[^`]+)` \|$"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^## (.+)$", line)
        if heading:
            family = heading.group(1)
            continue
        match = row_re.match(line)
        if match:
            if family is None:
                raise ValueError(f"manifest row before family heading: {line}")
            entries.append(
                Entry(
                    family=family,
                    definition=match.group("definition"),
                    workload_rows=int(match.group("rows")),
                    solution_name=match.group("solution"),
                )
            )
    if not entries:
        raise ValueError(f"no manifest entries parsed from {path}")
    return entries


def iter_workload_blob_paths(workload_jsonl: Path) -> Iterable[Path]:
    seen: set[str] = set()
    with workload_jsonl.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            workload = row.get("workload") or {}
            inputs = workload.get("inputs") or {}
            outputs = workload.get("outputs") or {}
            for group in (inputs, outputs):
                if not isinstance(group, dict):
                    continue
                for spec in group.values():
                    if not isinstance(spec, dict):
                        continue
                    rel = spec.get("path")
                    if spec.get("type") == "safetensors" and isinstance(rel, str):
                        if not rel.startswith("./blob/"):
                            raise ValueError(
                                f"{workload_jsonl}:{lineno} has unsupported blob path {rel!r}"
                            )
                        if rel not in seen:
                            seen.add(rel)
                            yield Path(rel)


def archive_reference(description: str, reference: str) -> str:
    if ARCHIVE_MARKER in description:
        return description
    return (
        description.rstrip()
        + "\n\n"
        + ARCHIVE_MARKER
        + "\n\n"
        + "The operator was originally expressed as the Python reference below. "
        + "The active reference now uses the FlashInfer baseline wrapper so "
        + "reference profiling measures the same implementation as the baseline solution.\n\n"
        + "```python\n"
        + reference.rstrip()
        + "\n```\n"
    )


def solution_source(solution_path: Path, entry: Entry) -> str:
    solution = normalized_solution(solution_path, entry)
    return main_source(solution, solution_path)


def reference_source(solution_path: Path, entry: Entry) -> str:
    source = solution_source(solution_path, entry)
    if entry.family == "gdn":
        source = add_cuda_device_guard(source, solution_path)
        if "_decode_" in entry.definition:
            source = add_gdn_decode_reference_workspace(source, solution_path)
        return source
    if entry.family == "rmsnorm" and entry.definition.startswith("fused_add_rmsnorm_"):
        return add_fused_add_rmsnorm_reference_workspace(source, solution_path)
    if entry.family == "sampling" and entry.definition == "top_k_sampling_from_probs_v128256":
        return add_sampling_reference_output_snapshot(source, solution_path)
    return source


def add_sampling_reference_output_snapshot(content: str, solution_path: Path) -> str:
    marker = "# Reference-only latency alignment for this sampling shape."
    if marker in content:
        return content
    deterministic_needle = "        deterministic=False,\n"
    if deterministic_needle not in content:
        raise ValueError(f"{solution_path} sampling wrapper has unexpected deterministic flag")
    return content.replace(
        deterministic_needle,
        f"        {marker}\n"
        "        deterministic=True,\n",
        1,
    )


def add_fused_add_rmsnorm_reference_workspace(content: str, solution_path: Path) -> str:
    marker = "_REFERENCE_FUSED_ADD_RMSNORM_WORKSPACES"
    if marker in content:
        return content

    import_needle = "import flashinfer\n"
    helper = (
        "\n\n"
        "_REFERENCE_FUSED_ADD_RMSNORM_WORKSPACES = {}\n"
        "_REFERENCE_FUSED_ADD_RMSNORM_OUTPUTS = {}\n\n"
        "def _reference_fused_add_rmsnorm_workspace(hidden_states, residual):\n"
        "    key = (\n"
        "        hidden_states.data_ptr(),\n"
        "        residual.data_ptr(),\n"
        "        tuple(hidden_states.shape),\n"
        "        tuple(residual.shape),\n"
        "        hidden_states.dtype,\n"
        "        residual.dtype,\n"
        "        hidden_states.device,\n"
        "        residual.device,\n"
        "    )\n"
        "    workspace = _REFERENCE_FUSED_ADD_RMSNORM_WORKSPACES.get(key)\n"
        "    if workspace is None:\n"
        "        workspace = (hidden_states.clone(), residual.clone())\n"
        "        _REFERENCE_FUSED_ADD_RMSNORM_WORKSPACES[key] = workspace\n"
        "        _REFERENCE_FUSED_ADD_RMSNORM_OUTPUTS.pop(key, None)\n"
        "    return key, workspace[0], workspace[1]\n"
        "\n\n"
        "def _reference_fused_add_rmsnorm_output(key, output):\n"
        "    snapshot = _REFERENCE_FUSED_ADD_RMSNORM_OUTPUTS.get(key)\n"
        "    if snapshot is None:\n"
        "        snapshot = output.clone()\n"
        "        _REFERENCE_FUSED_ADD_RMSNORM_OUTPUTS[key] = snapshot\n"
        "        return snapshot\n"
        "    return output\n"
    )
    if import_needle not in content:
        raise ValueError(f"{solution_path} fused-add RMSNorm wrapper is missing import flashinfer")
    content = content.replace(import_needle, import_needle + helper, 1)

    shape_needle = "    batch_size, hidden_size = hidden_states.shape\n"
    workspace_line = (
        "    ref_key, hidden_states_for_kernel, residual_for_kernel = "
        "_reference_fused_add_rmsnorm_workspace(hidden_states, residual)\n"
    )
    if shape_needle not in content:
        raise ValueError(f"{solution_path} fused-add RMSNorm wrapper has unexpected shape block")
    content = content.replace(shape_needle, workspace_line + shape_needle, 1)

    call_needle = "    flashinfer.norm.fused_add_rmsnorm(hidden_states, residual, weight, EPS)\n"
    call_replacement = (
        "    flashinfer.norm.fused_add_rmsnorm(\n"
        "        hidden_states_for_kernel, residual_for_kernel, weight, EPS\n"
        "    )\n"
    )
    if call_needle not in content:
        raise ValueError(f"{solution_path} fused-add RMSNorm wrapper has unexpected kernel call")
    content = content.replace(call_needle, call_replacement, 1)

    return content.replace(
        "    return hidden_states\n",
        "    return _reference_fused_add_rmsnorm_output(ref_key, hidden_states_for_kernel)\n",
        1,
    )


def add_gdn_decode_reference_workspace(content: str, solution_path: Path) -> str:
    marker = "_REFERENCE_STATE_WORKSPACES"
    if marker in content:
        return content

    import_needle = "from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose\n"
    helper = (
        "\n\n"
        "_REFERENCE_STATE_WORKSPACES = {}\n"
        "_REFERENCE_STATE_SNAPSHOTS = {}\n\n"
        "def _reference_state_workspace(state):\n"
        "    key = (state.data_ptr(), tuple(state.shape), state.dtype, state.device)\n"
        "    workspace = _REFERENCE_STATE_WORKSPACES.get(key)\n"
        "    if workspace is None:\n"
        "        workspace = state.clone()\n"
        "        _REFERENCE_STATE_WORKSPACES[key] = workspace\n"
        "        _REFERENCE_STATE_SNAPSHOTS.pop(key, None)\n"
        "    return key, workspace\n"
        "\n\n"
        "def _reference_state_output(key, new_state):\n"
        "    snapshot = _REFERENCE_STATE_SNAPSHOTS.get(key)\n"
        "    if snapshot is None:\n"
        "        snapshot = new_state.clone()\n"
        "        _REFERENCE_STATE_SNAPSHOTS[key] = snapshot\n"
        "        return snapshot\n"
        "    return new_state\n"
    )
    if import_needle not in content:
        raise ValueError(f"{solution_path} is not a GDN decode wrapper")
    content = content.replace(import_needle, import_needle + helper, 1)

    guard_needle = "    with torch.cuda.device(q.device):\n"
    workspace_line = "        state_key, state_for_kernel = _reference_state_workspace(state)\n"
    if guard_needle not in content:
        raise ValueError(f"{solution_path} reference is missing CUDA device guard")
    content = content.replace(guard_needle, guard_needle + workspace_line, 1)

    state_arg = "            state=state,\n"
    if state_arg not in content:
        raise ValueError(f"{solution_path} GDN decode wrapper is missing state=state argument")
    content = content.replace(state_arg, "            state=state_for_kernel,\n", 1)
    return content.replace(
        "        return out, new_state\n",
        "        return out, _reference_state_output(state_key, new_state)\n",
        1,
    )


def add_cuda_device_guard(content: str, solution_path: Path) -> str:
    if "with torch.cuda.device(q.device):" in content:
        return content

    lines = content.rstrip().splitlines()
    run_idx = next((i for i, line in enumerate(lines) if line.startswith("def run(")), None)
    if run_idx is None:
        raise ValueError(f"{solution_path} main.py source does not define top-level run()")

    guarded = lines[: run_idx + 1]
    guarded.append("    with torch.cuda.device(q.device):")
    for line in lines[run_idx + 1 :]:
        guarded.append(("    " + line) if line else "")
    return "\n".join(guarded) + "\n"


def normalized_solution(solution_path: Path, entry: Entry) -> dict[str, Any]:
    solution = load_json(solution_path)
    source_definition = solution.get("definition")
    allowed_alias = f"{entry.definition}_flashinfer"
    if source_definition not in (entry.definition, allowed_alias):
        raise ValueError(
            f"{solution_path} has definition={source_definition!r}, "
            f"expected {entry.definition!r}"
        )
    if source_definition != entry.definition:
        solution = dict(solution)
        solution["definition"] = entry.definition
    if solution.get("name") != entry.solution_name:
        raise ValueError(
            f"{solution_path} has name={solution.get('name')!r}, "
            f"expected {entry.solution_name!r}"
        )
    spec = solution.get("spec") or {}
    if spec.get("language") == "python" and spec.get("destination_passing_style") is not False:
        solution = dict(solution)
        spec = dict(spec)
        spec["destination_passing_style"] = False
        solution["spec"] = spec
    targets = spec.get("target_hardware") or []
    if not any(t in targets for t in ("NVIDIA H20", "NVIDIA H100", "NVIDIA H200")):
        raise ValueError(f"{solution_path} does not advertise Hopper target hardware")
    return solution


def main_source(solution: dict[str, Any], solution_path: Path) -> str:
    sources = solution.get("sources") or []
    main_sources = [s for s in sources if s.get("path") == "main.py"]
    if len(main_sources) != 1:
        raise ValueError(f"{solution_path} expected exactly one main.py source")
    content = main_sources[0].get("content")
    if not isinstance(content, str) or "def run(" not in content:
        raise ValueError(f"{solution_path} main.py source does not define run()")
    return content


def solution_json_bytes(solution: dict[str, Any]) -> bytes:
    return (json.dumps(solution, indent=2) + "\n").encode("utf-8")


def copy_solution_file(
    src: Path,
    dst: Path,
    entry: Entry,
    apply: bool,
    overwrite: bool,
    actions: list[str],
) -> None:
    solution = normalized_solution(src, entry)
    desired = solution_json_bytes(solution)
    normalized = load_json(src).get("definition") != entry.definition
    action_prefix = "copy normalized" if normalized else "copy"

    if dst.exists():
        current = dst.read_bytes()
        if current == desired:
            actions.append(f"unchanged {dst}")
            return
        if not overwrite:
            raise FileExistsError(f"target exists and differs: {dst}")
        actions.append(f"overwrite {dst}")
    else:
        actions.append(f"{action_prefix} {dst}")

    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(dst.parent)) as f:
            tmp = Path(f.name)
            f.write(desired)
        tmp.replace(dst)


def copy_file(src: Path, dst: Path, apply: bool, overwrite: bool, actions: list[str]) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        if sha256(src) == sha256(dst):
            actions.append(f"unchanged {dst}")
            return
        if not overwrite:
            raise FileExistsError(f"target exists and differs: {dst}")
        actions.append(f"overwrite {dst}")
    else:
        actions.append(f"copy {dst}")
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def find_solution_path(source_root: Path, entry: Entry) -> Path:
    rel = Path(entry.family) / entry.definition / f"{entry.solution_name}.json"
    candidates = [
        source_root / "solutions" / rel,
        source_root / "solutions" / "baseline" / rel,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "missing solution for "
        f"{entry.family}/{entry.definition}/{entry.solution_name}; tried "
        + ", ".join(str(c) for c in candidates)
    )


def copy_definition_file(
    src: Path,
    dst: Path,
    apply: bool,
    overwrite: bool,
    prepare_reference: bool,
    refresh_prepared_reference: bool,
    actions: list[str],
) -> str:
    if not src.exists():
        raise FileNotFoundError(src)
    if not dst.exists():
        actions.append(f"copy {dst}")
        if apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return "copied"
    if sha256(src) == sha256(dst):
        actions.append(f"unchanged {dst}")
        return "unchanged"

    if prepare_reference and not refresh_prepared_reference:
        target_definition = load_json(dst)
        if (
            isinstance(target_definition.get("reference"), str)
            and ARCHIVE_MARKER in str(target_definition.get("description", ""))
        ):
            actions.append(f"preserve existing prepared definition {dst}")
            return "preserved_prepared"

    if not overwrite:
        raise FileExistsError(f"target exists and differs: {dst}")
    actions.append(f"overwrite {dst}")
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return "overwritten"


def validate_workload_rows(path: Path, entry: Entry) -> None:
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row.get("definition") != entry.definition:
                    raise ValueError(f"{path} contains row for {row.get('definition')!r}")
                count += 1
    if count != entry.workload_rows:
        raise ValueError(f"{path} has {count} rows, expected {entry.workload_rows}")


def migrate_entry(
    entry: Entry,
    source_root: Path,
    target_root: Path,
    apply: bool,
    overwrite: bool,
    prepare_reference: bool,
    refresh_prepared_reference: bool,
    solutions_only: bool,
    actions: list[str],
) -> None:
    definition_src = source_root / "definitions" / entry.family / f"{entry.definition}.json"
    workload_src = source_root / "workloads" / entry.family / f"{entry.definition}.jsonl"
    solution_src = find_solution_path(source_root, entry)
    definition_dst = target_root / "definitions" / entry.family / f"{entry.definition}.json"
    workload_dst = target_root / "workloads" / entry.family / f"{entry.definition}.jsonl"
    solution_dst = (
        target_root
        / "solutions"
        / entry.family
        / entry.definition
        / f"{entry.solution_name}.json"
    )

    definition = load_json(definition_src)
    if definition.get("name") != entry.definition:
        raise ValueError(f"{definition_src} has unexpected name={definition.get('name')!r}")
    validate_workload_rows(workload_src, entry)
    new_reference = reference_source(solution_src, entry)

    if solutions_only:
        copy_solution_file(solution_src, solution_dst, entry, apply, overwrite, actions)
        return

    definition_copy_status = copy_definition_file(
        definition_src,
        definition_dst,
        apply,
        overwrite,
        prepare_reference,
        refresh_prepared_reference,
        actions,
    )
    copy_file(workload_src, workload_dst, apply, overwrite, actions)
    copy_solution_file(solution_src, solution_dst, entry, apply, overwrite, actions)

    for rel_blob in iter_workload_blob_paths(workload_src):
        src_blob = source_root / rel_blob
        dst_blob = target_root / rel_blob
        copy_file(src_blob, dst_blob, apply, overwrite, actions)

    if prepare_reference:
        if definition_copy_status == "preserved_prepared":
            actions.append(f"reference preserved {definition_dst}")
            return
        target_definition = load_json(definition_dst if definition_dst.exists() else definition_src)
        old_reference = target_definition.get("reference")
        if not isinstance(old_reference, str) or not old_reference.strip():
            raise ValueError(f"{definition_dst} has no string reference field")
        new_description = archive_reference(target_definition.get("description", ""), old_reference)
        changed = (
            target_definition.get("reference") != new_reference
            or target_definition.get("description", "") != new_description
        )
        if changed:
            actions.append(f"rewrite reference {definition_dst}")
            if apply:
                target_definition["description"] = new_description
                target_definition["reference"] = new_reference
                dump_json(definition_dst, target_definition)
        else:
            actions.append(f"reference unchanged {definition_dst}")


def selected_entries(entries: list[Entry], families: set[str] | None, definitions: set[str] | None) -> list[Entry]:
    selected = []
    for entry in entries:
        if entry.workload_rows == 0:
            continue
        if families and entry.family not in families:
            continue
        if definitions and entry.definition not in definitions:
            continue
        selected.append(entry)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-md", type=Path, default=DEFAULT_MANIFEST_MD)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--family", action="append", help="limit to one family; repeatable")
    parser.add_argument("--definition", action="append", help="limit to one definition; repeatable")
    parser.add_argument("--apply", action="store_true", help="write changes; default is dry-run")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow overwriting divergent copied artifacts; reference rewrites are still allowed",
    )
    parser.add_argument(
        "--no-prepare-reference",
        action="store_true",
        help="copy artifacts only; do not replace definition reference with FlashInfer wrapper",
    )
    parser.add_argument(
        "--refresh-prepared-references",
        action="store_true",
        help="regenerate references even when the target definition is already prepared",
    )
    parser.add_argument("--summary-only", action="store_true", help="print action counts only")
    parser.add_argument(
        "--solutions-only",
        action="store_true",
        help="only copy/normalize selected solution JSONs; skip definitions, workloads, blobs, references",
    )
    args = parser.parse_args()

    entries = parse_manifest(args.manifest_md)
    families = set(args.family) if args.family else None
    definitions = set(args.definition) if args.definition else None
    chosen = selected_entries(entries, families, definitions)
    if not chosen:
        raise ValueError("selection is empty")

    actions: list[str] = []
    for entry in chosen:
        actions.append(
            f"entry {entry.family}/{entry.definition} rows={entry.workload_rows} "
            f"solution={entry.solution_name}"
        )
        migrate_entry(
            entry=entry,
            source_root=args.source_root,
            target_root=args.target_root,
            apply=args.apply,
            overwrite=args.overwrite,
            prepare_reference=not args.no_prepare_reference,
            refresh_prepared_reference=args.refresh_prepared_references,
            solutions_only=args.solutions_only,
            actions=actions,
        )

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"{mode}: selected {len(chosen)} workload-bearing entries")
    print("Skipped zero-workload entries by design")
    if args.summary_only:
        counts = Counter(action.split(" ", 1)[0] for action in actions)
        for key in sorted(counts):
            print(f"{key}: {counts[key]}")
        return 0
    for action in actions:
        print(action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
