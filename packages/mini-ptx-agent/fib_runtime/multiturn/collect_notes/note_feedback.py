#!/usr/bin/env python3
"""Retrieve pair-note feedback for the current failed kernel.

The retrieval corpus is the notes JSONL emitted by collect_kernel_fix_notes.py.
Records are grouped by historical wrong kernel. At feedback time we rank
historical wrong kernels with BM25 over comment-stripped source tokens, collect
notes from all fixed-kernel variants in those groups, deduplicate instruction
notes, and render at most N notes for appending to the next LLM user message.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_NOTES_JSONL = Path(
    "/home/ubuntu/AccRL-exps/tasks/collect_notes/outputs/"
    "mha-d128-4def-kernel-fix-notes-full/notes.jsonl"
)

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


@dataclass(frozen=True)
class RetrievalConfig:
    notes_jsonl: Path
    top_k: int = 3
    max_notes: int = 8
    min_similarity: float = 0.0
    definition_prefixes: tuple[str, ...] = ()
    enabled: bool = False
    feedback_mode: int = 1


@dataclass
class NoteRecord:
    metadata: dict[str, Any]
    summary: str
    tags: list[Any]
    instruction_notes: list[dict[str, Any]]
    wrong_kernel_source: str
    vector: Counter[str]


@dataclass
class WrongKernelGroup:
    wrong_kernel_path: str
    definition: str
    trajectory_id: str
    records: list[NoteRecord]
    vector: Counter[str]


def _feedback_mode(value: str | None) -> int:
    text = str(value or "").strip().lower()
    if text in {"", "0", "false", "no", "off"}:
        return 0
    if text in {"1", "true", "yes", "on"}:
        return 1
    try:
        return max(int(text), 0)
    except ValueError:
        return 1


def config_from_env() -> RetrievalConfig:
    feedback_mode = _feedback_mode(os.environ.get("ACCRL_NOTE_FEEDBACK"))
    notes_path = Path(os.environ.get("ACCRL_NOTE_FEEDBACK_NOTES_JSONL") or DEFAULT_NOTES_JSONL)
    top_k = int(os.environ.get("ACCRL_NOTE_FEEDBACK_TOP_K") or "3")
    max_notes = int(os.environ.get("ACCRL_NOTE_FEEDBACK_MAX_NOTES") or "8")
    min_similarity = float(os.environ.get("ACCRL_NOTE_FEEDBACK_MIN_SIMILARITY") or "0")
    prefixes = tuple(
        value.strip()
        for value in (os.environ.get("ACCRL_NOTE_FEEDBACK_DEFINITION_PREFIXES") or "").split(",")
        if value.strip()
    )
    return RetrievalConfig(
        notes_jsonl=notes_path,
        top_k=max(top_k, 0),
        max_notes=max(max_notes, 0),
        min_similarity=min_similarity,
        definition_prefixes=prefixes,
        enabled=feedback_mode > 0,
        feedback_mode=feedback_mode,
    )


def strip_cpp_comments(source: str) -> str:
    out: list[str] = []
    i = 0
    n = len(source)
    in_string: str | None = None
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if in_string:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(source[i + 1])
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue
        if ch in {'"', "'"}:
            in_string = ch
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            i += 2
            while i < n and source[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (source[i] == "*" and source[i + 1] == "/"):
                i += 1
            i = min(i + 2, n)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def tokenize(source: str) -> Counter[str]:
    source = strip_cpp_comments(source)
    tokens = Counter(token.lower() for token in TOKEN_RE.findall(source))
    for token in list(tokens):
        if len(token) == 1 and not token.isdigit():
            tokens.pop(token, None)
    return tokens


def bm25_score(
    query: Counter[str],
    document: Counter[str],
    *,
    document_frequencies: Counter[str],
    document_count: int,
    average_document_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Vanilla BM25 score for one tokenized query and document."""
    if not query or not document or document_count <= 0 or average_document_length <= 0:
        return 0.0
    document_length = sum(document.values())
    length_norm = k1 * (1.0 - b + b * document_length / average_document_length)
    score = 0.0
    for token in query:
        term_frequency = document.get(token, 0)
        if term_frequency <= 0:
            continue
        df = document_frequencies.get(token, 0)
        idf = max(0.0, math.log((document_count - df + 0.5) / (df + 0.5) + 1.0))
        score += idf * (term_frequency * (k1 + 1.0)) / (term_frequency + length_norm)
    return score


def _read_text(path_text: str) -> str:
    try:
        return Path(path_text).expanduser().read_text(errors="replace")
    except OSError:
        return ""


def _definition_allowed(definition: str, prefixes: tuple[str, ...]) -> bool:
    if not prefixes:
        return True
    return any(definition.startswith(prefix) for prefix in prefixes)


def _load_records(notes_jsonl: Path, prefixes: tuple[str, ...]) -> list[NoteRecord]:
    records: list[NoteRecord] = []
    with notes_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            metadata = payload.get("metadata") or {}
            if not isinstance(metadata, dict):
                continue
            definition = str(metadata.get("definition") or "")
            if not _definition_allowed(definition, prefixes):
                continue
            wrong_path = str(metadata.get("wrong_kernel_path") or "")
            wrong_source = _read_text(wrong_path)
            if not wrong_source:
                continue
            instruction_notes = payload.get("instruction_notes") or []
            if not isinstance(instruction_notes, list) or not instruction_notes:
                continue
            vector = tokenize(wrong_source)
            records.append(
                NoteRecord(
                    metadata=metadata,
                    summary=str(payload.get("summary") or ""),
                    tags=payload.get("tags") if isinstance(payload.get("tags"), list) else [],
                    instruction_notes=[note for note in instruction_notes if isinstance(note, dict)],
                    wrong_kernel_source=wrong_source,
                    vector=vector,
                )
            )
    return records


def _build_groups(records: list[NoteRecord]) -> list[WrongKernelGroup]:
    grouped: dict[str, list[NoteRecord]] = {}
    for record in records:
        grouped.setdefault(str(record.metadata.get("wrong_kernel_path") or ""), []).append(record)

    groups: list[WrongKernelGroup] = []
    for wrong_path, group_records in grouped.items():
        vector: Counter[str] = Counter()
        for record in group_records:
            vector.update(record.vector)
        first = group_records[0]
        groups.append(
            WrongKernelGroup(
                wrong_kernel_path=wrong_path,
                definition=str(first.metadata.get("definition") or ""),
                trajectory_id=str(first.metadata.get("trajectory_id") or ""),
                records=group_records,
                vector=vector,
            )
        )
    return groups


def bm25_document_frequencies(groups: tuple[WrongKernelGroup, ...]) -> Counter[str]:
    frequencies: Counter[str] = Counter()
    for group in groups:
        frequencies.update(group.vector.keys())
    return frequencies


def average_document_length(groups: tuple[WrongKernelGroup, ...]) -> float:
    if not groups:
        return 0.0
    return sum(sum(group.vector.values()) for group in groups) / len(groups)


@lru_cache(maxsize=16)
def _cached_groups(notes_jsonl: str, prefixes: tuple[str, ...]) -> tuple[WrongKernelGroup, ...]:
    records = _load_records(Path(notes_jsonl), prefixes)
    return tuple(_build_groups(records))


def _note_key(note: dict[str, Any]) -> str:
    parts = [
        note.get("instruction", ""),
        note.get("shape_context", ""),
        note.get("correct_example", ""),
        note.get("operand_contract", []),
        note.get("required_sequence", []),
        note.get("diagnostics", []),
        note.get("do_not_do", []),
    ]
    return json.dumps(parts, sort_keys=True, ensure_ascii=False)


def _format_list(values: Any, *, indent: str = "  ") -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return [f"{indent}- {value}" for value in values if str(value).strip()]


def _append_field(lines: list[str], note: dict[str, Any], key: str, label: str) -> None:
    value = note.get(key)
    if isinstance(value, list):
        rendered = _format_list(value, indent="    ")
        if rendered:
            lines.append(f"  {label}:")
            lines.extend(rendered)
        return
    text = str(value or "").strip()
    if text:
        lines.append(f"  {label}: {text}")


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _correct_kernel_version(record: NoteRecord) -> int:
    return _safe_int(record.metadata.get("correct_kernel_version")) or 0


@lru_cache(maxsize=4096)
def _success_speedup(exp_dir: str, trajectory_id: str, version: int) -> float | None:
    record_path = Path(exp_dir) / "success" / trajectory_id / "record.json"
    try:
        records = json.loads(record_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(records, list):
        return None
    for item in records:
        if not isinstance(item, dict) or _safe_int(item.get("version")) != version:
            continue
        speedups: list[float] = []
        for trace in item.get("traces") or []:
            if not isinstance(trace, dict):
                continue
            performance = trace.get("evaluation", {}).get("performance", {})
            speedup = performance.get("speedup_factor") if isinstance(performance, dict) else None
            if isinstance(speedup, (int, float)):
                speedups.append(float(speedup))
        return min(speedups) if speedups else None
    return None


def _fixed_speedup(record: NoteRecord) -> float | None:
    exp_dir = str(record.metadata.get("exp_dir") or "")
    trajectory_id = str(record.metadata.get("trajectory_id") or "")
    if not exp_dir or not trajectory_id:
        return None
    return _success_speedup(exp_dir, trajectory_id, _correct_kernel_version(record))


def _best_fixed_record(group: WrongKernelGroup) -> NoteRecord:
    def rank(record: NoteRecord) -> tuple[float, int]:
        speedup = _fixed_speedup(record)
        return (-1.0 if speedup is None else speedup, _correct_kernel_version(record))

    return max(group.records, key=rank)


def _append_instruction_note(
    lines: list[str],
    note: dict[str, Any],
    *,
    index: int,
) -> None:
    lines.append(f"{index}. {str(note.get('instruction') or 'instruction note').strip()}")
    _append_field(lines, note, "shape_context", "Shape/context")
    _append_field(lines, note, "operand_contract", "Operand contract")
    _append_field(lines, note, "required_sequence", "Required sequence")
    _append_field(lines, note, "diagnostics", "Diagnostics")
    _append_field(lines, note, "do_not_do", "Do not do")
    correct_example = str(note.get("correct_example") or "").strip()
    if correct_example:
        lines.append("  Correct example:")
        lines.append("```cpp")
        lines.append(correct_example)
        lines.append("```")
    wrong_example = str(note.get("wrong_example") or "").strip()
    if wrong_example:
        lines.append("  Wrong example:")
        lines.append("```cpp")
        lines.append(wrong_example)
        lines.append("```")
    lines.append("")


def _render_notes_only_feedback(
    selected: list[tuple[float, WrongKernelGroup]],
    *,
    config: RetrievalConfig,
) -> str:
    lines = [
        "",
        "## Retrieved repair notes from fixing similar historical wrong kernels",
        ""
    ]
    seen_notes: set[str] = set()
    emitted = 0
    for rank, (score, group) in enumerate(selected, 1):
        if emitted >= config.max_notes:
            break
        lines.append(
            f"### Similar wrong kernel {rank}: bm25={score:.3f}, "
            f"definition={group.definition}, fixed_variants={len(group.records)}"
        )
        lines.append(f"source_wrong_kernel_path: {group.wrong_kernel_path}")
        lines.append("")

        for record in group.records:
            if emitted >= config.max_notes:
                break
            correct_path = str(record.metadata.get("correct_kernel_path") or "")
            correct_version = str(record.metadata.get("correct_kernel_version") or "")
            if correct_path:
                lines.append(f"Fixed variant: version={correct_version}, path={correct_path}")
            summary = record.summary.strip()
            if summary:
                lines.append(f"Summary: {summary}")
            for note in record.instruction_notes:
                if emitted >= config.max_notes:
                    break
                key = _note_key(note)
                if key in seen_notes:
                    continue
                seen_notes.add(key)
                emitted += 1
                _append_instruction_note(lines, note, index=emitted)
        lines.append("")

    if emitted == 0:
        return ""
    return "\n".join(lines).rstrip()


def _render_best_fixed_kernel_feedback(
    selected: list[tuple[float, WrongKernelGroup]],
    *,
    config: RetrievalConfig,
) -> str:
    lines = [
        "",
        "## Retrieved best fixed kernels and notes from similar historical wrong kernels",
        "",
        "Use each fixed kernel as a concrete repair example of the shape/layout/instruction contracts.",
        "",
    ]
    emitted_groups = 0
    for rank, (score, group) in enumerate(selected, 1):
        record = _best_fixed_record(group)
        correct_path = str(record.metadata.get("correct_kernel_path") or "")
        correct_source = _read_text(correct_path)
        if not correct_source:
            continue
        emitted_groups += 1
        speedup = _fixed_speedup(record)
        speedup_text = "unknown" if speedup is None else f"{speedup:.6g}"
        lines.append(
            f"### Similar error kernel {rank}: bm25={score:.3f}, "
            f"definition={group.definition}, fixed_variants={len(group.records)}"
        )
        lines.append(f"source_error_kernel_path: {group.wrong_kernel_path}")
        lines.append(f"best_fixed_kernel_path: {correct_path}")
        summary = record.summary.strip()
        if summary:
            lines.append(f"Summary: {summary}")
        lines.append("")
        lines.append("Best fixed kernel:")
        lines.append("```cpp")
        lines.append(correct_source.rstrip())
        lines.append("```")
        lines.append("")
        lines.append("Notes for this fixed kernel:")
        notes = record.instruction_notes[: config.max_notes]
        for note_index, note in enumerate(notes, 1):
            _append_instruction_note(lines, note, index=note_index)
        lines.append("")

    if emitted_groups == 0:
        return ""
    return "\n".join(lines).rstrip()


def retrieve_note_feedback(
    kernel_source: str,
    *,
    definition: str | None = None,
    config: RetrievalConfig | None = None,
) -> str:
    config = config or config_from_env()
    if not config.enabled or config.top_k <= 0 or config.max_notes <= 0:
        return ""
    if not config.notes_jsonl.is_file():
        return ""

    prefixes = config.definition_prefixes

    query = tokenize(kernel_source)
    if not query:
        return ""

    groups = _cached_groups(str(config.notes_jsonl), prefixes)
    filtered_groups = tuple(
        group
        for group in groups
        if prefixes or not definition or group.definition == definition
    )
    document_frequencies = bm25_document_frequencies(filtered_groups)
    average_length = average_document_length(filtered_groups)
    scored = [
        (
            bm25_score(
                query,
                group.vector,
                document_frequencies=document_frequencies,
                document_count=len(filtered_groups),
                average_document_length=average_length,
            ),
            group,
        )
        for group in filtered_groups
    ]
    scored = [
        item
        for item in scored
        if item[0] >= config.min_similarity and item[0] > 0.0
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[: config.top_k]
    if not selected:
        return ""

    if config.feedback_mode == 2:
        return _render_best_fixed_kernel_feedback(selected, config=config)
    return _render_notes_only_feedback(selected, config=config)


def append_note_feedback(feedback: str, kernel_source: str, *, definition: str | None = None) -> str:
    extra = retrieve_note_feedback(kernel_source, definition=definition)
    if not extra:
        return feedback
    return f"{feedback}\n{extra}"
