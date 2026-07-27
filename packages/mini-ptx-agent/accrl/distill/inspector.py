#!/usr/bin/env python3
"""Reasoning-pair inspector for a single distillation JSONL file.

Usage:
    python -m accrl.distill.inspector <path-to-jsonl>

Supports these record shapes:
    - {system_prompt, input, reasoning, thinking, metadata}   (reasoning_pairs.jsonl)
    - {system_prompt, task_prompt, reasoning, kernel_code, ...}  (reasoning_raw.jsonl)
    - {messages: [...], metadata}                    (sft_dataset.jsonl)

Keybindings:
    ]  / [                    next / previous record
    Ctrl-Home / Ctrl-End      first / last record
    Tab / Shift-Tab           next / previous section
    Arrow keys / PgUp / PgDn  scroll within the current section
    q                         quit

The TextArea shows full section content with native scrolling (arrow keys,
PgUp/PgDn, mouse wheel). Record/section navigation uses Ctrl-modified keys
so it doesn't conflict with cursor movement inside the viewer.
"""

import json
import re
from pathlib import Path

import typer
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Header, Static, TextArea

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_CURRENT_KERNEL_RE = re.compile(
    r"(?:### Turn \d+ — Kernel \(CURRENT\)|## Expert's Kernel)\s*"
    r"```(?:cpp|cuda|c\+\+)?\n(.*?)\n```",
    re.DOTALL,
)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                records.append({"__error__": f"line {i}: {e}", "__raw__": line[:500]})
    return records


def _record_to_sections(rec: dict) -> tuple[list[tuple[str, str]], dict]:
    """Split a JSONL record into (label, text) sections plus a metadata dict."""
    if "__error__" in rec:
        return [("ERROR", rec["__error__"]), ("RAW", rec.get("__raw__", ""))], {}

    if isinstance(rec.get("messages"), list):
        sections = []
        for m in rec["messages"]:
            role = (m.get("role") or m.get("type") or "unknown").upper()
            content = m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, indent=2)
            sections.append((role, content))
        return sections, rec.get("metadata", {}) or {}

    if "reasoning" in rec:
        sections: list[tuple[str, str]] = []
        if rec.get("task_prompt"):
            sections.append(("TASK", rec["task_prompt"]))
        if rec.get("input"):
            sections.append(("PROMPT", rec["input"]))
        if rec.get("prev_kernel_code"):
            sections.append(("PREV KERNEL", rec["prev_kernel_code"]))
        if rec.get("prev_feedback"):
            sections.append(("PREV FEEDBACK", rec["prev_feedback"]))
        if rec.get("reasoning"):
            sections.append(("VISIBLE REASONING", rec["reasoning"]))
        if rec.get("thinking"):
            sections.append(("HIDDEN REASONING", rec["thinking"]))
        if rec.get("kernel_code"):
            sections.append(("EXPERT KERNEL", rec["kernel_code"]))
        elif rec.get("input"):
            current_kernel = _extract_current_kernel(rec["input"])
            if current_kernel:
                sections.append(("EXPERT KERNEL", current_kernel))
        if rec.get("system_prompt"):
            sections.append(("DISTILL SYSTEM", rec["system_prompt"]))
        meta = dict(rec["metadata"]) if isinstance(rec.get("metadata"), dict) else {}
        for k in (
            "exp_id", "turn", "speedup", "passed", "improved",
            "failure_type", "definition_name", "model", "reasoning_len",
        ):
            if k in rec and k not in meta:
                meta[k] = rec[k]
        return sections, meta

    sections = [
        (k.upper(), v if isinstance(v, str) else json.dumps(v, indent=2))
        for k, v in rec.items()
    ]
    return sections, {}


def _extract_current_kernel(prompt: str) -> str:
    """Extract the current/expert kernel code block embedded in a distill prompt."""
    match = _CURRENT_KERNEL_RE.search(prompt)
    return match.group(1).strip() if match else ""


def _format_meta(meta: dict) -> str:
    if not meta:
        return ""

    def fmt_value(value: object) -> str:
        if not isinstance(value, str):
            value = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        return " ".join(value.split())

    keys = [k for k in meta if k != "pair_key"]
    return "  |  ".join(f"{k}={fmt_value(meta[k])}" for k in keys if meta[k] is not None)


DEFAULT_CSS = """
Screen { background: $surface; }
#meta {
    background: $boost;
    color: $text;
    padding: 0 1;
    height: 1;
}
#tabs {
    background: $panel;
    color: $text-muted;
    padding: 0 1;
    height: 1;
}
#tabs .active { color: $text; text-style: bold reverse; }
#body { height: 1fr; }
TextArea {
    height: 1fr;
    border: none;
}
"""


class ReasoningInspector(App):
    CSS = DEFAULT_CSS
    BINDINGS = [
        Binding("]", "next_record", "Rec+", priority=True),
        Binding("[", "previous_record", "Rec-", priority=True),
        Binding("ctrl+home", "first_record", "First", priority=True),
        Binding("ctrl+end", "last_record", "Last", priority=True),
        Binding("tab", "next_section", "Sec+", priority=True),
        Binding("shift+tab", "previous_section", "Sec-", priority=True),
        Binding("ctrl+d", "page_down", "PgDn", priority=True),
        Binding("ctrl+u", "page_up", "PgUp", priority=True),
        Binding("q", "quit", "Quit", priority=True),
    ]

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        self.records = _load_jsonl(path)
        self._i_rec = 0
        self._i_sec = 0
        self._sections: list[tuple[str, str]] = []
        self._meta: dict = {}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="meta")
        yield Static("", id="tabs")
        with Vertical(id="body"):
            yield TextArea.code_editor(
                "", read_only=True, show_line_numbers=False, id="viewer"
            )
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_record()

    def _refresh_record(self) -> None:
        if not self.records:
            self.query_one("#meta", Static).update("(no records)")
            self.query_one("#tabs", Static).update("")
            self.query_one("#viewer", TextArea).text = ""
            self.title = f"Reasoning Inspector — {self.path.name}"
            self.sub_title = "0/0"
            return
        rec = self.records[self._i_rec]
        self._sections, self._meta = _record_to_sections(rec)
        self._i_sec = 0
        self._refresh_section()

    def _refresh_section(self) -> None:
        meta_str = _format_meta(self._meta)
        self.query_one("#meta", Static).update(meta_str or " ")

        tab_parts = []
        for i, (label, text) in enumerate(self._sections):
            marker = f"[{i}] {label} ({len(text)})"
            if i == self._i_sec:
                marker = f"[reverse bold]{marker}[/]"
            tab_parts.append(marker)
        self.query_one("#tabs", Static).update("  ".join(tab_parts))

        viewer = self.query_one("#viewer", TextArea)
        if self._sections:
            label, text = self._sections[self._i_sec]
            viewer.language = _guess_language(label, text)
            viewer.text = (text or "").replace("\x00", "")
            viewer.scroll_home(animate=False)
        else:
            viewer.text = "(empty record)"

        self.title = f"Reasoning Inspector — {self.path.name}"
        self.sub_title = (
            f"Record {self._i_rec + 1}/{len(self.records)}  "
            f"Section {self._i_sec + 1}/{len(self._sections) or 1}"
        )

    def _set_record(self, value: int) -> None:
        if not self.records:
            return
        value = max(0, min(value, len(self.records) - 1))
        if value != self._i_rec:
            self._i_rec = value
            self._refresh_record()

    def _set_section(self, value: int) -> None:
        if not self._sections:
            return
        value = max(0, min(value, len(self._sections) - 1))
        if value != self._i_sec:
            self._i_sec = value
            self._refresh_section()

    def action_next_record(self) -> None: self._set_record(self._i_rec + 1)
    def action_previous_record(self) -> None: self._set_record(self._i_rec - 1)
    def action_first_record(self) -> None: self._set_record(0)
    def action_last_record(self) -> None: self._set_record(len(self.records) - 1)
    def action_next_section(self) -> None: self._set_section(self._i_sec + 1)
    def action_previous_section(self) -> None: self._set_section(self._i_sec - 1)

    def action_page_down(self) -> None:
        self.query_one("#viewer", TextArea).action_cursor_page_down()

    def action_page_up(self) -> None:
        self.query_one("#viewer", TextArea).action_cursor_page_up()


def _guess_language(label: str, text: str) -> str | None:
    L = label.upper()
    if "KERNEL" in L or L == "ASSISTANT":
        if "```cpp" in text or "#include" in text or "__global__" in text:
            return "cpp"
        if "@triton.jit" in text:
            return "python"
    if L == "USER" or L == "INPUT" or L == "PROMPT" or L == "TASK":
        return "markdown"
    return None


@app.command(help=__doc__)
def main(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, file_okay=True,
                                help="Path to a JSONL file"),
) -> None:
    ReasoningInspector(path).run()


if __name__ == "__main__":
    app()
