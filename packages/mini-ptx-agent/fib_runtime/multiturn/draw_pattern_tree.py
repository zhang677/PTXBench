#!/usr/bin/env python3
"""Render a CFG pattern tree for a single kernel .cu file as PNG/SVG/PDF.

Reuses the CFG extraction pipeline from ``analyze_pattern`` /
``analyze_patterns_batch`` and draws each node with Graphviz ``dot``.

Usage:
    python draw_pattern_tree.py <kernel.cu> [--output PATH]
        [--format png|svg|pdf] [--no-collapse-seq]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import graphviz

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_pattern import (  # noqa: E402
    CFGNode,
    build_cfg,
    cfg_stats,
    extract_filtered_kernel_text,
    parse_top_level,
    strip_comments,
)


# --------------------------------------------------------------------------- #
# CFG extraction (mirrors analyze_patterns_batch.analyze_one minus hashing).  #
# --------------------------------------------------------------------------- #

def extract_cfg_root(src: str) -> CFGNode:
    """Parse CUDA source → single CFG root. Multi-kernel files are wrapped
    in a top-level SEQ; files with no __global__ func return an empty SEQ."""
    stripped = strip_comments(src)
    items = parse_top_level(stripped)
    roots: list[CFGNode] = []
    for item in items:
        if item.kind != "func":
            continue
        if "__global__" in item.qualifiers:
            roots.append(build_cfg(item.body))
    if len(roots) == 1:
        return roots[0]
    return CFGNode("SEQ", "", tuple(roots))


# --------------------------------------------------------------------------- #
# SEQ collapse: remove pure structural glue without changing semantics.       #
# --------------------------------------------------------------------------- #

def collapse_seq(node: CFGNode) -> CFGNode:
    """Simplify SEQ-under-SEQ (splice up) and single-child SEQ (pass-through).

    Applied recursively bottom-up. Root is always kept: if root becomes
    degenerate, it's left as a SEQ wrapper so the output has something to draw.
    """
    kids = tuple(collapse_seq(c) for c in node.children)

    # Splice SEQ children up into a SEQ parent.
    if node.kind == "SEQ":
        flat: list[CFGNode] = []
        for ch in kids:
            if ch.kind == "SEQ":
                flat.extend(ch.children)
            else:
                flat.append(ch)
        kids = tuple(flat)
        # A SEQ with exactly one non-SEQ child becomes that child.
        if len(kids) == 1 and kids[0].kind != "SEQ":
            return kids[0]

    return CFGNode(node.kind, node.label, kids)


# --------------------------------------------------------------------------- #
# Rendering.                                                                  #
# --------------------------------------------------------------------------- #

_PALETTE = {
    "SEQ":     "#eeeeee",
    "IF":      "#ffb347",
    "FOR":     "#7fb3d5",
    "WHILE":   "#7fb3d5",
    "DO":      "#7fb3d5",
    "SWITCH":  "#c39bd3",
    "CASE":    "#c39bd3",
    "DEFAULT": "#c39bd3",
    "CALL":    "#82e0aa",
    "JMP":     "#f1948a",
    "ASM":     "#d5d8dc",
}

_LABEL_MAX = 32


def _node_label(n: CFGNode) -> str:
    kind = n.kind
    if kind == "CALL":
        lab = n.label or "?"
        if len(lab) > _LABEL_MAX:
            lab = lab[: _LABEL_MAX - 1] + "…"
        return f"call {lab}"
    if kind == "JMP":
        return n.label or "jmp"
    if kind == "ASM":
        return kind.lower()
    return kind.lower()


@dataclass
class _Builder:
    dot: graphviz.Digraph
    counter: int = 0

    def add(self, node: CFGNode) -> str:
        self.counter += 1
        nid = f"n{self.counter}"
        self.dot.node(
            nid,
            label=_node_label(node),
            fillcolor=_PALETTE.get(node.kind, "#ffffff"),
        )
        for ch in node.children:
            cid = self.add(ch)
            self.dot.edge(nid, cid)
        return nid


def _build_digraph(root: CFGNode, title: str) -> graphviz.Digraph:
    dot = graphviz.Digraph(
        "cfg",
        graph_attr={
            "rankdir": "TB",
            # "spline" (default) not "ortho": ortho triggers dot's maze.c
            # assertion on larger CFGs; splines render cleanly everywhere.
            "splines": "spline",
            "nodesep": "0.25",
            "ranksep": "0.35",
            "label": title,
            "labelloc": "t",
            "fontsize": "11",
            "fontname": "Helvetica",
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fontsize": "10",
            "fontname": "Helvetica",
            "margin": "0.05,0.03",
        },
        edge_attr={"arrowsize": "0.6"},
    )
    builder = _Builder(dot)
    builder.add(root)
    return dot


# --------------------------------------------------------------------------- #
# Public API.                                                                 #
# --------------------------------------------------------------------------- #

def render_kernel_tree(
    kernel_path: Path,
    output_path: Path,
    *,
    collapse: bool = True,
    fmt: str = "png",
    emit_source: bool = True,
) -> dict:
    """Parse `kernel_path`, render CFG tree, write image to `output_path`.

    `output_path` should include a suffix; graphviz adds one matching `fmt` if
    missing. When `emit_source` is true, also writes the filtered kernel text
    to `<stem>.cu` alongside the image (same source that build_cfg walked).
    Returns a dict with cfg_stats plus the resolved image and source paths.
    """
    src = Path(kernel_path).read_text()
    root = extract_cfg_root(src)

    if not root.children and root.kind == "SEQ":
        # File has no __global__ kernel — emit a placeholder.
        root = CFGNode("SEQ", "", (CFGNode("CALL", "<no __global__ kernel>", ()),))

    draw_root = collapse_seq(root) if collapse else root
    stats = cfg_stats(draw_root)

    kernel_id = f"{kernel_path.parent.name}/{kernel_path.name}"
    title = (
        f"{kernel_id}  nodes={stats['node_count']}  depth={stats['depth']}  "
        f"loops={stats['loops']}  branches={stats['branches']}"
    )
    dot = _build_digraph(draw_root, title)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # graphviz.render wants a filename stem (no suffix). Strip the format
    # suffix if the caller included it.
    stem = output_path
    if stem.suffix.lower().lstrip(".") == fmt.lower():
        stem = stem.with_suffix("")
    rendered = dot.render(
        filename=str(stem), format=fmt, cleanup=True,
    )
    source_out: str | None = None
    if emit_source:
        source_path = stem.with_suffix(".cu")
        source_path.write_text(extract_filtered_kernel_text(src))
        source_out = str(source_path)
    return {**stats, "image": rendered, "source": source_out, "kernel_id": kernel_id}


# --------------------------------------------------------------------------- #
# CLI.                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0] if __doc__ else "")
    ap.add_argument("kernel", type=Path, help="Path to kernel_*.cu")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output image path (default: <stem>.<fmt> in CWD)")
    ap.add_argument("--format", default="png", choices=("png", "svg", "pdf"))
    ap.add_argument("--no-collapse-seq", action="store_true",
                    help="Keep SEQ glue nodes in the drawing")
    args = ap.parse_args()

    kernel: Path = args.kernel.resolve()
    if not kernel.is_file():
        sys.exit(f"kernel not found: {kernel}")

    out: Path = args.output or Path(kernel.stem + "." + args.format)
    result = render_kernel_tree(
        kernel, out,
        collapse=not args.no_collapse_seq,
        fmt=args.format,
    )
    print(
        f"[draw_pattern_tree] {result['kernel_id']} → {result['image']} "
        f"(nodes={result['node_count']} depth={result['depth']})"
    )


if __name__ == "__main__":
    main()
