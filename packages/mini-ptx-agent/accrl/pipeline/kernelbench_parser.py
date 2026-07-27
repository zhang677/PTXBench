"""Parse KernelBench problem files into structured data.

Each KernelBench file has:
- Model class with __init__ and forward()
- Global variables (shape parameters)
- get_inputs() -> list of input tensors
- get_init_inputs() -> list of Model constructor args

Usage:
    from accrl.pipeline.kernelbench_parser import parse_problem, list_problems

    problems = list_problems("/path/to/KernelBench/KernelBench")
    parsed = parse_problem(problems[0])
"""

import ast
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class KernelBenchProblem:
    """Parsed representation of a KernelBench problem."""

    path: Path
    level: int
    problem_id: int
    name: str
    code: str
    # Extracted from the file
    model_class_source: str = ""
    forward_source: str = ""
    init_source: str = ""
    docstring: str = ""
    global_vars: dict = field(default_factory=dict)
    get_inputs_source: str = ""
    get_init_inputs_source: str = ""


def list_problems(kernelbench_root: str | Path) -> list[Path]:
    """List all KernelBench problem files across all levels."""
    root = Path(kernelbench_root)
    problems = []
    for level_dir in sorted(root.iterdir()):
        if level_dir.is_dir() and level_dir.name.startswith("level"):
            for f in sorted(level_dir.glob("*.py")):
                problems.append(f)
    return problems


def _extract_level(path: Path) -> int:
    """Extract level number from path like .../level1/40_LayerNorm.py"""
    for part in path.parts:
        if part.startswith("level"):
            return int(part.replace("level", ""))
    return 0


def _extract_problem_id(path: Path) -> int:
    """Extract problem ID from filename like 40_LayerNorm.py"""
    stem = path.stem
    parts = stem.split("_", 1)
    try:
        return int(parts[0])
    except ValueError:
        return 0


def _extract_name(path: Path) -> str:
    """Extract human-readable name from filename like 40_LayerNorm.py"""
    stem = path.stem
    parts = stem.split("_", 1)
    if len(parts) > 1:
        return parts[1].replace("_", " ").strip()
    return stem


def _get_node_source(code: str, node: ast.AST) -> str:
    """Extract source code for an AST node."""
    lines = code.splitlines()
    start = node.lineno - 1
    end = node.end_lineno
    return "\n".join(lines[start:end])


def _extract_global_vars(tree: ast.Module, code: str) -> dict:
    """Extract top-level variable assignments (shape parameters)."""
    global_vars = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        val = ast.literal_eval(node.value)
                        global_vars[target.id] = val
                    except (ValueError, SyntaxError):
                        # Store source for non-literal values
                        global_vars[target.id] = _get_node_source(code, node.value)
    return global_vars


def parse_problem(path: str | Path) -> KernelBenchProblem:
    """Parse a single KernelBench problem file."""
    path = Path(path)
    code = path.read_text()
    tree = ast.parse(code)

    problem = KernelBenchProblem(
        path=path,
        level=_extract_level(path),
        problem_id=_extract_problem_id(path),
        name=_extract_name(path),
        code=code,
    )

    # Extract global variables
    problem.global_vars = _extract_global_vars(tree, code)

    # Find Model class and functions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            problem.model_class_source = _get_node_source(code, node)
            # Get docstring
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                problem.docstring = node.body[0].value.value.strip()
            # Find __init__ and forward
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    if method.name == "__init__":
                        problem.init_source = _get_node_source(code, method)
                    elif method.name == "forward":
                        problem.forward_source = _get_node_source(code, method)

        elif isinstance(node, ast.FunctionDef):
            if node.name == "get_inputs":
                problem.get_inputs_source = _get_node_source(code, node)
            elif node.name == "get_init_inputs":
                problem.get_init_inputs_source = _get_node_source(code, node)

    return problem


def parse_all(kernelbench_root: str | Path) -> list[KernelBenchProblem]:
    """Parse all KernelBench problems."""
    paths = list_problems(kernelbench_root)
    problems = []
    for p in paths:
        try:
            problems.append(parse_problem(p))
        except Exception as e:
            logger.warning("Failed to parse %s: %s", p, e)
    return problems
