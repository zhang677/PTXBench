"""CPU-only preflight shared by Python-based GPU languages."""
from __future__ import annotations

import ast


def validation_error(source: str, *, language: str = "CuTeDSL") -> str | None:
    """Check Python without importing or executing the submitted module."""
    try:
        tree = ast.parse(source, filename="kernel.py")
        compile(tree, "kernel.py", "exec")
    except (SyntaxError, ValueError) as exc:
        return f"ERROR: Invalid Python in kernel.py: {exc}"
    if not any(isinstance(node, ast.FunctionDef) and node.name == "run" for node in tree.body):
        return "ERROR: kernel.py requires a destination-passing run function."
    # Resolve import aliases for common library shortcuts. This is a preflight,
    # not a sandbox or a proof of the submitted kernel's implementation.
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                aliases[name.asname or name.name.split('.')[0]] = name.name if name.asname else name.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                aliases[name.asname or name.name] = f"{node.module}.{name.name}"

    def qualified(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{qualified(node.value)}.{node.attr}"
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return f"ERROR: Implement GPU computation in {language}; Python matrix multiplication is not allowed."
        if isinstance(node, ast.Call):
            name = qualified(node.func)
            if (name in {"torch.mm", "torch.matmul", "torch.bmm", "torch.addmm", "torch.einsum"}
                    or "cublas" in name.lower() or "cudnn" in name.lower()):
                return f"ERROR: Implement GPU computation in {language}; library shortcut {name} is not allowed."
    return None
