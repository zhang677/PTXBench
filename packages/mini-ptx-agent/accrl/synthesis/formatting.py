"""Shared prompt-formatting helpers for kernel tasks.

Provides functions to turn Definition / Workload objects into human-readable
strings and to load language-specific system prompts.  Used by both the
training-data generator and the agent service.
"""

import json
from pathlib import Path

from flashinfer_bench.data import Definition, Workload, AxisConst

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def format_definition(defn: Definition) -> str:
    """Format a Definition into a human-readable spec string."""
    axes_str = "\nAxes:\n"
    for name, axis in defn.axes.items():
        if isinstance(axis, AxisConst):
            axes_str += f"  {name}: constant = {axis.value}"
        else:
            axes_str += f"  {name}: variable"
        if axis.description:
            axes_str += f" ({axis.description})"
        axes_str += "\n"

    inputs_str = "\nInputs:\n"
    for name, spec in defn.inputs.items():
        shape_str = "scalar" if spec.shape is None else f"[{', '.join(spec.shape)}]"
        inputs_str += f"  {name}: {shape_str} ({spec.dtype})"
        if spec.description:
            inputs_str += f" - {spec.description}"
        inputs_str += "\n"

    outputs_str = "\nOutputs:\n"
    for name, spec in defn.outputs.items():
        shape_str = "scalar" if spec.shape is None else f"[{', '.join(spec.shape)}]"
        outputs_str += f"  {name}: {shape_str} ({spec.dtype})"
        if spec.description:
            outputs_str += f" - {spec.description}"
        outputs_str += "\n"

    constraints_str = ""
    if defn.constraints:
        constraints_str = "\nConstraints:\n"
        for constraint in defn.constraints:
            constraints_str += f"  - {constraint}\n"

    return (
        f"Name: {defn.name}\n"
        f"Type: {defn.op_type}\n"
        f"{axes_str}{inputs_str}{outputs_str}{constraints_str}\n"
        f"Reference Implementation:\n{defn.reference}"
    )


def format_workload(wl: Workload) -> str:
    """Format a Workload into a human-readable string."""
    lines = ["Concrete axis values:"]
    for name, value in wl.axes.items():
        lines.append(f"  {name} = {value}")
    lines.append("Input specs:")
    for name, spec in wl.inputs.items():
        lines.append(f"  {name}: {spec.type}")
    return "\n".join(lines)


def load_system_prompt(language: str) -> str:
    """Load and prepare the system prompt for the given language."""
    if language == "triton":
        system_file = PROMPTS_DIR / "system_triton.txt"
        reference_file = PROMPTS_DIR / "triton_reference.txt"
        system_template = system_file.read_text()
        reference_text = reference_file.read_text()
        return system_template.replace("{triton_reference}", reference_text)
    elif language == "python":
        system_file = PROMPTS_DIR / "system_python.txt"
        return system_file.read_text()
    elif language in ("cuda", "cpp"):
        system_file = PROMPTS_DIR / "system_cuda.txt"
        reference_file = PROMPTS_DIR / "cuda_torch_reference.txt"
        system_template = system_file.read_text()
        reference_text = reference_file.read_text()
        return system_template.replace("{cuda_torch_reference}", reference_text)
    else:
        raise ValueError(f"Unsupported language: {language}")


def make_training_label(
    definition_name: str, workload_uuid: str, language: str
) -> str:
    """Serialise label metadata as a JSON string."""
    return json.dumps(
        {
            "definition_name": definition_name,
            "workload_uuid": workload_uuid,
            "language": language,
        }
    )
