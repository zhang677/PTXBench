"""Utilities for reverse-CoT distillation.

Contains: trajectory extraction, turn selectors, context builders,
reasoning prompt, SFT formatter, experiment config, quality filter.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from accrl.utils.code_utils import extract_code_block

# ---------------------------------------------------------------------------
# Trajectory extraction
# ---------------------------------------------------------------------------

_SPEEDUP_RE = re.compile(r"speedup:\s*([\d.]+)x")
_NAME_RE = re.compile(r'"name":\s*"([^"]+)"')


def _classify_failure(feedback: str) -> str | None:
    if not feedback:
        return None
    if "Could not extract" in feedback:
        return "no_code"
    if "error:" in feedback.lower() and ("returncode" in feedback or "compile" in feedback.lower()):
        return "compile"
    if "TIMEOUT" in feedback:
        return "timeout"
    if "INCORRECT" in feedback:
        return "incorrect"
    if "FAILED" in feedback:
        return "runtime"
    return None


def extract_turns(traj: dict) -> list[dict]:
    """Parse a trajectory into per-turn records."""
    messages = traj.get("messages", [])
    model = traj.get("info", {}).get("config", {}).get("model", {}).get("model_name", "unknown")

    system_prompt = ""
    task_prompt = ""
    definition_name = "unknown"

    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        elif m["role"] == "user" and not task_prompt:
            task_prompt = m["content"]
            match = _NAME_RE.search(task_prompt)
            if match:
                definition_name = match.group(1)
            break

    turns = []
    prev_feedback = None
    prev_kernel_code = None
    prev_speedup = None
    best_speedup = 0.0

    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue

        content = m.get("content") or ""
        kernel_code = extract_code_block(content, languages=["cpp"], keep_separators=False)

        feedback_content = ""
        speedup = None
        for j in range(i + 1, len(messages)):
            if messages[j]["role"] == "user":
                feedback_content = messages[j]["content"]
                sp_match = _SPEEDUP_RE.search(feedback_content)
                if sp_match:
                    speedup = float(sp_match.group(1))
                break

        passed = speedup is not None
        improved = passed and speedup > best_speedup
        if passed and speedup > best_speedup:
            best_speedup = speedup

        turns.append({
            "turn": len(turns),
            "definition_name": definition_name,
            "task_prompt": task_prompt,
            "system_prompt": system_prompt,
            "kernel_code": kernel_code,
            "prev_kernel_code": prev_kernel_code,
            "prev_feedback": prev_feedback,
            "prev_speedup": prev_speedup,
            "speedup": speedup,
            "passed": passed,
            "improved": improved,
            "failure_type": None if passed else _classify_failure(feedback_content),
            "model": model,
            "raw_assistant_content": content,
            "raw_feedback_content": feedback_content,
        })

        prev_feedback = feedback_content
        prev_kernel_code = kernel_code
        if passed:
            prev_speedup = speedup

    return turns


def extract_eval_dir(eval_dir: Path) -> list[dict]:
    """Extract turns from all trajectories in an eval directory."""
    traj_dir = eval_dir / "trajectories"
    if not traj_dir.exists():
        return []

    all_turns = []
    for traj_file in sorted(traj_dir.glob("*.json")):
        try:
            traj = json.loads(traj_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        turns = extract_turns(traj)
        for t in turns:
            t["run_id"] = eval_dir.name
            t["exp_id"] = traj_file.stem
        all_turns.extend(turns)
    return all_turns


# ---------------------------------------------------------------------------
# Selectors — which turns to process
# ---------------------------------------------------------------------------

def select_passing(turns: list[dict]) -> list[dict]:
    return [t for t in turns if t["passed"]]


def select_best(turns: list[dict]) -> list[dict]:
    passing = [t for t in turns if t["passed"] and t["speedup"] is not None]
    return [max(passing, key=lambda t: t["speedup"])] if passing else []


def select_all_with_history(turns: list[dict]) -> list[dict]:
    """Every turn, annotated with full trajectory history up to that point."""
    result = []
    for i, t in enumerate(turns):
        t_copy = {**t, "history": turns[:i]}
        result.append(t_copy)
    return result


SELECTORS = {
    "passing_only": select_passing,
    "best_per_trajectory": select_best,
    "all_with_history": select_all_with_history,
}


# ---------------------------------------------------------------------------
# Context builder — what to send to the reasoning model
# ---------------------------------------------------------------------------

def build_context(turn: dict) -> str:
    """Full original prompt (system + task) + expert kernel + result."""
    parts = [
        f"## Reference Manual\n{turn['system_prompt']}",
        f"## Task\n{turn['task_prompt']}",
        f"## Expert's Kernel\n```cpp\n{turn['kernel_code']}\n```",
    ]
    if turn.get("speedup") is not None:
        parts.append(f"## Result\nThis kernel achieved {turn['speedup']:.3f}x speedup.")
    return "\n\n".join(parts)


def build_trajectory_context(turn: dict) -> str:
    """Full manual + task + all previous turns (kernel + verdict) + current kernel.

    For turn N, shows:
      - Reference manual + task
      - Turn 0: kernel_0 + verdict_0
      - Turn 1: kernel_1 + verdict_1
      - ...
      - Turn N: kernel_N + verdict_N (current)
    """
    parts = [
        f"## Reference Manual\n{turn['system_prompt']}",
        f"## Task\n{turn['task_prompt']}",
    ]

    # Add history: previous turns with their kernels and verdicts
    history = turn.get("history", [])
    for h in history:
        label = f"### Turn {h['turn']}"
        if h.get("kernel_code"):
            parts.append(f"{label} — Kernel\n```cpp\n{h['kernel_code']}\n```")
        else:
            parts.append(f"{label} — (no code extracted)")
        if h.get("raw_feedback_content"):
            parts.append(f"### Turn {h['turn']} — Verdict\n{h['raw_feedback_content']}")

    # Current turn — no verdict (reasoning should be BEFORE seeing the result)
    parts.append(f"### Turn {turn['turn']} — Kernel (CURRENT)\n```cpp\n{turn['kernel_code']}\n```")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Reasoning prompt
# ---------------------------------------------------------------------------

REASONING_PROMPTS = {
    "v1": """## Your Task

You are generating **reasoning tokens** — the internal chain-of-thought that an expert CUDA kernel developer would think through BEFORE writing the kernel shown above.

Your output will be used as training data to teach another model how to reason about kernel optimization. The other model will see the task and must produce reasoning followed by code.

Generate the reasoning as a single <my_reasoning>...</my_reasoning> block. The reasoning should:

1. **Analyze the problem**: dimensions, memory layout, data types, compute-to-memory ratio
2. **Design the tiling strategy**: choose specific tile sizes (e.g., 128x256) and explain WHY — relate to warp size, shared memory capacity, register pressure
3. **Plan memory access**: how to load A and B tiles, coalescing, bank conflicts, any async copy (cp.async, TMA)
4. **Plan compute**: accumulation in registers, use of tensor cores / WMMA / WGMMA if applicable, instruction-level parallelism
5. **Arrive at the implementation**: connect each reasoning step to specific code patterns in the kernel above

Be concrete. Reference actual numbers from the kernel (tile sizes, thread counts, loop bounds). Do NOT just describe what the code does — explain WHY each design choice was made.

Output your reasoning in a <my_reasoning>...</my_reasoning> block. You may include code snippets or pseudocode within your reasoning if it helps illustrate your thought process.""",

    "v2_deep": """## Your Task

You are generating **reasoning tokens** at MAXIMUM DEPTH AND EFFORT.

An expert CUDA kernel developer wrote the kernel shown above. Your job is to reconstruct the COMPLETE internal reasoning they would have gone through — every decision point, every alternative considered and rejected, every calculation.

Use the HIGHEST reasoning effort possible. This is training data for another model — the more thorough and detailed your reasoning, the better that model will learn.

Generate the reasoning as a single <my_reasoning>...</my_reasoning> block. Cover ALL of the following in depth:

### 1. Problem Analysis (be exhaustive)
- Exact dimensions, data types, memory footprint of each tensor
- Compute intensity: total FLOPs vs total bytes moved, arithmetic intensity
- What makes this problem hard? What's the theoretical peak?

### 2. Hardware Constraints (specific to the target GPU)
- Shared memory per SM, max registers per thread, warp size
- Available compute units (tensor cores, CUDA cores)
- Memory bandwidth (HBM, L2, shared memory)
- How do these constraints bound the achievable performance?

### 3. Tiling Strategy (justify EVERY number)
- Why this M_TILE, N_TILE, K_TILE and not alternatives?
- What would happen with 2x larger tiles? 2x smaller?
- How do tile sizes relate to warp/warpgroup structure?
- How many CTAs, how does this map to SMs?

### 4. Memory Access Design (trace the data flow)
- How is each tile loaded? TMA vs cp.async vs regular loads?
- Shared memory layout — any swizzling or padding for bank conflicts?
- Pipeline stages — how many, why that number?
- Prefetching strategy — how far ahead?

### 5. Compute Design (instruction-level detail)
- Which compute instruction (HMMA, WGMMA, plain FMA)?
- Register allocation — how many accumulators, what precision?
- Warp specialization — producer/consumer split?
- Instruction-level parallelism — overlapping compute and memory?

### 6. Edge Cases and Correctness
- How are partial tiles handled (M not divisible by tile size)?
- How is numerical precision maintained with bfloat16?
- Any synchronization barriers and why they're placed where they are?

### 7. Performance Analysis
- Estimated occupancy and why
- Expected bottleneck (compute-bound or memory-bound?)
- What's left on the table — what would the next optimization be?

Be EXTREMELY concrete. Every claim must reference a specific line, constant, or pattern from the kernel code. Show your calculations (e.g., "shared memory usage: 128 * 64 * 2 bytes = 16KB per stage, 3 stages = 48KB").

Output your reasoning in a <my_reasoning>...</my_reasoning> block. You may include code snippets or pseudocode within your reasoning if it helps illustrate your thought process.""",
    "v3_trajectory": """## Your Task

Above you can see the full history of an expert CUDA developer ("the Expert") solving a kernel optimization problem: the reference manual, the task, and every previous kernel attempt with its evaluation verdict. The CURRENT turn's kernel is what the Expert wrote next.

Your job: WRITE the reasoning trace as if you ARE the Expert, in first person, BEFORE you wrote the CURRENT kernel. This is the internal monologue the Expert would have gone through — what they noticed, what they considered, what they decided, and why.

This trace will be used as training data, so it must be detailed and authentic.

### Hard requirements for the output:
- **First person** — "I notice the previous kernel had X. I should try Y because..."
- **VERY LONG** — target 10,000 to 50,000 characters. Capture the Expert's full thought process, not a summary. Be exhaustive.
- **Concrete** — reference specific tile sizes, specific error messages, specific numbers from the kernel and verdicts above
- **Forward-flowing** — start from the problem/feedback, show every step of the decision chain, end with the strategy about to be implemented
- **Include alternatives considered and rejected** — "I could try A, but that would cause B, so instead I'll do C"
- **Show calculations** — e.g. "shared memory: 128*64*2 = 16KB per stage, 3 stages = 48KB, fits in 228KB SMEM"
- **Reference the reference manual** — quote or paraphrase relevant sections when justifying decisions
- **Walk through small code patterns inline** if it helps illustrate a thought

### Format requirements (CRITICAL):
- Output ONLY the <my_reasoning>...</my_reasoning> block, nothing before, nothing after
- Feel free to write kernel code snipet in the thinking process just as an expert put some scratch on the paper
- DO NOT add a summary or conclusion outside the block

For Turn 0 (first attempt): walk through problem analysis → tile selection → memory hierarchy → compute strategy → final design, with rich justification at each step.
For Turn N > 0 (iteration): walk through what went wrong → diagnosis from verdict → considered fixes → chosen strategy → implementation choices.

Begin now with `<my_reasoning>` and end with `</my_reasoning>`.""",
}


def make_prompt(context: str, prompt_version: str = "v1") -> str:
    if prompt_version not in REASONING_PROMPTS:
        raise ValueError(f"Unknown prompt version: {prompt_version}. Available: {list(REASONING_PROMPTS.keys())}")
    return context + "\n\n" + REASONING_PROMPTS[prompt_version]


# ---------------------------------------------------------------------------
# SFT formatter
# ---------------------------------------------------------------------------

SFT_SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are an expert CUDA kernel developer. Given a kernel optimization task, "
        "think step-by-step about the optimization strategy, then provide your "
        "implementation in a single ```cpp code block."
    ),
}


def format_sft(reasoning: str, kernel_code: str, turn: dict) -> list[dict]:
    """Format as chat messages with <think> tags for SFT training."""
    user_content = turn["task_prompt"]
    if turn["turn"] > 0 and turn.get("prev_feedback"):
        user_content += "\n\n## Previous Evaluation\n" + turn["prev_feedback"]

    return [
        SFT_SYSTEM_MSG,
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>\n\n```cpp\n{kernel_code}\n```"},
    ]


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

def passes_quality(reasoning: str, kernel_code: str, min_len: int = 200, max_len: int = 50000) -> bool:
    if len(reasoning) < min_len or len(reasoning) > max_len:
        return False
    kernel_nums = set(re.findall(r"\b\d{2,}\b", kernel_code))
    return any(n in reasoning for n in kernel_nums if len(n) >= 2)


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str
    description: str = ""
    selector: str = "best_per_trajectory"
    reasoning_model: str = "MiniMax-M2.7"
    prompt_version: str = "v1"
    min_speedup: float = 0.3
    max_tokens: int = 196000
    min_reasoning_len: int = 200
    max_reasoning_len: int = 190000
    enable_thinking: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        return cls(**yaml.safe_load(Path(path).read_text()))

    def to_yaml(self, path: str | Path) -> None:
        d = {
            "name": self.name,
            "description": self.description,
            "selector": self.selector,
            "reasoning_model": self.reasoning_model,
            "prompt_version": self.prompt_version,
            "min_speedup": self.min_speedup,
            "max_tokens": self.max_tokens,
            "min_reasoning_len": self.min_reasoning_len,
            "max_reasoning_len": self.max_reasoning_len,
            "enable_thinking": self.enable_thinking,
        }
        Path(path).write_text(yaml.dump(d, default_flow_style=False, sort_keys=False))
