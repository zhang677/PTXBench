"""Multi-turn generate and reward functions for kernel optimisation via agent-env Docker.

Extends the single-turn agent-env pattern with iterative refinement:
each turn the model thinks (Te) + responds (Re), code is extracted and
evaluated, then profile feedback is fed back as a user message for the
next turn.

Turn pattern (T=thinking, R=response, e=execute):
  Turn 1: Te → Re → Profile
  Turn 2: (feedback) → Te → Re → Profile
  ...until max_turns

The agent-env container runs the multi-turn loop when
``sampling_params["context_management"] == "basic_multiturn"``.  All
assistant tokens (thinking + response) get loss_mask=1; feedback user
messages get loss_mask=0.

Loaded via:
    --custom-generate-function-path  multiturn_with_agent_env.generate
    --custom-rm-path                 multiturn_with_agent_env.reward_func
    --dynamic-sampling-filter-path   multiturn_with_agent_env.dynamic_filter
    --rollout-function-path          multiturn_with_agent_env.generate_rollout
"""

import logging
import os
from argparse import Namespace
from collections.abc import Callable
from typing import Any

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.sglang_rollout import GenerateState, eval_rollout
from miles.utils.async_utils import run
from miles.utils.http_utils import post
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# Module-level DB client, initialized lazily on first use.
_db = None


def _get_db():
    global _db
    if _db is None:
        from fib_runtime.db_client import KernelDB

        _db = KernelDB()
    return _db


# ---------------------------------------------------------------------------
# Token / loss-mask builder (same approach as SWE-agent)
# ---------------------------------------------------------------------------


def build_tokens_and_mask_from_messages(
    messages: list[dict],
    tokenizer,
) -> tuple[list[int], list[int], str, int]:
    """Build token IDs and loss mask from conversation messages.

    First two messages are treated as prompt (loss_mask = 0).
    Remaining messages: assistant → loss_mask = 1, others → 0.

    This naturally handles multi-turn conversations:
    - system + user prompt: mask=0
    - assistant turn 1 (including <think> blocks): mask=1
    - user feedback: mask=0
    - assistant turn 2: mask=1
    - ...
    """
    if not messages or len(messages) < 2:
        return [], [], "", 0

    prompt_msgs = messages[:2]
    response_msgs = messages[2:]

    prompt_tokens: list[int] = []
    for msg in prompt_msgs:
        content = msg.get("content", "")
        if content:
            prompt_tokens.extend(
                tokenizer(content, add_special_tokens=False)["input_ids"]
            )

    response_tokens: list[int] = []
    loss_mask: list[int] = []
    response_text_parts: list[str] = []

    for msg in response_msgs:
        content = msg.get("content", "")
        if not content:
            continue

        tokens = tokenizer(content, add_special_tokens=False)["input_ids"]
        response_tokens.extend(tokens)
        response_text_parts.append(content)

        mask_val = 1 if msg.get("role") == "assistant" else 0
        loss_mask.extend([mask_val] * len(tokens))

    all_tokens = prompt_tokens + response_tokens
    response_text = "".join(response_text_parts)
    response_length = len(response_tokens)

    return all_tokens, loss_mask, response_text, response_length


# ---------------------------------------------------------------------------
# Generate function
# ---------------------------------------------------------------------------


async def generate(
    args: Namespace, sample: Sample, sampling_params: dict[str, Any]
) -> Sample:
    """Send multi-turn task to agent-env Docker, receive trajectory + profiles.

    Follows the SWE-agent Gym pattern with multi-turn extensions:
    POST {AGENT_ENV_URL}/run with metadata + sglang_url + sampling_params
    where sampling_params["context_management"] = "basic_multiturn".

    The slim dataset only stores identifiers (definition_name, workload_uuids).
    Full Definition/Workload objects are resolved from the kernel database here.
    """
    db = _get_db()

    definition_name = sample.metadata["definition_name"]
    definition = db.get_definition(definition_name)

    workload_uuids: list[str] = sample.metadata["workload_uuids"]
    workloads = [db.get_workload(uuid) for uuid in workload_uuids]

    # Inject multi-turn context management into sampling_params
    sampling_params = dict(sampling_params)  # copy to avoid mutating caller's dict
    context_management = getattr(args, "context_management", None)
    if context_management is None:
        context_management = sample.metadata.get("context_management", "basic_multiturn")
    sampling_params["context_management"] = context_management

    # max_turns from args or sample metadata, default 3
    max_turns = getattr(args, "max_turns", None)
    if max_turns is None:
        max_turns = sample.metadata.get("max_turns", 3)

    request = {
        "definition": definition.model_dump(mode="json"),
        "workloads": [w.model_dump(mode="json") for w in workloads],
        "language": sample.metadata.get("language", "triton"),
        "definition_name": definition_name,
        "sglang_url": (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/v1"
        ),
        "sampling_params": sampling_params,
        "max_turns": max_turns,
    }

    agent_env_url = os.getenv("AGENT_ENV_URL", "http://fib_env:11000")
    response = await post(f"{agent_env_url}/run", request)

    messages = response.get("messages", [])
    exit_status = response.get("info", {}).get("exit_status", "")
    reward = response.get("reward", 0.0)

    num_assistant = sum(
        1 for m in messages if m.get("role") == "assistant" and m.get("content")
    )
    logger.info(
        "agent-env response: exit_status=%s, reward=%.4f, messages=%d (assistant_with_content=%d)",
        exit_status, reward, len(messages), num_assistant,
    )
    if num_assistant == 0 and len(messages) > 2:
        logger.warning(
            "All assistant messages are empty — sglang may be unreachable from agent-env. "
            "sglang_url=%s, agent_env_url=%s",
            request["sglang_url"], agent_env_url,
        )

    if len(messages) >= 2:
        sample.prompt = messages[:2]

    state = GenerateState(args)
    tokens, loss_mask, response_text, response_length = (
        build_tokens_and_mask_from_messages(
            messages=messages,
            tokenizer=state.tokenizer,
        )
    )

    sample.rollout_log_probs = None  # Log probs unavailable in agent-env pattern
    sample.tokens = tokens
    sample.loss_mask = loss_mask
    sample.response = response_text
    sample.response_length = response_length
    sample.metadata["reward"] = response.get("reward", 0.0)
    sample.metadata["workload_results"] = response.get("workload_results", [])
    sample.metadata["messages"] = messages

    agent_metrics = response.get("info", {}).get("agent_metrics", {})
    sample.metadata["agent_metrics"] = agent_metrics

    if exit_status == "Submitted":
        sample.status = Sample.Status.COMPLETED
    elif exit_status in ("Truncated",):
        sample.status = Sample.Status.TRUNCATED
    else:
        sample.status = Sample.Status.ABORTED
        sample.reward = 0.0

    return sample


# ---------------------------------------------------------------------------
# Reward function (pre-computed by agent env)
# ---------------------------------------------------------------------------


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Reward already computed by agent-env Docker during generate()."""
    return sample.metadata.get("reward", 0.0)


# ---------------------------------------------------------------------------
# Dynamic filter
# ---------------------------------------------------------------------------


def dynamic_filter(
    args, samples: list[Sample], **kwargs
) -> DynamicFilterOutput:
    """Filter out groups with any aborted samples from training."""
    has_aborted = any(s.status == Sample.Status.ABORTED for s in samples)
    if has_aborted:
        return DynamicFilterOutput(keep=False, reason="group_has_aborted")
    return DynamicFilterOutput(keep=True)


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def aggregate_agent_metrics(samples: list[Sample]) -> dict:
    """Aggregate kernel optimisation metrics across samples for logging.

    Extended from single-turn to also track multi-turn progression:
    - agent/max_turns: configured max turns
    - agent/per_turn_latency_*: per-turn latency progression stats
    """
    metrics: dict[str, float] = {}

    all_results: list[dict] = []
    for sample in samples:
        if hasattr(sample, "metadata") and sample.metadata:
            wl_results = sample.metadata.get("workload_results", [])
            all_results.extend(wl_results)

    if not all_results:
        return {}

    total = len(all_results)
    passed = sum(1 for r in all_results if r.get("eval_status") == "PASSED")
    latencies = [
        r.get("latency", 0.0)
        for r in all_results
        if r.get("eval_status") == "PASSED"
    ]

    metrics["kernel/pass_rate"] = passed / total if total > 0 else 0.0
    metrics["kernel/num_evaluated"] = float(total)
    metrics["kernel/num_passed"] = float(passed)
    if latencies:
        metrics["kernel/mean_latency"] = sum(latencies) / len(latencies)
        metrics["kernel/max_latency"] = max(latencies)
        metrics["kernel/min_latency"] = min(latencies)

    # Agent timing and turn metrics
    timing_values: list[float] = []
    turn_values: list[float] = []
    max_turns_values: list[float] = []
    all_per_turn_latencies: list[list[float]] = []

    for sample in samples:
        agent_m = sample.metadata.get("agent_metrics", {})
        if "agent_run_time" in agent_m:
            timing_values.append(agent_m["agent_run_time"])
        if "turns" in agent_m:
            turn_values.append(agent_m["turns"])
        if "max_turns" in agent_m:
            max_turns_values.append(agent_m["max_turns"])
        if "per_turn_latencies" in agent_m:
            all_per_turn_latencies.append(agent_m["per_turn_latencies"])

    if timing_values:
        metrics["agent/run_time_mean"] = sum(timing_values) / len(timing_values)
        metrics["agent/run_time_max"] = max(timing_values)
    if turn_values:
        metrics["agent/turns_mean"] = sum(turn_values) / len(turn_values)
    if max_turns_values:
        metrics["agent/max_turns"] = max(max_turns_values)

    # Per-turn latency progression: for each turn index, average across samples
    if all_per_turn_latencies:
        max_len = max(len(pts) for pts in all_per_turn_latencies)
        for t in range(max_len):
            values = [
                pts[t] for pts in all_per_turn_latencies if t < len(pts)
            ]
            if values:
                metrics[f"agent/turn_{t+1}_latency_mean"] = sum(values) / len(values)

    return metrics


# ---------------------------------------------------------------------------
# Rollout wrappers
# ---------------------------------------------------------------------------


async def generate_rollout_async(
    args: Namespace,
    rollout_id: int,
    data_source: Callable[[int], list[list[Sample]]],
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """Custom rollout wrapping the base sglang rollout with metrics."""
    from miles.rollout.sglang_rollout import (
        generate_rollout_async as base_generate_rollout_async,
    )

    rollout_output, aborted_samples = await base_generate_rollout_async(
        args, rollout_id, data_source
    )

    all_samples: list[Sample] = []
    for group in rollout_output.samples:
        if isinstance(group[0], list):
            for sample_list in group:
                all_samples.extend(sample_list)
        else:
            all_samples.extend(group)

    agent_metrics = aggregate_agent_metrics(all_samples)

    rollout_metrics = rollout_output.metrics or {}
    rollout_metrics.update(agent_metrics)

    logger.info(
        "Aggregated agent metrics for rollout %d: %s", rollout_id, agent_metrics
    )

    return (
        RolloutFnTrainOutput(
            samples=rollout_output.samples, metrics=rollout_metrics
        ),
        aborted_samples,
    )


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_buffer: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Synchronous rollout entry point for miles training loop."""
    output, aborted_samples = generate_abortable_samples(
        args, rollout_id, data_buffer.get_samples, evaluation=evaluation
    )
    data_buffer.add_samples(aborted_samples)
    return output


def generate_abortable_samples(
    args: Namespace,
    rollout_id: int,
    data_source: Callable[[int], list[list[Sample]]],
    evaluation: bool = False,
) -> tuple[Any, list[list[Sample]]]:
    assert args.rollout_global_dataset
    if evaluation:
        return run(eval_rollout(args, rollout_id))
    return run(generate_rollout_async(args, rollout_id, data_source))
