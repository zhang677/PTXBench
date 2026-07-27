"""Reverse-CoT distillation: generate reasoning tokens for expert kernels.

Usage:
    python -m accrl.distill.run_experiment \
        --config accrl/distill/configs/full_context_best_kernel.yaml \
        --turns accrl/distill/data/gemini_turns.jsonl \
        --output-dir accrl/distill/data/experiments/full_context_best_kernel/
"""

import argparse
import asyncio
import datetime as _dt
import json
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import httpx
import litellm
from openai import AsyncOpenAI

from accrl.distill.utils import (
    ExperimentConfig,
    SELECTORS,
    build_context,
    build_trajectory_context,
    make_prompt,
    passes_quality,
)

logger = logging.getLogger(__name__)

DISTILL_SYSTEM_PROMPT = """\
You are a CUDA kernel optimization expert generating training data for reinforcement learning.

Your task is to produce high-quality reasoning traces that explain the thought process behind expert-written CUDA kernels. These reasoning traces will be used to train another AI model to write optimized GPU kernels.

Requirements:
- Think at MAXIMUM depth and effort
- Be extremely concrete: reference specific numbers, constants, and code patterns
- Explain WHY, not just WHAT — every design choice should have a justification
- You may include code snippets, pseudocode, or partial implementations in your reasoning if it helps illustrate your thought process
- Output your reasoning inside <my_reasoning>...</my_reasoning> tags"""

MODEL_CONFIGS = {
    "MiniMax-M2.7": {
        "model": "openai/MiniMaxAI/MiniMax-M2.7",
        "api_base": "https://llm.gateway.msl-cw-use2-2.cw.metafb.cloud/playground/rift-chengze-minimax-m2p7-2017cbk/v1",
        "api_key": "dummy",
    },
    "GLM-5.1": {
        "model": "openai/zai-org/GLM-5.1-FP8",
        "api_base": "https://llm.gateway.msl-cw-use2-2.cw.metafb.cloud/playground/rift-chengze-glm5-1-1ac4a8k/v1",
        "api_key": "dummy",
    },
    "GLM-5.1-openrouter": {
        "model": "openrouter/z-ai/glm-5.1",
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
    },
    "gemini-3.1-pro-preview": {
        "model": "gemini/gemini-3.1-pro-preview",
        "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""),
    },
    # Routes through Meta's pod-internal LLM gateway, which requires both a
    # query-string tier name and a custom predictor header. litellm doesn't
    # forward query params, so we build a custom AsyncOpenAI client at runtime.
    "Kimi-K2.6": {
        "model": "openai/kimi-2.6",
        "api_base": "https://pod-internal.fbinfra.net/v1",
        "api_key": "dummy",
        "default_headers": {"X-Predictor-Tier": "smc.rift_chengze_kimi_k26_e75cf6t.https"},
        "default_query": {"smc_tier_name": "inference_platform.llm_gateway_https"},
        "verify_ssl": False,
    },
    # 1M-context (max_model_len=1048576). Returns CoT in `reasoning` field
    # (not `reasoning_content`); _generate_one checks both.
    "DeepSeek-V4-Pro": {
        "model": "openai/deepseek-v4-pro",
        "api_base": "https://llm.gateway.msl-cw-use2-2.cw.metafb.cloud/rift/rift-chengze-deepseek-v4-ddd62ak/v1",
        "api_key": "dummy",
    },
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_REASONING_RE = re.compile(r"<my_reasoning>(.*?)</my_reasoning>", re.DOTALL)


def _extract_reasoning(raw: str) -> str:
    """Strip model's own <think> block, extract <my_reasoning> content."""
    cleaned = _THINK_RE.sub("", raw).strip()
    m = _REASONING_RE.search(cleaned)
    return m.group(1).strip() if m else cleaned


async def _generate_one(prompt: str, model_name: str, model_kwargs: dict, sem: asyncio.Semaphore, client: AsyncOpenAI | None = None, max_tokens: int = 196000, context_limit: int = 202752) -> tuple[str, str] | None:
    """Returns (reasoning, thinking) where reasoning is the extracted <my_reasoning>
    block from `content` and thinking is the raw `reasoning_content` channel
    (Kimi's hidden meta-planning; "" for models that don't expose it)."""
    async with sem:
        try:
            messages = [
                {"role": "system", "content": DISTILL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            # Count input tokens via litellm, fall back to conservative estimate
            try:
                input_tokens = litellm.token_counter(model=model_name, messages=messages)
            except Exception:
                input_tokens = len(DISTILL_SYSTEM_PROMPT + prompt) // 2  # ~2 chars/token for code
            actual_max = min(max_tokens, context_limit - input_tokens - 100)
            if actual_max < 1000:
                return None
            extra = {"client": client} if client is not None else {}
            resp = await litellm.acompletion(
                model=model_name,
                messages=messages,
                max_tokens=actual_max,
                drop_params=True,
                **model_kwargs,
                **extra,
            )
            msg = resp.choices[0].message
            content = msg.content or ""
            # Different vLLM models surface CoT under different field names:
            # GLM/Kimi → reasoning_content, DeepSeek → reasoning.
            thinking = (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or ""
            )
            return _extract_reasoning(content), thinking
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return None


def _repo_provenance(repo: Path) -> dict:
    """Capture git SHA + uncommitted-state for the AccRL repo. Untracked files
    don't count as 'dirty' (they don't affect what code runs) but are recorded."""
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo,
            capture_output=True, text=True, check=True,
        ).stdout
    sha = git("rev-parse", "HEAD").strip()
    dirty = [l for l in git("status", "--porcelain", "--untracked-files=no").splitlines() if l]
    untracked = [l for l in git("ls-files", "--others", "--exclude-standard").splitlines() if l]
    return {"sha": sha, "dirty_files": dirty, "untracked_files": untracked}


def resolve_model(
    model_name_key: str,
    enable_thinking: bool = False,
    timeout: float = 600.0,
) -> tuple[str, dict, AsyncOpenAI | None, httpx.AsyncClient | None]:
    """Build model name, kwargs, and optional custom client from MODEL_CONFIGS.

    Returns (litellm_model_name, model_kwargs, custom_client, http_client).
    Caller must close http_client when done (if not None).
    """
    model_cfg = dict(MODEL_CONFIGS[model_name_key])
    model_name = model_cfg.pop("model")
    default_headers = model_cfg.pop("default_headers", None)
    default_query = model_cfg.pop("default_query", None)
    verify_ssl = model_cfg.pop("verify_ssl", True)
    model_kwargs = {k: v for k, v in model_cfg.items() if v}

    if model_name_key in ("GLM-5.1", "Kimi-K2.6", "DeepSeek-V4-Pro"):
        model_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

    custom_client = None
    http_client = None
    if default_query is not None or verify_ssl is False:
        http_client = httpx.AsyncClient(verify=verify_ssl, timeout=timeout)
        custom_client = AsyncOpenAI(
            api_key=model_kwargs.pop("api_key", "dummy"),
            base_url=model_kwargs.pop("api_base"),
            http_client=http_client,
            default_headers=default_headers or {},
            default_query=default_query or {},
        )

    return model_name, model_kwargs, custom_client, http_client


def load_turns_grouped(path: str) -> dict[str, list[dict]]:
    by_exp: dict[str, list[dict]] = {}
    with open(path) as f:
        for line in f:
            t = json.loads(line)
            by_exp.setdefault(t.get("exp_id", "?"), []).append(t)
    for v in by_exp.values():
        v.sort(key=lambda t: t["turn"])
    return by_exp


async def run(config: ExperimentConfig, turns_path: str, output_dir: Path,
              max_concurrent: int = 8, allow_dirty: bool = False,
              config_path: str | None = None, timeout: float = 600.0):
    import shutil
    output_dir.mkdir(parents=True, exist_ok=True)
    config.to_yaml(output_dir / "config.yaml")

    if config_path is not None:
        shutil.copy2(config_path, output_dir / "source_config.yaml")

    # Reproducibility: refuse to launch with uncommitted code, log SHA + command.
    repo = Path(__file__).resolve().parents[2]
    prov = _repo_provenance(repo)
    if prov["dirty_files"] and not allow_dirty:
        raise SystemExit(
            f"Refusing to run with {len(prov['dirty_files'])} uncommitted change(s) in {repo}.\n"
            + "\n".join(f"  {l}" for l in prov["dirty_files"])
            + "\n\nCommit, stash, or pass --allow-dirty to bypass."
        )
    provenance = {
        "command": " ".join(shlex.quote(a) for a in sys.argv),
        "argv": sys.argv,
        "git_sha": prov["sha"],
        "git_dirty_files": prov["dirty_files"],
        "git_untracked_files": prov["untracked_files"],
        "allow_dirty_bypass": bool(prov["dirty_files"]) and allow_dirty,
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "config_path": str(Path(config_path).resolve()) if config_path else None,
        "turns_path": str(Path(turns_path).resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    logger.info(f"[{config.name}] git={prov['sha'][:8]} dirty={len(prov['dirty_files'])} files; provenance.json written")

    # Select turns
    selector = SELECTORS[config.selector]
    selected = []
    for traj_turns in load_turns_grouped(turns_path).values():
        selected.extend(selector(traj_turns))

    if config.min_speedup > 0:
        selected = [t for t in selected if t.get("speedup") and t["speedup"] >= config.min_speedup]
    selected = [t for t in selected if t.get("kernel_code")]

    logger.info(f"[{config.name}] {len(selected)} turns selected")
    if not selected:
        return

    # Build prompts — use trajectory context if turns have history
    ctx_fn = build_trajectory_context if config.selector == "all_with_history" else build_context
    prompts = [make_prompt(ctx_fn(t), config.prompt_version) for t in selected]

    model_name, model_kwargs, custom_client, http_client = resolve_model(
        config.reasoning_model, enable_thinking=config.enable_thinking,
        timeout=timeout,
    )

    # Generate reasoning — write each result as it completes
    logger.info(f"[{config.name}] Generating reasoning...")
    sem = asyncio.Semaphore(max_concurrent)
    pairs_path = output_dir / "reasoning_pairs.jsonl"
    count = 0

    pairs_f = open(pairs_path, "w")

    async def _process_one(turn, prompt, idx):
        nonlocal count
        result = await _generate_one(prompt, model_name, model_kwargs, sem, client=custom_client, max_tokens=config.max_tokens)
        if result is None:
            logger.warning(f"  [{idx+1}/{len(selected)}] {turn.get('exp_id')} turn={turn['turn']} FAILED")
            return
        reasoning, thinking = result
        if not reasoning:
            logger.warning(f"  [{idx+1}/{len(selected)}] {turn.get('exp_id')} turn={turn['turn']} FAILED (empty content)")
            return
        if not passes_quality(reasoning, turn["kernel_code"], config.min_reasoning_len, config.max_reasoning_len):
            logger.warning(f"  [{idx+1}/{len(selected)}] {turn.get('exp_id')} turn={turn['turn']} DROPPED (quality)")
            return

        pair = {
            "system_prompt": DISTILL_SYSTEM_PROMPT,
            "input": prompt,
            "reasoning": reasoning,
            "thinking": thinking,
            "metadata": {
                "exp_id": turn.get("exp_id"),
                "turn": turn["turn"],
                "speedup": turn.get("speedup"),
                "passed": turn.get("passed"),
                "definition_name": turn["definition_name"],
            },
        }
        pairs_f.write(json.dumps(pair) + "\n")
        pairs_f.flush()
        count += 1

        logger.info(f"  [{idx+1}/{len(selected)}] {turn.get('exp_id')} turn={turn['turn']} OK (reasoning={len(reasoning)} thinking={len(thinking)} chars)")

    try:
        await asyncio.gather(*[_process_one(t, p, i) for i, (t, p) in enumerate(zip(selected, prompts))])
    finally:
        pairs_f.close()
        if http_client is not None:
            await http_client.aclose()
    logger.info(f"[{config.name}] Done: {count} reasoning pairs written to {pairs_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate reasoning tokens for expert kernels")
    parser.add_argument("--config", required=True)
    parser.add_argument("--turns", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-concurrent", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Per-request timeout in seconds (default: 600)")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Run even if the AccRL working tree has uncommitted changes (recorded in provenance.json).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    config = ExperimentConfig.from_yaml(args.config)
    asyncio.run(run(config, args.turns, Path(args.output_dir), args.max_concurrent,
                    allow_dirty=args.allow_dirty, config_path=args.config,
                    timeout=args.timeout))


if __name__ == "__main__":
    main()
