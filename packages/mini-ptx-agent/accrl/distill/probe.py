#!/usr/bin/env python3
"""Quick connectivity probe for distill model endpoints.

Usage:
    python -m accrl.distill.probe Kimi-K2.6
    python -m accrl.distill.probe GLM-5.1 --thinking
    python -m accrl.distill.probe --all
"""

import argparse
import asyncio
import json
import time

import litellm

from accrl.distill.run_experiment import MODEL_CONFIGS, resolve_model

SMOKE_PROMPT = "What is 2+2? Answer in one sentence."


async def test_one(model_key: str, enable_thinking: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"Testing: {model_key} (thinking={enable_thinking})")
    print(f"{'='*60}")

    model_name, model_kwargs, client, http_client = resolve_model(
        model_key, enable_thinking=enable_thinking,
    )

    # Show what we're sending
    cfg = MODEL_CONFIGS[model_key]
    print(f"\n--- Request ---")
    print(f"  litellm model:  {model_name}")
    print(f"  api_base:       {cfg.get('api_base', '(default)')}")
    if client is not None:
        print(f"  custom client:  yes (base_url={client.base_url})")
        print(f"    headers:      {dict(client._custom_headers)}")
        print(f"    query:        {dict(client._custom_query)}")
        print(f"    verify_ssl:   {cfg.get('verify_ssl', True)}")
    else:
        print(f"  custom client:  no")
    if "extra_body" in model_kwargs:
        print(f"  extra_body:     {json.dumps(model_kwargs['extra_body'])}")
    print(f"  prompt:         {SMOKE_PROMPT!r}")

    try:
        extra = {"client": client} if client is not None else {}
        t0 = time.time()
        resp = await litellm.acompletion(
            model=model_name,
            messages=[{"role": "user", "content": SMOKE_PROMPT}],
            max_tokens=256,
            drop_params=True,
            **model_kwargs,
            **extra,
        )
        elapsed = time.time() - t0

        msg = resp.choices[0].message
        content = msg.content or ""
        thinking = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        )

        usage = resp.usage
        print(f"\n--- Response ---")
        print(f"  Status:     OK ({elapsed:.1f}s)")
        print(f"  Model:      {resp.model}")
        print(f"  Content:    {content}")
        if thinking:
            print(f"  Thinking:   {thinking[:500]}{'...' if len(thinking) > 500 else ''}")
        else:
            print(f"  Thinking:   (empty)")
        if usage:
            print(f"  Tokens:     prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
            details = getattr(usage, "completion_tokens_details", None)
            if details:
                print(f"  Details:    {details}")
        print(f"  Finish:     {resp.choices[0].finish_reason}")
    except Exception as e:
        print(f"\n--- Response ---")
        print(f"  FAILED:     {type(e).__name__}: {e}")
    finally:
        if http_client is not None:
            await http_client.aclose()


async def main():
    parser = argparse.ArgumentParser(description="Probe distill model endpoints")
    parser.add_argument("models", nargs="*", help="Model name(s) from MODEL_CONFIGS")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking toggle")
    parser.add_argument("--all", action="store_true", help="Test all models")
    args = parser.parse_args()

    if args.all:
        models = list(MODEL_CONFIGS.keys())
    elif args.models:
        models = args.models
    else:
        parser.error("Provide model name(s) or --all")

    for m in models:
        if m not in MODEL_CONFIGS:
            print(f"Unknown model: {m}. Available: {list(MODEL_CONFIGS.keys())}")
            continue
        await test_one(m, enable_thinking=args.thinking)


if __name__ == "__main__":
    asyncio.run(main())
