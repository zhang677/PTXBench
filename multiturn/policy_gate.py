"""Apply the opt-in source policy before a candidate reaches a test.

The verifier is the bundled autotune-banned-v1 implementation.
Keep the test scripts unchanged; callers run the evaluator only through this
gate for policy-enabled configurations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from multiturn.candidate_policy import (
    VERIFIER_SHA256,
    policy_metadata,
    rejection_feedback,
    validate_policy,
    validate_policy_tag,
    verify_candidate,
)


def validate_config(config: Mapping[str, Any]) -> str:
    """Reject an unbound or changed policy before evaluation starts."""
    policy = config.get("kernel_policy", "default")
    validate_policy(policy, config["language"])
    validate_policy_tag(policy, config.get("prompt_tag"))
    if policy != "default" and config.get("verifier_sha256") != VERIFIER_SHA256:
        raise ValueError("Policy verifier hash differs from the experiment config; use a fresh experiment version.")
    return policy


def policy_record(config: Mapping[str, Any], prompt_sha256: str) -> dict[str, str]:
    """Record the policy identity saved with a trajectory."""
    policy = validate_config(config)
    return {**policy_metadata(policy), "prompt_sha256": prompt_sha256}


def validate_resume_policy(
    config: Mapping[str, Any], prompt_sha256: str, trajectory: Mapping[str, Any]
) -> None:
    """Reject a resume whose policy, verifier, or model-visible prompt changed."""
    policy = validate_config(config)
    saved = trajectory.get("info", {}).get("kernel_policy", {"id": "default"})
    if saved.get("id") != policy or (policy != "default" and saved != policy_record(config, prompt_sha256)):
        raise ValueError("Resume kernel policy/verifier differs from the saved trajectory; use a fresh experiment root.")


def evaluate_candidate(
    config: Mapping[str, Any],
    source: str,
    evaluator: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Check source, then forward its original bytes to the evaluator if admitted.

    The caller owns turn accounting. A rejected candidate still consumes its
    attempted turn, as required by the multi-turn protocol.
    """
    policy = validate_config(config)
    verification = None
    if policy != "default":
        verification = verify_candidate(source, policy)
        if verification["status"] != "passed":
            return {
                "output": rejection_feedback(verification),
                "returncode": 1,
                "exception_info": "",
                "extra": {
                    "event": "kernel_policy_rejected",
                    "verification": verification,
                    "traces": None,
                    "all_passed": False,
                    "target_met": False,
                },
            }
    result = evaluator(source)
    if verification is not None:
        result.setdefault("extra", {})["verification"] = verification
    return result
