"""Behavioral checks for the live no-autotune gate."""

from __future__ import annotations

import hashlib
import unittest

from multiturn.candidate_policy import VERIFIER_SHA256
from multiturn.policy_gate import evaluate_candidate, policy_record, validate_config, validate_resume_policy


POLICY_CONFIG = {
    "kernel_policy": "autotune-banned-v1",
    "language": "triton",
    "prompt_tag": "triton-blackwell-autotune-banned-v1",
    "verifier_sha256": VERIFIER_SHA256,
}


class PolicyGateTest(unittest.TestCase):
    def test_banned_source_never_reaches_evaluator(self) -> None:
        source = "import triton\n@triton.autotune(configs=[], key=[])\ndef kernel(): pass\n"

        def evaluator(_source: str) -> dict:
            self.fail("Rejected source reached the evaluator")

        result = evaluate_candidate(POLICY_CONFIG, source, evaluator)
        self.assertEqual(result["returncode"], 1)
        self.assertEqual(result["extra"]["event"], "kernel_policy_rejected")
        self.assertEqual(result["extra"]["verification"]["status"], "rejected")
        self.assertEqual(result["extra"]["verification"]["source_sha256"], hashlib.sha256(source.encode()).hexdigest())
        self.assertIsNone(result["extra"]["traces"])

    def test_unverifiable_source_never_reaches_evaluator(self) -> None:
        def evaluator(_source: str) -> dict:
            self.fail("Unverifiable source reached the evaluator")

        result = evaluate_candidate(POLICY_CONFIG, "getattr(triton, 'autotune')\n", evaluator)
        self.assertEqual(result["extra"]["verification"]["status"], "unverifiable")

    def test_explicit_source_passes_through_unchanged(self) -> None:
        source = "import triton\n@triton.heuristics({'BLOCK': lambda args: 128})\ndef kernel(): pass\n"
        seen = []

        def evaluator(candidate: str) -> dict:
            seen.append(candidate)
            return {"output": "ok", "returncode": 0, "exception_info": "", "extra": {}}

        result = evaluate_candidate(POLICY_CONFIG, source, evaluator)
        self.assertEqual(seen, [source])
        self.assertEqual(result["extra"]["verification"]["status"], "passed")

    def test_policy_identity_is_bound_before_evaluation(self) -> None:
        for change in (
            {"verifier_sha256": "wrong"},
            {"prompt_tag": "triton-blackwell"},
            {"language": "cuda"},
        ):
            with self.subTest(change=change):
                config = {**POLICY_CONFIG, **change}
                with self.assertRaises(ValueError):
                    evaluate_candidate(config, "pass\n", lambda _: self.fail("Evaluator called"))
        self.assertEqual(
            policy_record(POLICY_CONFIG, "prompt-hash"),
            {"id": "autotune-banned-v1", "verifier_sha256": VERIFIER_SHA256, "prompt_sha256": "prompt-hash"},
        )

    def test_default_policy_forwards_without_verification(self) -> None:
        config = {"kernel_policy": "default", "language": "triton", "prompt_tag": "triton-blackwell"}
        self.assertEqual(validate_config(config), "default")
        expected = {"output": "ok", "returncode": 0, "exception_info": ""}
        self.assertIs(evaluate_candidate(config, "pass\n", lambda _: expected), expected)

    def test_resume_requires_the_same_policy_verifier_and_prompt(self) -> None:
        saved = {"info": {"kernel_policy": policy_record(POLICY_CONFIG, "prompt-hash")}}
        validate_resume_policy(POLICY_CONFIG, "prompt-hash", saved)
        for changed in (
            {"info": {"kernel_policy": {**saved["info"]["kernel_policy"], "verifier_sha256": "wrong"}}},
            {"info": {"kernel_policy": {"id": "default"}}},
        ):
            with self.assertRaises(ValueError):
                validate_resume_policy(POLICY_CONFIG, "prompt-hash", changed)
        with self.assertRaises(ValueError):
            validate_resume_policy(POLICY_CONFIG, "changed-prompt", saved)


if __name__ == "__main__":
    unittest.main()
