"""Check the model-visible feedback policy used by the manuscript ablation."""

from __future__ import annotations

import logging
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import BLIND_REFINEMENT_PROMPT, KernelAgent, LLM_CONTEXT_POLICIES  # noqa: E402


class FakeEnvironment:
    def __init__(self) -> None:
        self.sources: list[str] = []

    def evaluate_kernel(self, code: str) -> dict:
        self.sources.append(code)
        return {
            "output": f"Evaluator feedback for {code}",
            "returncode": 1,
            "exception_info": "",
            "extra": {"traces": [{"evaluation": {"status": "FAILED"}}], "target_met": False},
        }


class FakeModel:
    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict) -> list[dict]:
        output = outputs[0]
        return [{"role": "user", "content": output["output"],
                 "extra": {"raw_output": output["output"], **output["extra"]}}]


def agent_for(policy: str, language: str = "cuda") -> KernelAgent:
    agent = object.__new__(KernelAgent)
    agent.llm_context_policy = policy
    agent.language = language
    agent.messages = [{"role": "system", "content": "system"},
                      {"role": "user", "content": "task"}]
    agent.env = FakeEnvironment()
    agent.model = FakeModel()
    agent.logger = logging.getLogger(__name__)
    agent.get_template_vars = lambda: {}
    return agent


class FullBlindRunnerTest(unittest.TestCase):
    def test_language_and_feedback_matrix_extracts_and_retains_candidates(self) -> None:
        for language, fence in (("cuda", "cpp"), ("triton", "python")):
            for policy in LLM_CONTEXT_POLICIES:
                with self.subTest(language=language, policy=policy):
                    agent = agent_for(policy, language)
                    for turn in (1, 2):
                        answer = {"role": "assistant", "content": f"```{fence}\nkernel_{turn}\n```"}
                        agent.add_messages(answer)
                        agent.evaluate_assistant_message(answer)
                    self.assertEqual(agent.env.sources, ["kernel_1", "kernel_2"])
                    visible = agent._messages_for_llm()
                    self.assertEqual(len(visible), 6)
                    self.assertTrue(all("extra" not in message for message in visible))
                    expected_feedback = (
                        BLIND_REFINEMENT_PROMPT if policy == "full-blind-refinement"
                        else "Evaluator feedback for kernel_2"
                    )
                    self.assertEqual(visible[-1]["content"], expected_feedback)

    def test_full_blind_keeps_all_candidates_and_hides_evaluator_feedback(self) -> None:
        self.assertEqual(LLM_CONTEXT_POLICIES, ("full", "full-blind-refinement"))
        agent = agent_for("full-blind-refinement")
        for turn in (1, 2):
            answer = {"role": "assistant", "content": f"candidate {turn}"}
            agent.add_messages(answer)
            agent.evaluate_assistant_message(answer, kernel_code_override=f"kernel {turn}")
        visible = agent._messages_for_llm()
        self.assertEqual([message["content"] for message in visible],
                         ["system", "task", "candidate 1", BLIND_REFINEMENT_PROMPT,
                          "candidate 2", BLIND_REFINEMENT_PROMPT])
        self.assertTrue(all("extra" not in message for message in visible))
        for observation in (agent.messages[3], agent.messages[5]):
            self.assertIn("Evaluator feedback", observation["extra"]["raw_output"])
            self.assertEqual(observation["extra"]["llm_context_policy"], "full-blind-refinement")
            self.assertEqual(observation["extra"]["traces"][0]["evaluation"]["status"], "FAILED")

    def test_full_policy_exposes_evaluator_feedback(self) -> None:
        agent = agent_for("full")
        answer = {"role": "assistant", "content": "candidate"}
        agent.add_messages(answer)
        agent.evaluate_assistant_message(answer, kernel_code_override="kernel")
        self.assertEqual(agent._messages_for_llm()[-1]["content"], "Evaluator feedback for kernel")


if __name__ == "__main__":
    unittest.main()
