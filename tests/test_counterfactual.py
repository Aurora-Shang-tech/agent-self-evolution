"""测试完整反事实验证"""

from pathlib import Path
from src.agents.react_agent import ReActAgent
from src.core.trajectory import Trajectory
from src.counterfactual.verifier import verify_counterfactual_candidates
from src.llm.client import LLMClient

TRAJECTORY_PATH = Path("outputs/baseline_qwen3.8-chat_train_seed42_50/episode_005.json")
CRITICAL_STEP = 12
COUNTERFACTUAL_ACTIONS = (
    "go to countertop 2",
    "go to countertop 3",
    "go to cabinet 4",
)
MAX_STEPS = 50

def main():
    # 1.加载失败轨迹
    trajectory = Trajectory.load_json(str(TRAJECTORY_PATH))

    print("=" * 60)
    print("Counterfactual Verification")
    print("=" * 60)

    print(f"Task: {trajectory.task}")
    print(f"Original success: {trajectory.success}")
    print(f"Critical step: {CRITICAL_STEP}")
    print(
        "Original action: "
        f"{trajectory.steps[CRITICAL_STEP].action}"
    )
    print("Counterfactual actions: ")
    for index, action in enumerate(COUNTERFACTUAL_ACTIONS, start=1):
        print(f" {index}. {action}")

    # 2.创建与baseline相同的LLM
    llm = LLMClient()
    agent = ReActAgent(
        llm=llm,
        max_steps=MAX_STEPS,
    )

    # 3.真正执行反事实验证
    result = verify_counterfactual_candidates(
        trajectory=trajectory,
        critical_step=CRITICAL_STEP,
        counterfactual_actions=COUNTERFACTUAL_ACTIONS,
        agent=agent,
        max_steps=MAX_STEPS,
    )

    # 4. 输出结果
    print()
    print("=" * 60)
    print("Counterfactual Result")
    print("=" * 60)

    print(f"Critical step: {result.critical_step}")
    print(f"Original action: {result.original_action}")
    for index, candidate in enumerate(result.results, start=1):
        print()
        print(f"Candidate {index}: {candidate.counterfactual_action}")
        print(f"Success: {candidate.success}")
        print(f"Continuation steps: {candidate.continuation_steps}")

        if candidate.trajectory is not None:
            output_path = Path(f"outputs/counterfactual_episode005_candidate_{index}.json")
            candidate.trajectory.save_json(str(output_path))
            print(f"Saved trajectory: {output_path}")


if __name__ == "__main__":
    main()



import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.core.trajectory import TrajectoryStep
from src.counterfactual.verifier import (
    CounterfactualBatchResult,
    verify_counterfactual,
)
from src.memory.updater import ComparativeExtractor


class DirectWinTests(unittest.TestCase):
    def test_direct_win_can_be_saved_and_extracted(self):
        baseline = Trajectory(
            task="put a cool potato in microwave.",
            gamefile="/tmp/example-game",
        )

        baseline.add_step(
            TrajectoryStep(
                observation="Kitchen",
                admissible_actions=["look", "take potato 1"],
                thought="Search",
                action="look",
                valid_format=True,
                admissible=True,
                next_observation="Kitchen",
                score=0,
                done=False,
                won=False,
            )
        )

        baseline.messages = [
            {"role": "system", "content": "Test"},
            {"role": "user", "content": "Kitchen"},
        ]

        replayed = SimpleNamespace(
            observation="Kitchen",
            admissible_actions=("look", "take potato 1"),
        )

        env = Mock()
        env.step.return_value = SimpleNamespace(
            observation="Task completed",
            admissible_actions=(),
            admissible=True,
            score=1.0,
            done=True,
            won=True,
        )

        agent = SimpleNamespace(
            use_state_tracking=False,
            experience_store=None,
            _state_reminder=lambda tracker: "",
            continue_from=Mock(),
        )

        with patch(
            "src.counterfactual.verifier.replay_to_step",
            return_value=(env, replayed),
        ):
            result = verify_counterfactual(
                baseline,
                critical_step=0,
                counterfactual_action="take potato 1",
                agent=agent,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.continuation_steps, 0)
        self.assertEqual(len(result.trajectory.steps), 1)
        self.assertTrue(result.trajectory.steps[0].won)
        self.assertEqual(
            result.trajectory.steps[0].next_observation,
            "Task completed",
        )

        agent.continue_from.assert_not_called()
        env.close.assert_called_once()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branch.json"
            result.trajectory.save_json(str(path))
            restored = Trajectory.load_json(str(path))

        self.assertTrue(restored.success)
        self.assertEqual(restored.steps[0].action, "take potato 1")

        response = json.dumps({
            "decision": "CANDIDATE",
            "lesson": (
                "When the target action is available, "
                "execute it instead of repeating searches."
            ),
            "reason": "The alternative action completed the task.",
            "limitations": "Evidence comes from one task.",
            "evidence_refs": [
                "baseline:critical",
                "counterfactual:replacement",
                "counterfactual:step_0",
            ],
        })

        extractor = ComparativeExtractor(
            SimpleNamespace(chat=lambda messages: response)
        )

        candidate = extractor.extract(
            baseline,
            CounterfactualBatchResult(
                critical_step=0,
                original_action="look",
                results=(result,),
            ),
        )

        self.assertEqual(candidate.decision, "CANDIDATE")
        self.assertEqual(
            candidate.evidence["counterfactual:replacement"][
                "observation_after"
            ],
            "Task completed",
        )
