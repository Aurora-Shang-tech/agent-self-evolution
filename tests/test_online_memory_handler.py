import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.core.trajectory import Trajectory, TrajectoryStep
from src.counterfactual.verifier import CounterfactualResult
from src.evolution.online_memory_handler import (
    OnlineMemoryHandler,
)
from src.evolution.online_scheduler import sha256


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1

        return "\n".join([
            "Critical step: 0",
            "Failure type: Incorrect action",
            "Failure reason: A better action is available.",
            "Original action: look",
            "Counterfactual action 1: examine desk 1",
            "Counterfactual action 2: NONE",
            "Counterfactual action 3: NONE",
            "Expected effect: Inspect the desk.",
        ])


class OnlineMemoryHandlerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        self.root = Path(temp.name)
        self.trace_path = self.root / "trajectory.json"
        self.event_path = self.root / "event.json"
        self.outcome_path = (
            self.root / "reviews" / "fresh_00"
            / "review_outcome.json"
        )

        trace = Trajectory(
            task="look at bowl under the desklamp.",
            gamefile="mock_game",
            success=False,
            steps=[
                TrajectoryStep(
                    observation="desk",
                    admissible_actions=[
                        "look",
                        "examine desk 1",
                    ],
                    thought="inspect",
                    action="look",
                    valid_format=True,
                    admissible=True,
                    next_observation="desk",
                    score=0.0,
                    done=False,
                    won=False,
                )
            ],
        )

        trace.save_json(str(self.trace_path))

        event = {
            "schema": "unified_feedback_event_v1",
            "episode_id": "fresh_00",
            "game_id": "train:mock",
            "phase": "learning",
            "trajectory_path": str(self.trace_path),
            "trajectory_sha256": sha256(self.trace_path),
            "memory_version": "v000",
            "memory_sha256": "memory_hash",
            "active_skill_id": None,
            "decision": {
                "route": "MEMORY_REVIEW",
                "success": False,
            },
        }

        self.event_path.write_text(
            json.dumps(event),
            encoding="utf-8",
        )

        self.job = {
            "job_id": "fresh_00",
            "game_id": "train:mock",
            "route": "MEMORY_REVIEW",
            "source_scope": "ONLINE_TRAIN_DEVELOPMENT",
            "trajectory_path": str(self.trace_path),
            "trajectory_sha256": sha256(self.trace_path),
            "event_path": str(self.event_path),
            "event_sha256": sha256(self.event_path),
        }

        self.caps = SimpleNamespace(
            memory_info=SimpleNamespace(
                version="v000",
                memory_hash="memory_hash",
            ),
            skill_id=None,
        )

        self.fake_llm = FakeLLM()

        self.handler = OnlineMemoryHandler(
            self.caps,
            model="mock-model",
            max_steps=50,
            top_k=3,
            max_continuation_decisions=12,
            max_llm_calls=16,
            llm_factory=lambda: self.fake_llm,
        )

    def fake_agent(self, llm, caps, **kwargs):
        return SimpleNamespace(
            llm=llm,
            llm_calls=0,
        )

    def failed_verification(self, trace, index, action, agent, **kwargs):
        agent.llm.chat([])
        return CounterfactualResult(
            critical_step=index,
            original_action=trace.steps[index].action,
            counterfactual_action=action,
            success=False,
            continuation_steps=1,
            trajectory=None,
        )

    def test_failed_branch_produces_no_memory_update(self):
        with (
            patch(
                "src.evolution.online_memory_handler.build_active_agent",
                side_effect=self.fake_agent,
            ),
            patch(
                "src.evolution.online_memory_handler.verify_counterfactual",
                side_effect=self.failed_verification,
            ),
        ):
            status = self.handler.process(
                self.job,
                self.outcome_path,
            )

        self.assertEqual(status, "NO_SUCCESS_WITHIN_BUDGET")
        self.assertEqual(self.fake_llm.calls, 2)

        record = json.loads(
            self.outcome_path.read_text(encoding="utf-8")
        )

        self.assertEqual(record["memory_operation"], "NO_CHANGE")
        self.assertFalse(record["memory_updated"])
        self.assertIsNone(record["candidate_sha256"])

    def test_completed_review_is_reused(self):
        with (
            patch(
                "src.evolution.online_memory_handler.build_active_agent",
                side_effect=self.fake_agent,
            ),
            patch(
                "src.evolution.online_memory_handler.verify_counterfactual",
                side_effect=self.failed_verification,
            ),
        ):
            self.handler.process(self.job, self.outcome_path)

        original_calls = self.fake_llm.calls

        self.assertEqual(
            self.handler.process(self.job, self.outcome_path),
            "NO_SUCCESS_WITHIN_BUDGET",
        )

        self.assertEqual(
            self.fake_llm.calls,
            original_calls,
        )

    def test_budget_exhaustion_preserves_analysis(self):
        self.handler.max_llm_calls = 1

        with (
            patch(
                "src.evolution.online_memory_handler.build_active_agent",
                side_effect=self.fake_agent,
            ),
            patch(
                "src.evolution.online_memory_handler.verify_counterfactual",
                side_effect=self.failed_verification,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "budget exhausted",
            ):
                self.handler.process(
                    self.job,
                    self.outcome_path,
                )

        self.assertTrue(
            (
                self.outcome_path.parent
                / "automated_analysis.json"
            ).is_file()
        )
        self.assertFalse(self.outcome_path.exists())

        self.handler.max_llm_calls = 16

        with (
            patch(
                "src.evolution.online_memory_handler.build_active_agent",
                side_effect=self.fake_agent,
            ),
            patch(
                "src.evolution.online_memory_handler.verify_counterfactual",
                side_effect=self.failed_verification,
            ),
        ):
            self.handler.process(
                self.job,
                self.outcome_path,
            )

        self.assertEqual(self.fake_llm.calls, 2)
        self.assertTrue(self.outcome_path.exists())

    def test_holdout_cannot_enter_memory_learning(self):
        event = json.loads(
            self.event_path.read_text(encoding="utf-8")
        )

        event["phase"] = "holdout"

        self.event_path.write_text(
            json.dumps(event),
            encoding="utf-8",
        )

        self.job["event_sha256"] = sha256(self.event_path)

        with self.assertRaises(ValueError):
            self.handler.process(
                self.job,
                self.outcome_path,
            )

        self.assertEqual(self.fake_llm.calls, 0)


if __name__ == "__main__":
    unittest.main()
