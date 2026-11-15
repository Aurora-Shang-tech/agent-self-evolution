import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.core.trajectory import Trajectory, TrajectoryStep
from src.data.alfworld_data import GameExample
from src.evolution.feedback_router import FeedbackRouter
from src.evolution.unified_controller import UnifiedFeedbackController


TASK_TYPE = "pick_heat_then_place_in_recep"


def make_trace(gamefile, actions, success):
    return Trajectory(
        task="heat some egg and put it in sidetable.",
        gamefile=str(gamefile),
        success=success,
        steps=[
            TrajectoryStep(
                observation="",
                admissible_actions=["look"],
                thought="test",
                action="look",
                valid_format=True,
                admissible=True,
                next_observation="",
                score=0.0,
                done=i == actions - 1,
                won=success and i == actions - 1,
            )
            for i in range(actions)
        ],
    )


class UnifiedFeedbackControllerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        self.root = Path(temp.name)

        capabilities = SimpleNamespace(
            memory_info=SimpleNamespace(
                version="v000",
                memory_hash="mock_memory_hash",
            ),
            skill_id=None,
            skill_mode=None,
        )

        self.controller = UnifiedFeedbackController(
            capabilities,
            self.root / "events_output",
        )

        self.game = GameExample(
            id="train:mock_game",
            split="train",
            game_file=self.root / "game.tw-pddl",
            task_type=TASK_TYPE,
        )

        self.router = FeedbackRouter({
            TASK_TYPE: [10, 11, 12, 13, 14],
        })

        self.trace_path = self.root / "trace.json"

    def process(self, actions, success, phase="learning"):
        make_trace(
            self.game.game_file,
            actions,
            success,
        ).save_json(str(self.trace_path))

        return self.controller.process_saved_episode(
            episode_id="test_001",
            game=self.game,
            trajectory_path=self.trace_path,
            router=self.router,
            phase=phase,
        )

    def test_failure_enters_memory_queue(self):
        event = self.process(50, False)

        self.assertEqual(
            event["decision"]["route"], "MEMORY_REVIEW"
        )
        self.assertEqual(
            len(self.controller.review_queue()), 1
        )

    def test_inefficient_success_enters_skill_queue(self):
        event = self.process(30, True)

        self.assertEqual(
            event["decision"]["route"], "SKILL_REVIEW"
        )
        self.assertIsNone(event["active_skill_id"])

    def test_normal_success_not_queued(self):
        event = self.process(12, True)

        self.assertEqual(
            event["decision"]["route"], "ARCHIVE"
        )
        self.assertEqual(
            self.controller.review_queue(), []
        )

    def test_changed_episode_cannot_overwrite(self):
        self.process(12, True)

        with self.assertRaisesRegex(
            RuntimeError, "Conflicting saved event"
        ):
            self.process(30, True)

    def test_holdout_never_enters_review_queue(self):
        event = self.process(
            50,
            False,
            phase="holdout",
        )

        self.assertEqual(
            event["decision"]["route"], "IGNORE"
        )
        self.assertEqual(
            self.controller.review_queue(), []
        )


if __name__ == "__main__":
    unittest.main()
