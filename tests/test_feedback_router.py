import unittest

from src.core.trajectory import Trajectory, TrajectoryStep
from src.evolution.feedback_router import FeedbackRouter


TASK_TYPE = "pick_heat_then_place_in_recep"


def make_trace(actions, success, invalid_indices=()):
    steps = []

    for index in range(actions):
        valid = index not in invalid_indices

        steps.append(
            TrajectoryStep(
                observation="",
                admissible_actions=["look"],
                thought="test",
                action="look" if valid else "",
                valid_format=valid,
                admissible=valid,
                next_observation="",
                score=0.0,
                done=index == actions - 1,
                won=success and index == actions - 1,
            )
        )

    return Trajectory(
        task="heat some egg and put it in sidetable.",
        gamefile="/tmp/mock_game",
        success=success,
        steps=steps,
    )


class FeedbackRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = FeedbackRouter({
            TASK_TYPE: [10, 11, 12, 13, 14],
        })

    def test_failure_routes_to_memory(self):
        trace = make_trace(50, False)

        decision = self.router.route(trace, TASK_TYPE)

        self.assertEqual(decision.route, "MEMORY_REVIEW")
        self.assertEqual(decision.environment_actions, 50)

    def test_inefficient_success_routes_to_skill(self):
        trace = make_trace(30, True)

        decision = self.router.route(trace, TASK_TYPE)

        self.assertEqual(decision.route, "SKILL_REVIEW")
        self.assertEqual(decision.reference_median, 12.0)

    def test_normal_success_is_archived(self):
        trace = make_trace(15, True)

        decision = self.router.route(trace, TASK_TYPE)

        self.assertEqual(decision.route, "ARCHIVE")

    def test_insufficient_reference_is_archived(self):
        router = FeedbackRouter({
            TASK_TYPE: [10, 12],
        })

        decision = router.route(
            make_trace(40, True),
            TASK_TYPE,
        )

        self.assertEqual(decision.route, "ARCHIVE")

    def test_unknown_type_is_archived(self):
        decision = self.router.route(
            make_trace(40, True),
            "unknown_task_type",
        )

        self.assertEqual(decision.route, "ARCHIVE")
        self.assertEqual(decision.reference_samples, 0)

    def test_holdout_never_enters_learning(self):
        decision = self.router.route(
            make_trace(50, False),
            TASK_TYPE,
            phase="holdout",
        )

        self.assertEqual(decision.route, "IGNORE")

    def test_count_environment_actions_not_decisions(self):
        trace = make_trace(
            5, True, invalid_indices=(0, 1)
        )

        decision = self.router.route(trace, TASK_TYPE)

        self.assertEqual(decision.environment_actions, 3)
        self.assertEqual(decision.route, "ARCHIVE")


if __name__ == "__main__":
    unittest.main()
