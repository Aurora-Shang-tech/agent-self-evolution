
import unittest

from src.core.trajectory import TrajectoryStep
from src.memory.state_tracker import TaskStateTracker


def make_step(
    action: str,
    feedback: str,
    *,
    admissible: bool = True,
    valid_format: bool = True,
) -> TrajectoryStep:
    return TrajectoryStep(
        observation="",
        admissible_actions=[action],
        thought="",
        action=action,
        valid_format=valid_format,
        admissible=admissible,
        next_observation=feedback,
        score=0.0,
        done=False,
        won=False,
    )


class TestTaskStateTracker(unittest.TestCase):

    def test_task_parsing(self):
        cases = [
            ("put a clean egg in microwave.", "clean", "egg"),
            ("clean some potato and put it in microwave.", "clean", "potato"),
            ("put a cool cup in microwave.", "cool", "cup"),
            ("put a cold lettuce in countertop.", "cool", "lettuce"),
            ("put a hot tomato in countertop.", "heat", "tomato"),
            ("put a heated potato in drawer.", "heat", "potato"),
            ("heat some potato and put it in drawer.", "heat", "potato"),
        ]

        for task, operation, target in cases:
            with self.subTest(task=task):
                tracker = TaskStateTracker(task)
                self.assertEqual(tracker.operation, operation)
                self.assertEqual(tracker.target, target)

    def test_cool_then_heat(self):
        tracker = TaskStateTracker(
            "cool some potato and put it in microwave."
        )

        tracker.observe(make_step(
            "cool potato 1 with fridge 1",
            "You cool the potato 1 using the fridge 1.",
        ))

        self.assertTrue(tracker.confirmed)
        self.assertTrue(tracker.is_confirmed("potato 1"))

        tracker.observe(make_step(
            "heat potato 1 with microwave 1",
            "You heat the potato 1 using the microwave 1.",
        ))

        self.assertFalse(tracker.confirmed)
        self.assertFalse(tracker.is_confirmed("potato 1"))
        self.assertTrue(tracker.action_observed)

    def test_cool_heat_cool(self):
        tracker = TaskStateTracker(
            "put a cool potato in microwave."
        )

        tracker.observe(make_step(
            "cool potato 1 with fridge 1",
            "You cool the potato 1 using the fridge 1.",
        ))

        tracker.observe(make_step(
            "heat potato 1 with microwave 1",
            "You heat the potato 1 using the microwave 1.",
        ))

        tracker.observe(make_step(
            "cool potato 1 with fridge 1",
            "You cool the potato 1 using the fridge 1.",
        ))

        self.assertTrue(tracker.confirmed)
        self.assertEqual(
            tracker.object_states["potato 1"],
            {"cool"},
        )

    def test_object_id_isolation(self):
        tracker = TaskStateTracker(
            "put a cool potato in microwave."
        )

        tracker.observe(make_step(
            "cool potato 1 with fridge 1",
            "You cool the potato 1 using the fridge 1.",
        ))

        tracker.observe(make_step(
            "heat potato 2 with microwave 1",
            "You heat the potato 2 using the microwave 1.",
        ))

        self.assertTrue(tracker.is_confirmed("potato 1"))
        self.assertFalse(tracker.is_confirmed("potato 2"))
        self.assertTrue(tracker.confirmed)

    def test_missing_confirmation(self):
        tracker = TaskStateTracker(
            "put a clean egg in microwave."
        )

        tracker.observe(make_step(
            "clean egg 1 with sinkbasin 1",
            "Nothing happens.",
        ))

        self.assertTrue(tracker.action_observed)
        self.assertFalse(tracker.confirmed)

    def test_invalid_action(self):
        tracker = TaskStateTracker(
            "put a clean egg in microwave."
        )

        tracker.observe(make_step(
            "clean egg 1 with sinkbasin 1",
            "You clean the egg 1 using the sinkbasin 1.",
            admissible=False,
        ))

        self.assertFalse(tracker.action_observed)
        self.assertFalse(tracker.confirmed)

    def test_unrelated_operation(self):
        tracker = TaskStateTracker(
            "put a clean egg in microwave."
        )

        tracker.observe(make_step(
            "cool egg 1 with fridge 1",
            "You cool the egg 1 using the fridge 1.",
        ))

        self.assertFalse(tracker.confirmed)

        tracker.observe(make_step(
            "clean egg 1 with sinkbasin 1",
            "You clean the egg 1 using the sinkbasin 1.",
        ))

        self.assertTrue(tracker.confirmed)

    def test_no_state_requirement(self):
        tracker = TaskStateTracker(
            "find two glassbottle and put them in shelf."
        )

        self.assertIsNone(tracker.operation)
        self.assertIsNone(tracker.target)
        self.assertTrue(tracker.requirement_met)
        self.assertFalse(tracker.confirmed)


if __name__ == "__main__":
    unittest.main()

