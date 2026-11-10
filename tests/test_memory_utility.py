
import unittest

from scripts.evaluation.verify_memory_utility import (
    OneLessonStore,
    exact_mcnemar,
    paired_stats,
)

from src.memory.experience import Experience


class UtilityTests(unittest.TestCase):

    def test_exact_mcnemar(self):
        self.assertEqual(
            exact_mcnemar(0, 0),
            1.0,
        )

        self.assertEqual(
            exact_mcnemar(3, 0),
            0.25,
        )

        self.assertEqual(
            exact_mcnemar(1, 1),
            1.0,
        )

    def test_one_lesson_store(self):
        experience = Experience(
            "test",
            "Use the sink to clean the egg",
            "clean an egg",
        )

        store = OneLessonStore(
            experience
        )

        self.assertEqual(
            store.retrieve(
                "any task",
                3,
            ),
            [experience],
        )

        with self.assertRaises(
            ValueError
        ):
            store.retrieve(
                "any task",
                0,
            )

    def test_paired_stats(self):

        def record(
            success,
            steps,
        ):
            return {
                "success": success,
                "decision_steps": steps,
            }

        records = [
            {
                "control": record(
                    False,
                    50,
                ),
                "with_lesson": record(
                    True,
                    8,
                ),
            },
            {
                "control": record(
                    True,
                    10,
                ),
                "with_lesson": record(
                    False,
                    50,
                ),
            },
            {
                "control": record(
                    True,
                    12,
                ),
                "with_lesson": record(
                    True,
                    7,
                ),
            },
        ]

        result = paired_stats(
            records
        )

        self.assertEqual(
            result["improved"],
            1,
        )

        self.assertEqual(
            result["regressed"],
            1,
        )

        self.assertEqual(
            result["net"],
            0,
        )

        self.assertEqual(
            result["control_successes"],
            2,
        )

        self.assertEqual(
            result["with_lesson_successes"],
            2,
        )

        self.assertAlmostEqual(
            result["avg_step_delta"],
            (
                8 - 50
                + 50 - 10
                + 7 - 12
            ) / 3,
        )


if __name__ == "__main__":
    unittest.main()

