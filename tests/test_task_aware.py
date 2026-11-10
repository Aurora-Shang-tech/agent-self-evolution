
import unittest

from src.memory.experience import Experience
from src.memory.task_aware import (
    TaskAwareExperienceStore,
    parse_lesson,
    parse_task,
    rejection_reason,
)


class TaskAwareTest(unittest.TestCase):

    def test_task_parsing(self):
        task = parse_task(
            "put a hot cup in cabinet."
        )

        self.assertEqual(
            task.operation,
            "heat",
        )

        self.assertEqual(
            task.quantity,
            1,
        )

        self.assertEqual(
            task.destination,
            "cabinet",
        )

        self.assertEqual(
            task.target_object,
            "cup",
        )

        self.assertEqual(
            parse_task(
                "put two apples in drawer"
            ).quantity,
            2,
        )

    def test_multiple_object_rule(self):
        scope = parse_lesson(
            "When a task requires collecting "
            "multiple specific items, "
            "gather all items first."
        )

        reason = rejection_reason(
            parse_task(
                "put a hot cup in cabinet"
            ),
            scope,
        )

        self.assertEqual(
            reason,
            "requires_multiple_objects",
        )

    def test_operation_mismatch(self):
        scope = parse_lesson(
            "When a task requires placing "
            "a clean object, "
            "clean it before placement."
        )

        self.assertEqual(
            rejection_reason(
                parse_task(
                    "put a hot cup in cabinet"
                ),
                scope,
            ),
            "operation_mismatch",
        )

        self.assertIsNone(
            rejection_reason(
                parse_task(
                    "put a clean cup in cabinet"
                ),
                scope,
            )
        )

    def test_generic_state_example(self):
        lesson = (
            "When a task requires an object "
            "to be in a specific state "
            "(e.g., 'cool'), "
            "identify and utilize the appropriate "
            "environmental tool "
            "(e.g., a fridge) "
            "to modify the object's state "
            "before completing placement."
        )

        scope = parse_lesson(
            lesson
        )

        self.assertEqual(
            scope.operations,
            (),
        )

        self.assertIsNone(
            rejection_reason(
                parse_task(
                    "put a cool apple in garbagecan"
                ),
                scope,
            )
        )

        self.assertIsNone(
            rejection_reason(
                parse_task(
                    "put a hot cup in cabinet"
                ),
                scope,
            )
        )

    def test_object_specific_rule(self):
        scope = parse_lesson(
            "When a task requires cleaning "
            "a generic item like 'cloth', "
            "prioritize objects explicitly "
            "labeled as 'cloth' over 'towel'."
        )

        self.assertEqual(
            scope.specific_object,
            "cloth",
        )

        self.assertEqual(
            rejection_reason(
                parse_task(
                    "put a clean mug in coffeemachine"
                ),
                scope,
            ),
            "object_mismatch",
        )

        self.assertIsNone(
            rejection_reason(
                parse_task(
                    "put a clean cloth in countertop"
                ),
                scope,
            )
        )

    def test_examples_do_not_define_operation(self):
        scope = parse_lesson(
            "When a task requires placing "
            "an object into a container, "
            "prioritize the physical placement "
            "action (e.g., 'move object to container') "
            "over state-modification actions "
            "(e.g., 'cool object with container')."
        )

        self.assertEqual(
            scope.operations,
            (),
        )

        self.assertIsNone(
            rejection_reason(
                parse_task(
                    "put a hot cup in cabinet"
                ),
                scope,
            )
        )

    def test_prior_failure_condition(self):
        scope = parse_lesson(
            "When an action to move "
            "to a location fails because "
            "the target is ambiguous, "
            "consult admissible actions."
        )

        self.assertEqual(
            rejection_reason(
                parse_task(
                    "put a clean egg in microwave"
                ),
                scope,
            ),
            "requires_prior_failure",
        )

    def test_heat_rule_not_clean_rule(self):
        experiences = [
            Experience(
                "test",
                "When a task requires placing "
                "a clean object, "
                "clean it before placement.",
                "put a clean cup in cabinet",
            ),
            Experience(
                "test",
                "When a task requires collecting "
                "multiple specific items, "
                "gather both first.",
                "put two cups in cabinet",
            ),
            Experience(
                "test",
                "When using a closed appliance "
                "to heat an item, "
                "open the appliance first.",
                "put a hot cup in cabinet",
            ),
        ]

        store = TaskAwareExperienceStore(
            experiences,
            pool_size=3,
        )

        result = store.explain(
            "put a hot cup in cabinet",
            top_k=2,
        )

        selected = [
            item["index"]
            for item in result["selected"]
        ]

        self.assertIn(
            2,
            selected,
        )

        self.assertNotIn(
            0,
            selected,
        )

        self.assertNotIn(
            1,
            selected,
        )

    def test_no_filter_preserves_baseline(self):
        experiences = [
            Experience(
                "test",
                "Always inspect the destination.",
                "put a hot cup in cabinet",
            ),
            Experience(
                "test",
                "Always examine the inventory.",
                "put a cool apple in garbagecan",
            ),
            Experience(
                "test",
                "Always check the list "
                "of available actions.",
                "put a clean mug in cabinet",
            ),
        ]

        store = TaskAwareExperienceStore(
            experiences,
            pool_size=3,
        )

        result = store.explain(
            "put a hot cup in cabinet",
            top_k=3,
        )

        baseline = [
            item["index"]
            for item in result["baseline"]
        ]

        selected = [
            item["index"]
            for item in result["selected"]
        ]

        self.assertEqual(
            baseline,
            selected,
        )


if __name__ == "__main__":
    unittest.main()

