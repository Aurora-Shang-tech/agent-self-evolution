import tempfile
import unittest
from pathlib import Path

from src.memory.experience import Experience
from src.memory.store import ExperienceStore


class ExperienceStoreTests(unittest.TestCase):
    def setUp(self):
        self.originals = [
            Experience("old", "Heat objects carefully.", "put a hot cup in shelf."),
            Experience("old", "Clean objects first.", "put a clean apple in cabinet."),
            Experience("old", "Find the lamp.", "look at pillow under the desklamp."),
            Experience("old", "Cool objects first.", "put a cool potato in fridge."),
        ]

        self.new = Experience(
            "memory_evolution",
            "Prioritize plausible locations for target objects.",
            "heat some plate and put it in countertop.",
        )

    def make_store(self):
        store = ExperienceStore()
        for item in self.originals:
            store.add(item)
        return store

    def test_original_memory_unchanged(self):
        store = self.make_store()
        query = "put a hot mug in shelf."

        before = store.retrieve(query, 3)
        self.assertEqual(len(before), 3)

        store.add(
            Experience(
                "new",
                "A different task type.",
                "look at bowl under the desklamp.",
            ),
            is_new=True,
        )

        after = store.retrieve(query, 3)
        self.assertEqual(before, after)

    def test_new_same_type_takes_one_slot(self):
        store = self.make_store()
        query = "put a hot mug in shelf."

        before = store.retrieve(query, 3)
        store.add(self.new, is_new=True)
        after = store.retrieve(query, 3)

        self.assertEqual(after[:2], before[:2])
        self.assertEqual(after[2], self.new)

    def test_unrelated_task_type_not_injected(self):
        store = self.make_store()
        query = "put a clean apple in cabinet."

        before = store.retrieve(query, 3)
        store.add(self.new, is_new=True)
        after = store.retrieve(query, 3)

        self.assertEqual(before, after)
        self.assertNotIn(self.new, after)

    def test_save_load_preserves_new_flag(self):
        store = self.make_store()
        store.add(self.new, is_new=True)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            store.save(path)

            restored = ExperienceStore()
            restored.load(path)

            self.assertEqual(len(restored), 5)
            self.assertIn(
                self.new,
                restored.retrieve("put a hot mug in shelf.", 3),
            )

    def test_invalid_top_k(self):
        store = self.make_store()

        with self.assertRaises(ValueError):
            store.retrieve("any task", 0)


if __name__ == "__main__":
    unittest.main()
