import json
import tempfile
import unittest
from pathlib import Path

from src.evolution.persistent_llm_budget import (
    PersistentLLMBudget,
)


class FakeLLM:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def chat(self, messages):
        self.calls += 1

        if self.fail:
            raise ConnectionError("Simulated API failure")

        return "Mock model response"


class PersistentLLMBudgetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        self.root = Path(temp.name)

        self.job = {
            "job_id": "batch_07",
            "game_id": "train:mock",
            "route": "MEMORY_REVIEW",
            "source_scope": "ONLINE_TRAIN_DEVELOPMENT",
            "trajectory_sha256": "trajectory_hash",
            "event_sha256": "event_hash",
        }

        self.llm = FakeLLM()

    def create_budget(self, limit=2, llm=None, job=None):
        return PersistentLLMBudget(
            self.root,
            job=job or self.job,
            model="mock-model",
            memory_version="v000",
            memory_sha256="memory_hash",
            active_skill_id=None,
            max_calls=limit,
            client_factory=lambda: (
                llm if llm is not None else self.llm
            ),
        )

    def test_budget_persists_across_instances(self):
        first = self.create_budget()
        first.chat([])

        second = self.create_budget()
        self.assertEqual(second.used_calls, 1)
        self.assertEqual(second.remaining_calls, 1)

        second.chat([])

        third = self.create_budget()
        self.assertEqual(third.used_calls, 2)

        with self.assertRaisesRegex(
            RuntimeError, "budget exhausted"
        ):
            third.chat([])

        self.assertEqual(self.llm.calls, 2)

    def test_failed_request_still_consumes_budget(self):
        failed = FakeLLM(fail=True)
        budget = self.create_budget(limit=1, llm=failed)

        with self.assertRaises(ConnectionError):
            budget.chat([])

        self.assertEqual(budget.used_calls, 1)

        resumed = self.create_budget(limit=1)

        with self.assertRaisesRegex(
            RuntimeError, "budget exhausted"
        ):
            resumed.chat([])

        self.assertEqual(self.llm.calls, 0)

    def test_changed_provenance_rejected(self):
        self.create_budget()

        changed = dict(self.job)
        changed["trajectory_sha256"] = "changed"

        with self.assertRaises(RuntimeError):
            self.create_budget(job=changed)

    def test_corrupted_reservation_rejected(self):
        budget = self.create_budget()
        budget.chat([])

        path = (
            self.root / "batch_07"
            / "call_0001.json"
        )

        path.write_text(
            json.dumps({"index": 999}),
            encoding="utf-8",
        )

        with self.assertRaises(RuntimeError):
            self.create_budget()


if __name__ == "__main__":
    unittest.main()
