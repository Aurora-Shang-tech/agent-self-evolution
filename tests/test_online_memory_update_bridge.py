import json
import tempfile
import unittest
from pathlib import Path

from scripts.memory.run_evolution import memory_hash
from src.evolution.online_memory_update_bridge import (
    OnlineMemoryUpdateBridge,
)
from src.memory.updater import LessonCandidate, MemoryItem


class FakeLLM:
    def __init__(self, operation="ADD"):
        self.operation = operation
        self.calls = 0

    def chat(self, messages):
        self.calls += 1

        if self.operation == "KEEP":
            return json.dumps({
                "operation": "KEEP",
                "target_id": None,
                "new_lesson": "",
                "reason": "Insufficient evidence for modification.",
                "evidence_refs": [],
            })

        return json.dumps({
            "operation": "ADD",
            "target_id": None,
            "new_lesson": (
                "Search task-relevant locations before "
                "checking every container sequentially."
            ),
            "reason": "A distinct search strategy.",
            "evidence_refs": [
                "baseline:critical",
                "counterfactual:replacement",
            ],
        })


class OnlineMemoryUpdateBridgeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        self.output = (
            Path(temp.name) / "update_proposal.json"
        )

        self.job = {
            "job_id": "batch_07",
            "game_id": "train:test_game",
            "route": "MEMORY_REVIEW",
            "source_scope": "ONLINE_TRAIN_DEVELOPMENT",
            "trajectory_sha256": "trajectory_hash",
            "event_sha256": "event_hash",
        }

        self.memory = [
            MemoryItem(
                "base_0000",
                "Verify the exact object type before acting.",
                "find a cup",
            ),
        ]

        self.candidate = LessonCandidate(
            decision="CANDIDATE",
            lesson=(
                "Search task-relevant locations before "
                "checking every container sequentially."
            ),
            reason="Counterfactual continuation succeeded.",
            limitations="Only one local comparison.",
            evidence_refs=(
                "baseline:critical",
                "counterfactual:replacement",
            ),
            source_task="find a plate",
            critical_step=1,
            original_action="open cabinet 1",
            replacement_action="go to shelf 1",
            evidence={
                "baseline:critical": {
                    "action": "open cabinet 1",
                },
                "counterfactual:replacement": {
                    "action": "go to shelf 1",
                },
            },
        )

    def test_add_proposal_stages_without_activation(self):
        llm = FakeLLM()
        before = memory_hash(self.memory)

        record = OnlineMemoryUpdateBridge(llm).process(
            self.job,
            self.candidate,
            self.memory,
            self.output,
        )

        self.assertEqual(llm.calls, 1)
        self.assertEqual(record["proposal"]["operation"], "ADD")
        self.assertEqual(
            record["staged"]["status"],
            "PENDING_VALIDATION",
        )
        self.assertEqual(
            len(record["staged"]["candidate_memory"]),
            2,
        )
        self.assertEqual(memory_hash(self.memory), before)
        self.assertFalse(record["active_memory_changed"])

    def test_cached_proposal_avoids_new_llm_call(self):
        first = FakeLLM()

        OnlineMemoryUpdateBridge(first).process(
            self.job,
            self.candidate,
            self.memory,
            self.output,
        )

        second = FakeLLM()

        record = OnlineMemoryUpdateBridge(second).process(
            self.job,
            self.candidate,
            self.memory,
            self.output,
        )

        self.assertEqual(second.calls, 0)
        self.assertEqual(record["proposal"]["operation"], "ADD")

    def test_keep_produces_no_memory_change(self):
        llm = FakeLLM("KEEP")

        record = OnlineMemoryUpdateBridge(llm).process(
            self.job,
            self.candidate,
            self.memory,
            self.output,
        )

        self.assertEqual(
            record["staged"]["status"],
            "NO_CHANGE",
        )
        self.assertEqual(
            record["staged"]["parent_memory_sha256"],
            record["staged"]["candidate_memory_sha256"],
        )

    def test_tampering_is_rejected(self):
        OnlineMemoryUpdateBridge(FakeLLM()).process(
            self.job,
            self.candidate,
            self.memory,
            self.output,
        )

        record = json.loads(
            self.output.read_text(encoding="utf-8")
        )

        record["proposal"]["new_lesson"] = "Tampered lesson."

        self.output.write_text(
            json.dumps(record),
            encoding="utf-8",
        )

        with self.assertRaises(
            (RuntimeError, ValueError)
        ):
            OnlineMemoryUpdateBridge(FakeLLM()).process(
                self.job,
                self.candidate,
                self.memory,
                self.output,
            )


if __name__ == "__main__":
    unittest.main()
