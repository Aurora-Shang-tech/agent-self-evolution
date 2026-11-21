import unittest

from scripts.memory.run_evolution import memory_hash
from src.evolution.online_memory_gate import (
    assess_memory_candidate,
    stage_memory_candidate,
)
from src.memory.updater import (
    LessonCandidate,
    MemoryItem,
    UpdateProposal,
)


class OnlineMemoryGateTests(unittest.TestCase):
    def setUp(self):
        self.parent = [
            MemoryItem(
                "base_0000",
                "Check the correct object type before acting.",
                "find a cup",
            )
        ]

        self.refs = (
            "baseline:critical",
            "counterfactual:replacement",
        )

        self.candidate = LessonCandidate(
            decision="CANDIDATE",
            lesson=(
                "Check likely object locations before "
                "searching every container sequentially."
            ),
            reason="Successful counterfactual evidence.",
            limitations="Only a local trajectory comparison.",
            evidence_refs=self.refs,
            source_task="find a plate",
            critical_step=1,
            original_action="open cabinet 1",
            replacement_action="go to shelf 1",
            evidence={
                "baseline:critical": {"action": "open cabinet 1"},
                "counterfactual:replacement": {
                    "action": "go to shelf 1"
                },
            },
        )

        self.job = {
            "job_id": "batch_07",
            "game_id": "train:example_07",
            "route": "MEMORY_REVIEW",
            "source_scope": "ONLINE_TRAIN_DEVELOPMENT",
            "trajectory_sha256": "trajectory_hash",
            "event_sha256": "event_hash",
        }

        self.proposal = UpdateProposal(
            operation="ADD",
            target_id=None,
            new_lesson=self.candidate.lesson,
            reason="Distinct guidance.",
            evidence_refs=self.refs,
            memory_snapshot_sha256=memory_hash(self.parent),
        )

        self.ids = [
            "train:validation_1",
            "train:validation_2",
        ]

    def staged(self):
        return stage_memory_candidate(
            self.job,
            self.candidate,
            self.proposal,
            self.parent,
        )

    def test_stage_add_preserves_active_memory(self):
        original = memory_hash(self.parent)
        staged = self.staged()

        self.assertEqual(staged["status"], "PENDING_VALIDATION")
        self.assertEqual(len(staged["candidate_memory"]), 2)
        self.assertEqual(memory_hash(self.parent), original)
        self.assertFalse(staged["active_memory_changed"])

    def test_stale_memory_rejected(self):
        changed = [
            MemoryItem(
                "base_0000",
                "Changed existing memory.",
                "find a cup",
            )
        ]

        with self.assertRaisesRegex(ValueError, "Stale"):
            stage_memory_candidate(
                self.job,
                self.candidate,
                self.proposal,
                changed,
            )

    def test_holdout_cannot_stage_candidate(self):
        job = dict(self.job)
        job["source_scope"] = "FINAL_HOLDOUT"

        with self.assertRaises(ValueError):
            stage_memory_candidate(
                job,
                self.candidate,
                self.proposal,
                self.parent,
            )

    def test_accepts_genuine_success_gain(self):
        staged = self.staged()

        baseline = [
            {"game_id": self.ids[0], "success": False,
             "decision_steps": 30},
            {"game_id": self.ids[1], "success": True,
             "decision_steps": 12},
        ]

        candidate = [
            {"game_id": self.ids[0], "success": True,
             "decision_steps": 12},
            {"game_id": self.ids[1], "success": True,
             "decision_steps": 12},
        ]

        decision = assess_memory_candidate(
            staged,
            baseline,
            candidate,
            validation_game_ids=self.ids,
            validation_manifest_sha256="a" * 64,
        )

        self.assertEqual(
            decision["status"],
            "ELIGIBLE_FOR_PROMOTION",
        )
        self.assertEqual(decision["result"]["improved"], 1)
        self.assertFalse(decision["active_memory_changed"])

    def test_rejects_unimproved_candidate(self):
        staged = self.staged()

        baseline = [
            {"game_id": game_id, "success": True,
             "decision_steps": 10}
            for game_id in self.ids
        ]

        candidate = [
            {"game_id": game_id, "success": True,
             "decision_steps": 15}
            for game_id in self.ids
        ]

        decision = assess_memory_candidate(
            staged,
            baseline,
            candidate,
            validation_game_ids=self.ids,
            validation_manifest_sha256="a" * 64,
        )

        self.assertEqual(decision["status"], "REJECTED")

    def test_training_validation_overlap_rejected(self):
        staged = self.staged()

        with self.assertRaisesRegex(ValueError, "overlap"):
            assess_memory_candidate(
                staged,
                [],
                [],
                validation_game_ids=[
                    self.job["game_id"],
                    "train:validation_2",
                ],
                validation_manifest_sha256="a" * 64,
            )


if __name__ == "__main__":
    unittest.main()
