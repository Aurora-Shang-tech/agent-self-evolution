import json
import tempfile
import unittest
from pathlib import Path

from scripts.memory.run_evolution import memory_hash
from src.evolution.controller import EvolutionController
from src.evolution.online_memory_gate import stage_memory_candidate
from src.evolution.online_memory_version_bridge import (
    stage_online_memory_version,
)
from src.evolution.version_store import load_memory_version
from src.memory.updater import (
    LessonCandidate,
    MemoryItem,
    UpdateProposal,
)


class OnlineMemoryVersionBridgeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        root = Path(temp.name)
        self.parent_root = root / "active"
        self.staging_root = root / "staging"

        self.memory = [
            MemoryItem(
                "base_0000",
                "Verify the exact object type before acting.",
                "find a cup",
            )
        ]

        controller = EvolutionController.__new__(
            EvolutionController
        )
        controller.home = self.parent_root
        controller.save_version(
            "v000", None, 0, self.memory
        )

        self.state_path = self.parent_root / "state.json"
        self.state_path.write_text(
            json.dumps({
                "active_version": "v000",
                "completed_rounds": 0,
            }),
            encoding="utf-8",
        )

        self.job = {
            "job_id": "batch_07",
            "game_id": "train:mock",
            "route": "MEMORY_REVIEW",
            "source_scope": "ONLINE_TRAIN_DEVELOPMENT",
            "trajectory_sha256": "trajectory_hash",
            "event_sha256": "event_hash",
        }

        lesson = (
            "Search likely object locations before "
            "checking all containers sequentially."
        )

        self.candidate = LessonCandidate(
            decision="CANDIDATE",
            lesson=lesson,
            reason="Local counterfactual improvement.",
            limitations="Only one task comparison.",
            evidence_refs=(
                "baseline:critical",
                "counterfactual:replacement",
            ),
            source_task="find a plate",
            critical_step=0,
            original_action="open cabinet 1",
            replacement_action="go to shelf 1",
            evidence={
                "baseline:critical": {
                    "action": "open cabinet 1"
                },
                "counterfactual:replacement": {
                    "action": "go to shelf 1"
                },
            },
        )

        proposal = UpdateProposal(
            operation="ADD",
            target_id=None,
            new_lesson=lesson,
            reason="New guidance.",
            evidence_refs=self.candidate.evidence_refs,
            memory_snapshot_sha256=memory_hash(self.memory),
        )

        self.staged = stage_memory_candidate(
            self.job,
            self.candidate,
            proposal,
            self.memory,
        )

    def run_stage(self, staged=None, staging_root=None):
        return stage_online_memory_version(
            self.job,
            staged if staged is not None else self.staged,
            parent_run_dir=self.parent_root,
            staging_root=(
                staging_root
                if staging_root is not None
                else self.staging_root
            ),
        )

    def test_candidate_snapshot_and_resume(self):
        original_state = self.state_path.read_bytes()

        first = self.run_stage()
        second = self.run_stage()

        self.assertEqual(first, second)
        self.assertEqual(first["candidate_version"], "v001")
        self.assertEqual(first["status"], "PENDING_VALIDATION")

        _, info = load_memory_version(
            self.staging_root / "batch_07",
            "v001",
        )

        self.assertEqual(info.memory_size, 2)
        self.assertEqual(info.parent, "v000")
        self.assertEqual(self.state_path.read_bytes(), original_state)
        self.assertFalse(first["active_memory_changed"])

    def test_stale_parent_rejected(self):
        staged = dict(self.staged)
        staged["parent_memory_sha256"] = "invalid"

        with self.assertRaisesRegex(ValueError, "parent memory"):
            self.run_stage(staged=staged)

    def test_tampered_snapshot_rejected(self):
        self.run_stage()

        path = (
            self.staging_root / "batch_07"
            / "versions" / "v001.json"
        )

        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["memory"][0]["lesson"] = "Tampered."

        path.write_text(
            json.dumps(raw),
            encoding="utf-8",
        )

        with self.assertRaises(RuntimeError):
            self.run_stage()

    def test_cannot_write_inside_active_run(self):
        with self.assertRaisesRegex(
            ValueError, "inside active memory run"
        ):
            self.run_stage(
                staging_root=self.parent_root / "staging"
            )


if __name__ == "__main__":
    unittest.main()
