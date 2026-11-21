import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.memory.run_evolution import memory_hash
from src.evolution.controller import EvolutionController
from src.evolution.online_memory_gate import stage_memory_candidate
from src.evolution.online_memory_precheck import (
    digest,
    run_retrieval_precheck,
)
from src.evolution.online_memory_version_bridge import (
    stage_online_memory_version,
)
from src.memory.updater import (
    LessonCandidate,
    MemoryItem,
    UpdateProposal,
)


class OnlineMemoryPrecheckTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)

        root = Path(tmp.name)
        self.parent = root / "active"
        self.staging = root / "staging"
        self.manifest = root / "validation.json"

        memory = [
            MemoryItem(
                "base_0000",
                "Verify target type.",
                "put a cup on table",
            )
        ]

        writer = EvolutionController.__new__(EvolutionController)
        writer.home = self.parent
        writer.save_version("v000", None, 0, memory)

        (self.parent / "state.json").write_text(
            json.dumps({
                "active_version": "v000",
                "completed_rounds": 0,
            }),
            encoding="utf-8",
        )

        job = {
            "job_id": "batch_07",
            "game_id": "train:learning",
            "route": "MEMORY_REVIEW",
            "source_scope": "ONLINE_TRAIN_DEVELOPMENT",
            "trajectory_sha256": "trace_hash",
            "event_sha256": "event_hash",
        }

        lesson = (
            "Search likely locations before exploring every cabinet."
        )

        candidate = LessonCandidate(
            decision="CANDIDATE",
            lesson=lesson,
            reason="Verified local contrast.",
            limitations="Single task only.",
            evidence_refs=(
                "baseline:critical",
                "counterfactual:replacement",
            ),
            source_task="put a cool pan in stoveburner",
            critical_step=0,
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

        proposal = UpdateProposal(
            operation="ADD",
            target_id=None,
            new_lesson=lesson,
            reason="New guidance.",
            evidence_refs=candidate.evidence_refs,
            memory_snapshot_sha256=memory_hash(memory),
        )

        staged = stage_memory_candidate(
            job, candidate, proposal, memory
        )

        stage_online_memory_version(
            job,
            staged,
            parent_run_dir=self.parent,
            staging_root=self.staging,
        )

        games = []

        for i in range(2):
            file = root / f"game_{i}.json"
            file.write_text("{}", encoding="utf-8")

            games.append({
                "id": f"train:validation_{i}",
                "split": "train",
                "game_file": str(file),
                "task_type": "pick_cool_then_place_in_recep",
            })

        self.manifest.write_text(
            json.dumps({"games": games}),
            encoding="utf-8",
        )

    def run_check(self):
        return run_retrieval_precheck(
            job_id="batch_07",
            parent_run_dir=self.parent,
            staging_root=self.staging,
            validation_manifest_path=self.manifest,
            expected_manifest_sha256=digest(self.manifest),
            excluded_game_ids={"train:other_learning"},
        )

    @patch.object(EvolutionController, "retrieval_changes")
    def test_changed_retrieval_allows_evaluation(self, compare):
        compare.return_value = 1

        result = self.run_check()

        self.assertEqual(
            result["status"],
            "READY_FOR_PAIRED_EVAL",
        )
        self.assertEqual(result["changed_retrieval_games"], 1)
        self.assertFalse(result["active_memory_changed"])

    @patch.object(EvolutionController, "retrieval_changes")
    def test_unchanged_retrieval_skips_evaluation(self, compare):
        compare.return_value = 0

        result = self.run_check()

        self.assertEqual(
            result["status"],
            "SKIP_UNCHANGED_TOP3",
        )

    @patch.object(EvolutionController, "retrieval_changes")
    def test_learning_validation_overlap_rejected(self, compare):
        data = json.loads(self.manifest.read_text())
        data["games"][0]["id"] = "train:learning"

        self.manifest.write_text(
            json.dumps(data),
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            self.run_check()

        compare.assert_not_called()

    @patch.object(EvolutionController, "retrieval_changes")
    def test_tampered_candidate_snapshot_rejected(self, compare):
        path = (
            self.staging / "batch_07"
            / "versions" / "v001.json"
        )

        path.write_text("{}", encoding="utf-8")

        with self.assertRaises(RuntimeError):
            self.run_check()

        compare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
