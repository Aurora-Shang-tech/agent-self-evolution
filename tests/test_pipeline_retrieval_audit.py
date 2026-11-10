"""Offline regression tests for Hybrid/Utility retrieval composition audits."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.audit_pipeline_retrieval import audit, change_kind, summary


class RetrievalCompositionAuditTests(unittest.TestCase):
    def test_change_kind(self):
        self.assertEqual(change_kind(["a", "b"], ["a", "b"]), "identical")
        self.assertEqual(change_kind(["a", "b"], ["b", "a"]), "reordered_only")
        self.assertEqual(change_kind(["a", "b"], ["a", "c"]), "membership_changed")
        with self.assertRaises(ValueError):
            change_kind(["a", "a"], ["a", "b"])

    def test_empty_summary(self):
        self.assertEqual(summary([])["completed_pairs"], 0)
        self.assertIsNone(summary([])["mean_step_delta"])

    def test_audit_splits_reordering_and_replacement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = {
                "schema": "pipeline_paired_state_v1",
                "rows": [
                    {
                        "candidate_index": 0,
                        "game": {"id": "game0"},
                        "task": "clean an egg",
                        "hybrid_ids": ["a", "b"],
                        "utility_ids": ["b", "a"],
                        "changed": True,
                    },
                    {
                        "candidate_index": 1,
                        "game": {"id": "game1"},
                        "task": "cool a potato",
                        "hybrid_ids": ["a", "b"],
                        "utility_ids": ["a", "c"],
                        "changed": True,
                    },
                    {
                        "candidate_index": 2,
                        "game": {"id": "game2"},
                        "task": "heat a mug",
                        "hybrid_ids": ["a", "b"],
                        "utility_ids": ["a", "b"],
                        "changed": False,
                    },
                ],
            }
            (root / "experiment_plan.json").write_text(
                json.dumps(plan), encoding="utf-8"
            )
            outcomes = [
                (0, "hybrid", ["a", "b"], True, 10, "game0", "clean an egg"),
                (0, "utility", ["b", "a"], True, 13, "game0", "clean an egg"),
                (1, "hybrid", ["a", "b"], False, 50, "game1", "cool a potato"),
                (1, "utility", ["a", "c"], True, 12, "game1", "cool a potato"),
            ]
            for index, arm, ids, success, steps, game, task in outcomes:
                folder = root / "eval" / arm
                folder.mkdir(parents=True, exist_ok=True)
                (folder / f"episode_{index:03d}_meta.json").write_text(
                    json.dumps({
                        "arm": arm, "game_id": game, "task": task,
                        "selected_ids": ids, "success": success,
                        "decision_steps": steps,
                    }),
                    encoding="utf-8",
                )
            result = audit(root)
            self.assertEqual(result["change_counts"], {
                "identical": 1, "reordered_only": 1, "membership_changed": 1,
            })
            self.assertEqual(result["overall"]["completed_pairs"], 2)
            self.assertEqual(result["overall"]["improved"], 1)
            self.assertEqual(
                result["by_change_kind"]["reordered_only"]["mean_step_delta"],
                3,
            )
            self.assertEqual(
                result["by_change_kind"]["membership_changed"]["mean_step_delta"],
                -38,
            )
            self.assertEqual(result["pairs"][1]["added_by_utility"], ["c"])
            self.assertEqual(result["pairs"][1]["removed_by_utility"], ["b"])
            self.assertFalse(result["pairs"][2]["evaluated"])

    def test_incomplete_pair_is_not_counted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "experiment_plan.json").write_text(json.dumps({
                "schema": "pipeline_paired_state_v1",
                "rows": [{
                    "candidate_index": 3, "game": {"id": "g"},
                    "task": "clean egg", "hybrid_ids": ["a"],
                    "utility_ids": ["b"], "changed": True,
                }],
            }), encoding="utf-8")
            result = audit(root)
            self.assertEqual(result["incomplete_changed_indices"], [3])
            self.assertEqual(result["overall"]["completed_pairs"], 0)

    def test_reject_metadata_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "experiment_plan.json").write_text(json.dumps({
                "schema": "pipeline_paired_state_v1",
                "rows": [{
                    "candidate_index": 0, "game": {"id": "g"},
                    "task": "clean egg", "hybrid_ids": ["a"],
                    "utility_ids": ["b"], "changed": True,
                }],
            }), encoding="utf-8")
            for arm in ("hybrid", "utility"):
                folder = root / "eval" / arm
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "episode_000_meta.json").write_text(json.dumps({
                    "arm": arm, "game_id": "wrong", "task": "clean egg",
                    "selected_ids": [("a" if arm == "hybrid" else "b")],
                    "success": True, "decision_steps": 9,
                }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                audit(root)


if __name__ == "__main__":
    unittest.main()
