
"""Offline acceptance tests for continual memory version lifecycle."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.memory.run_evolution import apply_proposal, atomic_json, memory_hash
from src.evolution.controller import EvolutionController
from src.evolution.version_store import load_memory_version, read_round_history
from src.memory.updater import MemoryItem, UpdateProposal


class VersionLifecycleTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name) / "offline_run"
        self.home.mkdir()

        self.initial = [
            MemoryItem(
                "base_0000",
                "Check the correct location first.",
                "find a plate",
            ),
            MemoryItem(
                "base_0001",
                "Check the object state before finishing.",
                "heat a plate",
            ),
        ]

        self.controller = EvolutionController.__new__(EvolutionController)
        c = self.controller

        c.home = self.home
        c.mode = "dynamic"
        c.rounds = 2
        c.learn_sets = [
            [SimpleNamespace(id="mock_game_1")],
            [SimpleNamespace(id="mock_game_2")],
        ]
        c.protocol = {
            "selection_policy": {
                "max_regressions": 1,
                "min_step_saving_if_tied": 1.0,
            }
        }

        c.save_version("v000", None, 0, self.initial)

        atomic_json(
            self.home / "state.json",
            {
                "active_version": "v000",
                "completed_rounds": 0,
            },
        )

        c.initialize = lambda: self.read_state()
        c.do_learning = Mock(return_value=[(0, None, None)])
        c.process_failure = Mock(side_effect=self.make_candidate)
        c.retrieval_changes = Mock(return_value=1)
        c.eval_version = Mock(side_effect=self.accepting_evaluation)

    def read_state(self):
        return json.loads(
            (self.home / "state.json").read_text(encoding="utf-8")
        )

    def make_candidate(self, round_dir, index, trace, memory):
        return list(memory) + [
            MemoryItem(
                f"evo_{round_dir.name}",
                "A distinct test lesson for version lifecycle checks.",
                "find and heat a plate",
            )
        ]

    @staticmethod
    def accepting_evaluation(version, memory):
        candidate = version != "v000"
        return [{
            "game_id": "mock_validation_game",
            "success": candidate,
            "decision_steps": 10 if candidate else 20,
        }]

    def run_offline(self, rounds):
        with patch(
            "src.evolution.controller.Trajectory.load_json",
            return_value=SimpleNamespace(success=False),
        ), redirect_stdout(io.StringIO()):
            self.controller.run(max_rounds=rounds)

    def test_add_edit_remove_and_snapshot_loading(self):
        items = list(self.initial)
        candidate = SimpleNamespace(
            source_task="find and heat a plate"
        )
        previous = "v000"

        changes = [
            (
                "ADD",
                None,
                "New actionable guidance for a distinct situation.",
            ),
            (
                "EDIT",
                "base_0000",
                "Check the task-specific object locations before broad search.",
            ),
            (
                "REMOVE",
                "evo_test_add",
                "",
            ),
        ]

        for i, (op, target, new_lesson) in enumerate(changes, 1):
            proposal = UpdateProposal(
                operation=op,
                target_id=target,
                new_lesson=new_lesson,
                reason="Offline test proposal",
                evidence_refs=(),
                memory_snapshot_sha256=memory_hash(items),
            )

            event_id = (
                "test_add" if op == "ADD" else f"test_{i}"
            )

            apply_proposal(
                items,
                candidate,
                proposal,
                event_id,
            )

            version = f"v{i:03d}"

            self.controller.save_version(
                version,
                previous,
                i,
                items,
            )

            _, info = load_memory_version(
                self.home,
                version,
            )

            self.assertEqual(info.memory_size, len(items))
            self.assertEqual(info.parent, previous)
            previous = version

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].id, "base_0000")
        self.assertIn("task-specific", items[0].lesson)

    def test_corrupted_snapshot_is_rejected(self):
        path = self.home / "versions" / "v000.json"
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        payload["memory"][0]["lesson"] = "Tampered lesson."
        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "hash mismatch",
        ):
            load_memory_version(self.home, "v000")

    def test_accept_candidate(self):
        self.run_offline(1)

        state = self.read_state()

        self.assertEqual(
            state,
            {
                "active_version": "v001",
                "completed_rounds": 1,
            },
        )

        _, info = load_memory_version(
            self.home,
            "v001",
        )
        self.assertEqual(info.memory_size, 3)

        decision = json.loads(
            (
                self.home
                / "rounds/round_01/decision.json"
            ).read_text()
        )

        self.assertTrue(decision["result"]["accept"])
        self.assertEqual(
            decision["result"]["reason"],
            "success_gain",
        )

    def test_reject_candidate(self):
        self.controller.eval_version.side_effect = (
            lambda version, memory: [{
                "game_id": "mock_validation_game",
                "success": True,
                "decision_steps": (
                    15 if version != "v000" else 10
                ),
            }]
        )

        self.run_offline(1)

        self.assertEqual(
            self.read_state()["active_version"],
            "v000",
        )

        self.assertTrue(
            (
                self.home
                / "versions/v001.json"
            ).exists()
        )

        decision = json.loads(
            (
                self.home
                / "rounds/round_01/decision.json"
            ).read_text()
        )

        self.assertFalse(decision["result"]["accept"])
        self.assertEqual(
            decision["result"]["reason"],
            "no_sufficient_gain",
        )

    def test_resume_after_decision_before_state_write(self):
        from src.evolution import controller as module

        original_write = module.atomic_json
        crashed = False

        def interrupt_once(path, data):
            nonlocal crashed

            if (
                Path(path).name == "state.json"
                and not crashed
            ):
                crashed = True
                raise RuntimeError(
                    "simulated crash after decision"
                )

            return original_write(path, data)

        with patch(
            "src.evolution.controller.atomic_json",
            side_effect=interrupt_once,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated crash",
            ):
                self.run_offline(1)

        self.assertEqual(
            self.read_state()["active_version"],
            "v000",
        )

        self.assertTrue(
            (
                self.home
                / "rounds/round_01/decision.json"
            ).exists()
        )

        self.run_offline(1)

        self.assertEqual(
            self.read_state()["active_version"],
            "v001",
        )

        self.assertEqual(
            self.controller.do_learning.call_count,
            1,
        )
        self.assertEqual(
            self.controller.process_failure.call_count,
            1,
        )

    def test_rollback_then_continue_history_consistency(self):
        self.run_offline(1)

        with redirect_stdout(io.StringIO()):
            self.controller.rollback("v000")

        self.assertEqual(
            self.read_state()["active_version"],
            "v000",
        )

        self.run_offline(2)

        history = read_round_history(self.home)

        self.assertEqual(
            history[-1]["parent"],
            "v000",
        )

        self.assertEqual(
            history[0]["rollbacks_after"],
            [{
                "after_round": 1,
                "from": "v001",
                "to": "v000",
            }],
        )

        self.assertEqual(
            history[-1]["active_after"],
            self.read_state()["active_version"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

