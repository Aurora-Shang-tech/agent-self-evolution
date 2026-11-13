
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.memory.run_evolution import (
    apply_proposal,
    atomic_json,
    memory_hash,
)
from src.evolution.controller import EvolutionController
from src.evolution.version_store import (
    load_active_memory,
    read_round_history,
)
from src.memory.updater import (
    LessonCandidate,
    MemoryItem,
    UpdateProposal,
)


class ContinualRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)

        self.root = Path(temporary.name) / "offline_run"

        self.initial = [
            MemoryItem(
                id="base_0000",
                lesson="Check likely locations before broad search.",
                source_task="find a plate",
            ),
            MemoryItem(
                id="base_0001",
                lesson="Confirm the required state before placement.",
                source_task="heat a plate",
            ),
        ]

        controller = EvolutionController.__new__(
            EvolutionController
        )

        controller.home = self.root
        controller.mode = "dynamic"
        controller.rounds = 1
        controller.learn_sets = [
            [SimpleNamespace(id="mock_game")]
        ]
        controller.protocol = {
            "selection_policy": {
                "max_regressions": 1,
                "min_step_saving_if_tied": 1.0,
            }
        }

        controller.save_version(
            "v000", None, 0, self.initial
        )

        atomic_json(
            self.root / "state.json",
            {
                "active_version": "v000",
                "completed_rounds": 0,
            },
        )

        controller.initialize = self.read_state
        controller.do_learning = Mock(
            return_value=[(0, None, object())]
        )
        controller.retrieval_changes = Mock(
            return_value=1
        )
        controller.eval_version = Mock(
            side_effect=lambda version, memory: [{
                "game_id": "mock_validation",
                "success": version == "v001",
                "decision_steps": (
                    10 if version == "v001" else 20
                ),
            }]
        )

        self.controller = controller
        self.event_path = (
            self.root
            / "rounds"
            / "round_01"
            / "failures"
            / "episode_000"
            / "event.json"
        )

        self.create_saved_event()

    def read_state(self):
        return json.loads(
            (self.root / "state.json").read_text(
                encoding="utf-8"
            )
        )

    def create_saved_event(self):
        candidate = LessonCandidate(
            decision="CANDIDATE",
            lesson=(
                "Use the task-specific destination and "
                "object constraints to prioritize search."
            ),
            reason="Simulated successful counterfactual",
            limitations="Engineering test only",
            evidence_refs=(
                "baseline:critical",
                "counterfactual:replacement",
            ),
            source_task="find a plate",
            critical_step=0,
            original_action="go to fridge 1",
            replacement_action="go to shelf 1",
            evidence={},
        )

        proposal = UpdateProposal(
            operation="ADD",
            target_id=None,
            new_lesson=candidate.lesson,
            reason="Synthetic update for recovery testing",
            evidence_refs=candidate.evidence_refs,
            memory_snapshot_sha256=memory_hash(
                self.initial
            ),
        )

        updated = list(self.initial)

        apply_proposal(
            updated,
            candidate,
            proposal,
            "r01_e000",
        )

        atomic_json(
            self.event_path,
            {
                "before_hash": memory_hash(self.initial),
                "after_hash": memory_hash(updated),
                "candidate": asdict(candidate),
                "proposal": asdict(proposal),
                "outcome": "ADD",
            },
        )

    def run_offline(self):
        with (
            patch(
                "src.evolution.controller.Trajectory.load_json",
                return_value=SimpleNamespace(success=False),
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.controller.run(max_rounds=1)

    def test_resume_after_saved_event_and_interrupted_decision(self):
        original_event = self.event_path.read_bytes()

        from src.evolution import controller as module

        original_write = module.atomic_json
        interrupted = False

        def interrupt_decision(path, data):
            nonlocal interrupted

            if (
                Path(path).name == "decision.json"
                and not interrupted
            ):
                interrupted = True
                raise RuntimeError(
                    "simulated decision write interruption"
                )

            return original_write(path, data)

        with patch(
            "src.evolution.controller.atomic_json",
            side_effect=interrupt_decision,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated decision write interruption",
            ):
                self.run_offline()

        self.assertTrue(interrupted)
        self.assertEqual(
            self.read_state()["active_version"],
            "v000",
        )
        self.assertEqual(
            self.read_state()["completed_rounds"],
            0,
        )
        self.assertTrue(
            (self.root / "versions/v001.json").exists()
        )
        self.assertFalse(
            (
                self.root
                / "rounds/round_01/decision.json"
            ).exists()
        )

        self.run_offline()

        store, info = load_active_memory(self.root)
        history = read_round_history(self.root)

        self.assertEqual(info.version, "v001")
        self.assertEqual(info.memory_size, 3)
        self.assertEqual(len(history), 1)
        self.assertEqual(
            history[0]["reason"],
            "success_gain",
        )
        self.assertEqual(
            self.event_path.read_bytes(),
            original_event,
        )

        snapshot = json.loads(
            (self.root / "versions/v001.json").read_text()
        )

        self.assertEqual(
            len({
                item["id"]
                for item in snapshot["memory"]
            }),
            3,
        )
        self.assertEqual(
            sum(
                item["id"] == "evo_r01_e000"
                for item in snapshot["memory"]
            ),
            1,
        )

        # A completed run must remain unchanged on retry.
        saved_state = self.read_state()
        self.run_offline()

        self.assertEqual(
            self.read_state(),
            saved_state,
        )
        self.assertEqual(
            self.event_path.read_bytes(),
            original_event,
        )
        self.assertEqual(
            self.controller.do_learning.call_count,
            2,
        )

    def test_corrupted_saved_event_is_rejected(self):
        original = json.loads(
            self.event_path.read_text(encoding="utf-8")
        )

        round_dir = (
            self.root / "rounds" / "round_01"
        )

        for key, message in (
            ("before_hash", "Memory replay divergence"),
            ("after_hash", "Memory replay after-hash mismatch"),
        ):
            with self.subTest(key=key):
                corrupted = dict(original)
                corrupted[key] = "0" * 64

                atomic_json(
                    self.event_path,
                    corrupted,
                )

                with self.assertRaisesRegex(
                    RuntimeError,
                    message,
                ):
                    self.controller.process_failure(
                        round_dir,
                        0,
                        None,
                        list(self.initial),
                    )

        atomic_json(self.event_path, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)

