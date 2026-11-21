import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

from scripts.memory.run_evolution import memory_hash
from src.core.trajectory import Trajectory, TrajectoryStep
from src.evolution.online_memory_source_bridge import (
    load_verified_online_candidate,
    stage_verified_online_update,
)
from src.memory.updater import LessonCandidate, MemoryItem


class FakeLLM:
    def __init__(self, lesson):
        self.lesson = lesson
        self.calls = 0

    def chat(self, messages):
        self.calls += 1

        return json.dumps({
            "operation": "ADD",
            "target_id": None,
            "new_lesson": self.lesson,
            "reason": "Successful verified comparison.",
            "evidence_refs": [
                "baseline:critical",
                "counterfactual:replacement",
            ],
        })


class OnlineMemorySourceBridgeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        self.root = Path(temp.name)
        self.outcome_path = (
            self.root / "review_outcome.json"
        )
        self.candidate_path = (
            self.root / "automated_candidate.json"
        )
        self.trace_path = (
            self.root / "trajectory.json"
        )

        self.lesson = (
            "Search likely object locations before "
            "checking every container sequentially."
        )

        trace = Trajectory(
            task="find a plate",
            gamefile="mock_game",
            success=False,
            steps=[
                TrajectoryStep(
                    observation="You are in a room.",
                    admissible_actions=[
                        "open cabinet 1",
                        "go to shelf 1",
                    ],
                    thought="Search for the plate.",
                    action="open cabinet 1",
                    valid_format=True,
                    admissible=True,
                    next_observation="The cabinet is empty.",
                    score=0.0,
                    done=False,
                    won=False,
                )
            ],
        )

        trace.save_json(str(self.trace_path))

        self.job = {
            "job_id": "batch_07",
            "game_id": "train:mock",
            "route": "MEMORY_REVIEW",
            "source_scope": "ONLINE_TRAIN_DEVELOPMENT",
            "trajectory_path": str(self.trace_path),
            "trajectory_sha256": "mock_trace_hash",
            "event_sha256": "mock_event_hash",
        }

        self.candidate = LessonCandidate(
            decision="CANDIDATE",
            lesson=self.lesson,
            reason="Successful local counterfactual.",
            limitations="One local comparison.",
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
                    "action": "open cabinet 1",
                },
                "counterfactual:replacement": {
                    "action": "go to shelf 1",
                },
            },
        )

        self.memory = [
            MemoryItem(
                "base_0000",
                "Verify the exact object type before acting.",
                "find a cup",
            )
        ]

        self.handler = Mock()
        self.handler.validate_outcome.return_value = (
            "CANDIDATE_PENDING_GATE"
        )

        self.write_outcome("CANDIDATE_PENDING_GATE")
        self.write_candidate(self.candidate)

    def write_outcome(self, status):
        self.outcome_path.write_text(
            json.dumps({
                "schema": "online_memory_outcome_v2",
                "status": status,
                "job_id": "batch_07",
            }),
            encoding="utf-8",
        )

    def write_candidate(self, candidate):
        self.candidate_path.write_text(
            json.dumps({
                "schema": "automated_lesson_candidate_v1",
                "candidate": asdict(candidate),
            }),
            encoding="utf-8",
        )

    def stage(self, llm):
        return stage_verified_online_update(
            self.job,
            self.outcome_path,
            self.handler,
            llm,
            self.memory,
            memory_hash(self.memory),
        )

    def test_verified_candidate_stages_and_resumes(self):
        first = FakeLLM(self.lesson)
        record = self.stage(first)

        self.assertEqual(first.calls, 1)
        self.assertEqual(
            record["staged"]["status"],
            "PENDING_VALIDATION",
        )
        self.assertFalse(
            record["active_memory_changed"]
        )

        second = FakeLLM(self.lesson)
        cached = self.stage(second)

        self.assertEqual(second.calls, 0)
        self.assertEqual(cached, record)

    def test_non_candidate_outcome_rejected(self):
        self.write_outcome("NO_SUCCESS_WITHIN_BUDGET")

        with self.assertRaises(ValueError):
            self.stage(FakeLLM(self.lesson))

    def test_handler_validation_failure_blocks_update(self):
        self.handler.validate_outcome.side_effect = (
            RuntimeError("Counterfactual evidence mismatch")
        )

        with self.assertRaises(RuntimeError):
            self.stage(FakeLLM(self.lesson))

    def test_baseline_action_mismatch_rejected(self):
        changed = LessonCandidate(
            **{
                **asdict(self.candidate),
                "original_action": "open drawer 1",
            }
        )

        self.write_candidate(changed)

        with self.assertRaises(ValueError):
            load_verified_online_candidate(
                self.job,
                self.outcome_path,
                self.handler,
            )

    def test_parent_hash_mismatch_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "parent memory hash"
        ):
            stage_verified_online_update(
                self.job,
                self.outcome_path,
                self.handler,
                FakeLLM(self.lesson),
                self.memory,
                "incorrect_hash",
            )


if __name__ == "__main__":
    unittest.main()
