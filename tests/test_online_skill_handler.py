import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.core.trajectory import Trajectory, TrajectoryStep
from src.evolution.online_scheduler import (
    OnlineReviewScheduler,
    sha256,
)
from src.evolution.online_skill_handler import OnlineSkillHandler


class OnlineSkillHandlerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        self.root = Path(temp.name)
        self.trace_path = self.root / "trajectory.json"
        self.event_path = self.root / "event.json"
        self.output = self.root / "skill_review.json"

        actions = [
            ("go to drawer 1", "The drawer 1 is closed."),
            ("open drawer 1", "The drawer 1 is empty."),
            ("go to garbagecan 1",
             "On the garbagecan 1, you see a spraybottle 1."),
            ("take spraybottle 1 from garbagecan 1",
             "You pick up the spraybottle 1."),
            ("move spraybottle 1 to countertop 1",
             "You move the spraybottle 1 to countertop 1."),
        ]

        steps = [
            TrajectoryStep(
                observation="room",
                admissible_actions=[action],
                thought="test",
                action=action,
                valid_format=True,
                admissible=True,
                next_observation=observation,
                score=0.0,
                done=i == len(actions) - 1,
                won=i == len(actions) - 1,
            )
            for i, (action, observation) in enumerate(actions)
        ]

        trace = Trajectory(
            task="put some spraybottle on countertop.",
            gamefile="mock_game",
            success=True,
            steps=steps,
        )

        trace.save_json(str(self.trace_path))

        event = {
            "schema": "unified_feedback_event_v1",
            "episode_id": "fresh_05",
            "game_id": "train:mock",
            "phase": "learning",
            "trajectory_path": str(self.trace_path),
            "trajectory_sha256": sha256(self.trace_path),
            "memory_version": "v000",
            "memory_sha256": "memory_hash",
            "active_skill_id": None,
            "decision": {
                "route": "SKILL_REVIEW",
                "success": True,
                "task": trace.task,
                "environment_actions": 5,
                "reference_median": 2.0,
                "reference_samples": 5,
            },
        }

        self.event_path.write_text(
            json.dumps(event),
            encoding="utf-8",
        )

        self.job = {
            "job_id": "fresh_05",
            "game_id": "train:mock",
            "route": "SKILL_REVIEW",
            "source_scope": "ONLINE_TRAIN_DEVELOPMENT",
            "trajectory_path": str(self.trace_path),
            "trajectory_sha256": sha256(self.trace_path),
            "event_path": str(self.event_path),
            "event_sha256": sha256(self.event_path),
        }

        caps = SimpleNamespace(
            memory_info=SimpleNamespace(
                version="v000",
                memory_hash="memory_hash",
            ),
            skill_id=None,
        )

        self.handler = OnlineSkillHandler(caps)

    def test_extract_search_diagnostic(self):
        record = self.handler.build_record(self.job)

        self.assertEqual(record["status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(record["pre_first_take_steps"], 3)
        self.assertEqual(record["first_take_step_index"], 3)
        self.assertEqual(
            record["first_object_sighting"]["step_index"], 2
        )
        self.assertEqual(
            record["distinct_searched_locations"], 2
        )
        self.assertFalse(record["candidate_generated"])
        self.assertFalse(record["skill_promoted"])

    def test_save_and_resume(self):
        scheduler = OnlineReviewScheduler(
            [self.job],
            outcome_path=lambda job: self.output,
            validate_outcome=self.handler.validate_outcome,
            handlers={
                "SKILL_REVIEW": self.handler.process
            },
        )

        self.assertEqual(
            scheduler.run(max_new_jobs=1)[0]["state"],
            "EXECUTED",
        )
        self.assertEqual(
            scheduler.run(max_new_jobs=1)[0]["state"],
            "CACHED",
        )

    def test_tampered_outcome_rejected(self):
        self.handler.process(self.job, self.output)
        record = json.loads(
            self.output.read_text(encoding="utf-8")
        )
        record["candidate_generated"] = True

        self.output.write_text(
            json.dumps(record),
            encoding="utf-8",
        )

        with self.assertRaises(RuntimeError):
            self.handler.validate_outcome(
                self.job, self.output
            )

    def test_holdout_is_rejected(self):
        event = json.loads(
            self.event_path.read_text(encoding="utf-8")
        )
        event["phase"] = "holdout"

        self.event_path.write_text(
            json.dumps(event),
            encoding="utf-8",
        )
        self.job["event_sha256"] = sha256(self.event_path)

        with self.assertRaises(ValueError):
            self.handler.process(self.job, self.output)

    def test_capability_mismatch_rejected(self):
        self.handler.capabilities.memory_info.memory_hash = (
            "different_hash"
        )

        with self.assertRaises(RuntimeError):
            self.handler.process(self.job, self.output)


if __name__ == "__main__":
    unittest.main()
