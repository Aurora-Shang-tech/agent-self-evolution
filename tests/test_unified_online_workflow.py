import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_unified_online import build_scheduler, outcome_path
from src.core.trajectory import Trajectory, TrajectoryStep
from src.evolution.online_scheduler import sha256


class UnifiedOnlineWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        self.root = Path(temp.name)

        self.capabilities = SimpleNamespace(
            memory_info=SimpleNamespace(
                version="v000",
                memory_hash="test_memory_hash",
            ),
            skill_id=None,
        )

        self.protocol = {
            "model": "test-model",
            "max_steps": 50,
            "top_k": 3,
        }

    def make_job(self, job_id, route, success):
        folder = self.root / job_id
        folder.mkdir(parents=True)

        actions = [
            (
                "go to garbagecan 1",
                "On the garbagecan 1, you see a spraybottle 1.",
            ),
            (
                "take spraybottle 1 from garbagecan 1",
                "You pick up the spraybottle 1.",
            ),
            (
                "move spraybottle 1 to countertop 1",
                "You move the spraybottle 1 to countertop 1.",
            ),
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
                won=success and i == len(actions) - 1,
            )
            for i, (action, observation) in enumerate(actions)
        ]

        trace = Trajectory(
            task="put some spraybottle on countertop.",
            gamefile="mock_game",
            success=success,
            steps=steps,
        )

        trace_path = folder / "trajectory.json"
        event_path = folder / "event.json"

        trace.save_json(str(trace_path))

        event = {
            "schema": "unified_feedback_event_v1",
            "episode_id": job_id,
            "game_id": "train:mock",
            "phase": "learning",
            "trajectory_path": str(trace_path),
            "trajectory_sha256": sha256(trace_path),
            "memory_version": "v000",
            "memory_sha256": "test_memory_hash",
            "active_skill_id": None,
            "decision": {
                "route": route,
                "success": success,
                "task": trace.task,
                "environment_actions": 3,
                "reference_median": 1.0,
                "reference_samples": 5,
            },
        }

        event_path.write_text(
            json.dumps(event),
            encoding="utf-8",
        )

        return {
            "job_id": job_id,
            "game_id": "train:mock",
            "route": route,
            "source_scope": "ONLINE_TRAIN_DEVELOPMENT",
            "trajectory_path": str(trace_path),
            "trajectory_sha256": sha256(trace_path),
            "event_path": str(event_path),
            "event_sha256": sha256(event_path),
        }

    def scheduler(self, jobs, enable_handlers=False):
        return build_scheduler(
            jobs,
            self.capabilities,
            self.protocol,
            self.root,
            enable_handlers=enable_handlers,
        )

    def test_legacy_memory_and_skill_cached(self):
        memory_job = self.make_job(
            "fresh_04", "MEMORY_REVIEW", False
        )
        skill_job = self.make_job(
            "fresh_05", "SKILL_REVIEW", True
        )

        for job, schema, status in [
            (
                memory_job,
                "unified_memory_review_outcome_v1",
                "STOPPED_WITH_UNTESTED_CANDIDATE",
            ),
            (
                skill_job,
                "online_skill_review_outcome_v1",
                "NO_CANDIDATE_IN_PILOT",
            ),
        ]:
            path = outcome_path(self.root, job)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({
                    "schema": schema,
                    "status": status,
                }),
                encoding="utf-8",
            )

        with patch(
            "scripts.run_unified_online.validate_legacy_outcome",
            side_effect=lambda job, path: json.loads(
                path.read_text(encoding="utf-8")
            )["status"],
        ):
            results = self.scheduler([
                memory_job, skill_job
            ]).run(max_new_jobs=0)

        self.assertEqual(
            [row["state"] for row in results],
            ["CACHED", "CACHED"],
        )

    def test_zero_budget_does_not_execute_memory(self):
        job = self.make_job(
            "fresh_04", "MEMORY_REVIEW", False
        )

        with patch(
            "src.evolution.online_memory_handler."
            "OnlineMemoryHandler.process",
            side_effect=AssertionError(
                "Memory execution must not occur"
            ),
        ):
            results = self.scheduler(
                [job],
                enable_handlers=True,
            ).run(max_new_jobs=0)

        self.assertEqual(results[0]["state"], "PENDING")

    def test_skill_execution_then_cached(self):
        job = self.make_job(
            "fresh_05", "SKILL_REVIEW", True
        )

        scheduler = self.scheduler(
            [job],
            enable_handlers=True,
        )

        first = scheduler.run(max_new_jobs=1)
        second = scheduler.run(max_new_jobs=1)

        self.assertEqual(first[0]["state"], "EXECUTED")
        self.assertEqual(second[0]["state"], "CACHED")
        self.assertEqual(
            second[0]["outcome_status"],
            "DIAGNOSTIC_ONLY",
        )

        record = json.loads(
            outcome_path(self.root, job).read_text(
                encoding="utf-8"
            )
        )

        self.assertFalse(record["candidate_generated"])
        self.assertFalse(record["skill_promoted"])


if __name__ == "__main__":
    unittest.main()
