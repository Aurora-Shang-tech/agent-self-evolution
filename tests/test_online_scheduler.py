import json
import tempfile
import unittest
from pathlib import Path

from src.evolution.online_scheduler import (
    OnlineReviewScheduler,
    sha256,
)


class OnlineSchedulerTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)

        self.root = Path(folder.name)
        self.trace = self.root / "trace.json"
        self.event = self.root / "event.json"
        self.outcome = self.root / "outcome.json"

        self.trace.write_text(
            '{"task": "mock"}',
            encoding="utf-8",
        )

        self.event_data = {
            "schema": "unified_feedback_event_v1",
            "episode_id": "fresh_00",
            "game_id": "train:test_game",
            "phase": "learning",
            "trajectory_path": str(self.trace),
            "trajectory_sha256": sha256(self.trace),
            "decision": {
                "route": "MEMORY_REVIEW",
                "success": False,
            },
        }

        self.save_event()

        self.job = {
            "job_id": "fresh_00",
            "route": "MEMORY_REVIEW",
            "game_id": "train:test_game",
            "source_scope": "ONLINE_TRAIN_DEVELOPMENT",
            "trajectory_path": str(self.trace),
            "trajectory_sha256": sha256(self.trace),
            "event_path": str(self.event),
            "event_sha256": sha256(self.event),
        }

        self.calls = 0

    def save_event(self):
        self.event.write_text(
            json.dumps(self.event_data),
            encoding="utf-8",
        )

    def validate(self, job, path):
        result = json.loads(
            path.read_text(encoding="utf-8")
        )

        if (
            result["job_id"] != job["job_id"]
            or result["trajectory_sha256"]
            != job["trajectory_sha256"]
        ):
            raise RuntimeError(
                "Wrong cached outcome source"
            )

        return result["status"]

    def handle(self, job, path):
        self.calls += 1

        path.write_text(
            json.dumps({
                "job_id": job["job_id"],
                "trajectory_sha256":
                    job["trajectory_sha256"],
                "status": "DONE",
            }),
            encoding="utf-8",
        )

    def scheduler(self, handlers=None):
        return OnlineReviewScheduler(
            [self.job],
            outcome_path=lambda job: self.outcome,
            validate_outcome=self.validate,
            handlers=handlers,
        )

    def test_zero_budget_does_not_execute(self):
        scheduler = self.scheduler({
            "MEMORY_REVIEW": self.handle,
        })

        result = scheduler.run()

        self.assertEqual(
            result[0]["state"], "PENDING"
        )
        self.assertEqual(self.calls, 0)

    def test_execute_once_and_resume_from_cache(self):
        scheduler = self.scheduler({
            "MEMORY_REVIEW": self.handle,
        })

        self.assertEqual(
            scheduler.run(max_new_jobs=1)[0]["state"],
            "EXECUTED",
        )

        self.assertEqual(
            scheduler.run(max_new_jobs=1)[0]["state"],
            "CACHED",
        )

        self.assertEqual(self.calls, 1)

    def test_changed_trajectory_rejected_before_handler(self):
        self.trace.write_text(
            '{"task": "changed"}',
            encoding="utf-8",
        )

        with self.assertRaises(RuntimeError):
            self.scheduler({
                "MEMORY_REVIEW": self.handle,
            }).run(max_new_jobs=1)

        self.assertEqual(self.calls, 0)

    def test_non_train_phase_rejected(self):
        self.event_data["phase"] = "holdout"
        self.save_event()

        self.job["event_sha256"] = sha256(self.event)

        with self.assertRaises(ValueError):
            self.scheduler({
                "MEMORY_REVIEW": self.handle,
            }).run(max_new_jobs=1)

        self.assertEqual(self.calls, 0)

    def test_missing_handler_fails_closed(self):
        with self.assertRaises(RuntimeError):
            self.scheduler().run(max_new_jobs=1)

    def test_modified_cached_outcome_rejected(self):
        self.handle(self.job, self.outcome)

        data = json.loads(
            self.outcome.read_text(encoding="utf-8")
        )

        data["trajectory_sha256"] = "tampered"

        self.outcome.write_text(
            json.dumps(data),
            encoding="utf-8",
        )

        with self.assertRaises(RuntimeError):
            self.scheduler().run(max_new_jobs=0)


if __name__ == "__main__":
    unittest.main()
