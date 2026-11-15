import unittest
from copy import deepcopy
from types import SimpleNamespace

from src.evolution.review_consumer import ReviewQueueConsumer


def make_event(episode_id, route, success):
    return {
        "episode_id": episode_id,
        "game_id": f"train:{episode_id}",
        "task_type": "pick_heat_then_place_in_recep",
        "phase": "learning",
        "memory_version": "v000",
        "memory_sha256": "memory_hash",
        "active_skill_id": None,
        "skill_mode": None,
        "trajectory_path": f"/tmp/{episode_id}.json",
        "trajectory_sha256": "a" * 64,
        "decision": {
            "route": route,
            "success": success,
        },
    }


class ReviewConsumerTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            make_event("r1_e001", "MEMORY_REVIEW", False),
            make_event("r1_e002", "SKILL_REVIEW", True),
        ]

        capabilities = SimpleNamespace(
            memory_info=SimpleNamespace(
                version="v000",
                memory_hash="memory_hash",
            ),
            skill_id=None,
            skill_mode=None,
        )

        controller = SimpleNamespace(
            capabilities=capabilities,
            review_queue=lambda: deepcopy(self.events),
        )

        self.consumer = ReviewQueueConsumer(controller)

    def test_routes_to_correct_handlers(self):
        plan = self.consumer.prepare_historical()

        self.assertEqual(
            plan["summary"]["memory_reviews"], 1
        )
        self.assertEqual(
            plan["summary"]["skill_reviews"], 1
        )

        jobs = {job["job_id"]: job for job in plan["jobs"]}

        self.assertEqual(
            jobs["r1_e001"]["handler"],
            "memory_failure_analysis",
        )
        self.assertEqual(
            jobs["r1_e002"]["handler"],
            "skill_inefficiency_analysis",
        )

    def test_historical_jobs_cannot_execute(self):
        plan = self.consumer.prepare_historical()

        self.assertEqual(
            plan["summary"]["executable_jobs"], 0
        )
        self.assertTrue(
            all(
                not job["eligible_for_learning"]
                and job["status"] == "REFERENCE_ONLY"
                for job in plan["jobs"]
            )
        )

    def test_duplicate_episode_rejected(self):
        self.events.append(deepcopy(self.events[0]))

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.consumer.prepare_historical()

    def test_evaluation_data_rejected(self):
        self.events[0]["phase"] = "holdout"

        with self.assertRaisesRegex(
            ValueError, "Evaluation episode"
        ):
            self.consumer.prepare_historical()

    def test_memory_version_mismatch_rejected(self):
        self.events[0]["memory_version"] = "v001"

        with self.assertRaisesRegex(
            ValueError, "Memory version mismatch"
        ):
            self.consumer.prepare_historical()

    def test_invalid_route_outcome_rejected(self):
        self.events[0]["decision"]["success"] = True

        with self.assertRaisesRegex(
            ValueError, "contradicts"
        ):
            self.consumer.prepare_historical()


if __name__ == "__main__":
    unittest.main()
