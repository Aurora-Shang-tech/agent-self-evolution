import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_unified_v3 import (
    execute_workflow,
    make_scheduler,
    review_lock,
)


class UnifiedV3Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)

        self.root = Path(temp.name)

        self.capabilities = SimpleNamespace(
            memory_info=SimpleNamespace(
                version="v000",
                memory_hash="mock_hash",
            ),
            skill_id=None,
        )

        self.protocol = {
            "model": "mock-model",
            "max_steps": 50,
            "top_k": 3,
            "games": [],
        }

        self.job = {
            "job_id": "batch_07",
            "game_id": "train:mock",
            "route": "MEMORY_REVIEW",
            "source_scope": "ONLINE_TRAIN_DEVELOPMENT",
            "trajectory_sha256": "trajectory_hash",
            "event_sha256": "event_hash",
        }

    def test_process_lock_rejects_second_owner(self):
        lock_root = self.root / "locks"

        with review_lock("batch_07", lock_root):
            with self.assertRaisesRegex(
                RuntimeError, "already running"
            ):
                with review_lock("batch_07", lock_root):
                    pass

        with review_lock("batch_07", lock_root):
            pass

    @patch(
        "scripts.run_unified_v3.PersistentLLMBudget"
    )
    @patch(
        "scripts.run_unified_v3.OnlineMemoryHandler"
    )
    @patch(
        "scripts.run_unified_v3.OnlineReviewScheduler"
    )
    def test_memory_handler_receives_persistent_budget(
        self,
        scheduler_class,
        memory_class,
        budget_class,
    ):
        make_scheduler(
            [self.job],
            self.capabilities,
            self.protocol,
            batch_root=self.root,
            state_root=self.root / "v3",
            execute=True,
            max_llm_calls=3,
        )

        handlers = scheduler_class.call_args.kwargs[
            "handlers"
        ]
        process = handlers["MEMORY_REVIEW"]

        outcome_path = (
            self.root / "reviews"
            / "batch_07"
            / "review_outcome.json"
        )

        process(self.job, outcome_path)

        budget_class.assert_called_once()
        self.assertEqual(
            budget_class.call_args.kwargs["max_calls"],
            3,
        )

        handler_options = memory_class.call_args.kwargs

        self.assertIs(
            handler_options["llm_factory"](),
            budget_class.return_value,
        )

        memory_class.return_value.process.assert_called_once()

    @patch("scripts.run_unified_v3.make_scheduler")
    @patch("scripts.run_unified_v3.collect_jobs")
    @patch("scripts.run_unified_v3.run_episodes")
    @patch("scripts.run_unified_v3.verify_batch")
    def test_report_never_runs_episodes(
        self,
        verify,
        run_episodes,
        collect_jobs,
        make_scheduler,
    ):
        verify.return_value = (
            self.protocol,
            self.capabilities,
        )
        collect_jobs.return_value = ([], 6)
        make_scheduler.return_value.run.return_value = []

        execute_workflow(
            stage="report",
            max_new_jobs=0,
        )

        run_episodes.assert_not_called()
        make_scheduler.return_value.run.assert_called_once_with(
            max_new_jobs=0
        )

    def test_report_rejects_execution_budget(self):
        with self.assertRaises(ValueError):
            execute_workflow(
                stage="report",
                max_new_jobs=1,
            )


if __name__ == "__main__":
    unittest.main()
