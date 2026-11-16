import argparse
import fcntl
import os
import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from scripts.run_unified_batch import (
    ROOT,
    collect_jobs,
    run_episodes,
    verify_batch,
)
from scripts.run_unified_online import outcome_path
from scripts.run_unified_workflow import (
    validate_outcome as validate_legacy_outcome,
)
from src.evolution.online_memory_handler import OnlineMemoryHandler
from src.evolution.online_scheduler import OnlineReviewScheduler
from src.evolution.online_skill_handler import OnlineSkillHandler
from src.evolution.persistent_llm_budget import PersistentLLMBudget
from src.evolution.unified_controller import read_json


STATE_ROOT = Path(
    "outputs/unified_evolution_v3/online_batch_01"
)

MAX_CONTINUATION_DECISIONS = 12


@contextmanager
def review_lock(job_id, lock_root):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
        raise ValueError("Invalid review job ID")

    lock_root = Path(lock_root)
    lock_root.mkdir(parents=True, exist_ok=True)
    path = lock_root / f"{job_id}.lock"

    with path.open("a+") as stream:
        try:
            fcntl.flock(
                stream.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Review job is already running: {job_id}"
            ) from exc

        try:
            yield
        finally:
            fcntl.flock(
                stream.fileno(),
                fcntl.LOCK_UN,
            )


def make_scheduler(
    jobs,
    capabilities,
    protocol,
    *,
    batch_root=ROOT,
    state_root=STATE_ROOT,
    execute=False,
    max_llm_calls=16,
):
    if not 1 <= max_llm_calls <= 9999:
        raise ValueError("Invalid LLM call limit")

    memory_validator = OnlineMemoryHandler(
        capabilities,
        model=protocol["model"],
        max_steps=protocol["max_steps"],
        top_k=protocol["top_k"],
        max_continuation_decisions=MAX_CONTINUATION_DECISIONS,
        max_llm_calls=max_llm_calls,
    )

    skill = OnlineSkillHandler(capabilities)

    def resolve(job):
        return outcome_path(batch_root, job)

    def validate(job, path):
        record = read_json(path)
        schema = record.get("schema")

        if job["route"] == "MEMORY_REVIEW":
            if schema == "online_memory_outcome_v2":
                return memory_validator.validate_outcome(
                    job, path
                )

            if schema == "unified_memory_review_outcome_v1":
                return validate_legacy_outcome(job, path)

        if job["route"] == "SKILL_REVIEW":
            if schema == "online_skill_diagnostic_v1":
                return skill.validate_outcome(job, path)

            if schema == "online_skill_review_outcome_v1":
                return validate_legacy_outcome(job, path)

        raise RuntimeError(
            f"Unrecognized review outcome: {job['job_id']}"
        )

    def create_llm_client():
        if os.getenv("OPENAI_MODEL") != protocol["model"]:
            raise RuntimeError(
                "Configured model differs from frozen protocol"
            )

        from src.llm.client import LLMClient
        return LLMClient()

    def process_memory(job, path):
        with review_lock(
            job["job_id"], state_root / "locks"
        ):
            path = Path(path)

            if path.is_file():
                return validate(job, path)

            budget = PersistentLLMBudget(
                state_root / "budgets",
                job=job,
                model=protocol["model"],
                memory_version=capabilities.memory_info.version,
                memory_sha256=capabilities.memory_info.memory_hash,
                active_skill_id=capabilities.skill_id,
                max_calls=max_llm_calls,
                client_factory=create_llm_client,
            )

            handler = OnlineMemoryHandler(
                capabilities,
                model=protocol["model"],
                max_steps=protocol["max_steps"],
                top_k=protocol["top_k"],
                max_continuation_decisions=(
                    MAX_CONTINUATION_DECISIONS
                ),
                max_llm_calls=max_llm_calls,
                llm_factory=lambda: budget,
            )

            return handler.process(job, path)

    def process_skill(job, path):
        with review_lock(
            job["job_id"], state_root / "locks"
        ):
            path = Path(path)

            if path.is_file():
                return validate(job, path)

            return skill.process(job, path)

    handlers = {}

    if execute:
        handlers = {
            "MEMORY_REVIEW": process_memory,
            "SKILL_REVIEW": process_skill,
        }

    return OnlineReviewScheduler(
        jobs,
        outcome_path=resolve,
        validate_outcome=validate,
        handlers=handlers,
    )


def execute_workflow(
    *,
    stage,
    limit=6,
    max_new_jobs=0,
    max_llm_calls=16,
):
    if stage not in ("run", "report"):
        raise ValueError("Unsupported stage")

    if type(max_new_jobs) is not int or max_new_jobs < 0:
        raise ValueError("Invalid new job limit")

    if stage == "report" and max_new_jobs:
        raise ValueError(
            "Report stage cannot execute reviews"
        )

    protocol, capabilities = verify_batch()

    if stage == "run":
        run_episodes(protocol, capabilities, limit)

    jobs, episode_count = collect_jobs(
        protocol, capabilities
    )

    scheduler = make_scheduler(
        jobs,
        capabilities,
        protocol,
        execute=(stage == "run"),
        max_llm_calls=max_llm_calls,
    )

    results = scheduler.run(
        max_new_jobs=(
            max_new_jobs if stage == "run" else 0
        )
    )

    counts = Counter(
        row["state"] for row in results
    )

    print("\n========== UNIFIED V3 WORKFLOW ==========")
    print("Stage:", stage)
    print("Saved episodes:", episode_count)
    print("Active memory:",
          capabilities.memory_info.version)
    print("Active skill:", capabilities.skill_id)
    print("Review jobs:", len(jobs))

    for row in results:
        print(
            row["job_id"],
            row["route"],
            row["state"],
            row["outcome_status"],
        )

    print("Cached:", counts["CACHED"])
    print("Pending:", counts["PENDING"])
    print("Executed:", counts["EXECUTED"])
    print("Automatic promotion: disabled")

    if counts["EXECUTED"] == 0:
        print("New review LLM calls: 0")
    else:
        print(
            "Review LLM reservations: inspect V3 budget ledger"
        )

    print("V3 state:", STATE_ROOT)

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        required=True,
        choices=("run", "report"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--max-new-jobs",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    execute_workflow(
        stage=args.stage,
        limit=args.limit,
        max_new_jobs=args.max_new_jobs,
        max_llm_calls=args.max_llm_calls,
    )


if __name__ == "__main__":
    main()
