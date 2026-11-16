import argparse
from collections import Counter

from scripts.review_unified_online import prepare_plan
from scripts.run_unified_pilot import ROOT, verify_protocol
from scripts.run_unified_workflow import (
    validate_outcome as validate_legacy_outcome,
)
from src.evolution.online_memory_handler import (
    OnlineMemoryHandler,
    read_json,
)
from src.evolution.online_scheduler import OnlineReviewScheduler


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        choices=("report", "run"),
        required=True,
    )
    parser.add_argument(
        "--max-new-jobs",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-continuation-decisions",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    if args.stage == "report" and args.max_new_jobs != 0:
        parser.error("Report mode cannot execute jobs")

    protocol, caps = verify_protocol()
    plan = prepare_plan()

    handler = OnlineMemoryHandler(
        caps,
        model=protocol["model"],
        max_steps=protocol["max_steps"],
        top_k=protocol["top_k"],
        max_continuation_decisions=args.max_continuation_decisions,
        max_llm_calls=args.max_llm_calls,
    )

    jobs = [
        job for job in plan["jobs"]
        if job["route"] == "MEMORY_REVIEW"
    ]

    def outcome_path(job):
        return (
            ROOT / "reviews"
            / job["job_id"]
            / "review_outcome.json"
        )

    def validate(job, path):
        record = read_json(path)

        if record.get("schema") == "online_memory_outcome_v2":
            return handler.validate_outcome(job, path)

        if record.get("schema") == "unified_memory_review_outcome_v1":
            return validate_legacy_outcome(job, path)

        raise RuntimeError("Unknown Memory Review outcome schema")

    scheduler = OnlineReviewScheduler(
        jobs,
        outcome_path=outcome_path,
        validate_outcome=validate,
        handlers=(
            {"MEMORY_REVIEW": handler.process}
            if args.stage == "run"
            else {}
        ),
    )

    results = scheduler.run(
        max_new_jobs=(
            args.max_new_jobs
            if args.stage == "run"
            else 0
        )
    )

    counts = Counter(
        item["state"] for item in results
    )

    print("========== UNIFIED MEMORY HANDLER ==========")
    print("Active memory:", caps.memory_info.version)
    print("Active skill:", caps.skill_id)
    print("Memory jobs:", len(results))

    for item in results:
        print(
            item["job_id"],
            item["state"],
            item["outcome_status"],
        )

    print("Cached:", counts["CACHED"])
    print("Pending:", counts["PENDING"])
    print("Executed:", counts["EXECUTED"])
    print("Memory promotion: disabled")


if __name__ == "__main__":
    main()
