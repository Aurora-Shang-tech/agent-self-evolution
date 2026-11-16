import argparse
from collections import Counter

from scripts.review_unified_online import prepare_plan
from scripts.run_unified_pilot import ROOT, verify_protocol
from scripts.run_unified_workflow import (
    validate_outcome as validate_legacy_outcome,
)
from src.evolution.online_scheduler import OnlineReviewScheduler
from src.evolution.online_skill_handler import OnlineSkillHandler
from src.evolution.unified_controller import read_json


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
    args = parser.parse_args()

    if args.stage == "report" and args.max_new_jobs != 0:
        parser.error("Report cannot execute new jobs")

    protocol, caps = verify_protocol()
    plan = prepare_plan()

    if (
        plan["memory_version"] != caps.memory_info.version
        or plan["memory_sha256"] != caps.memory_info.memory_hash
        or plan["active_skill_id"] != caps.skill_id
        or plan["model"] != protocol["model"]
    ):
        raise RuntimeError("Protocol capability mismatch")

    handler = OnlineSkillHandler(caps)

    jobs = [
        job for job in plan["jobs"]
        if job["route"] == "SKILL_REVIEW"
    ]

    def outcome_path(job):
        return (
            ROOT / "reviews" / job["job_id"]
            / "skill_review.json"
        )

    def validate(job, path):
        record = read_json(path)

        if record.get("schema") == "online_skill_diagnostic_v1":
            return handler.validate_outcome(job, path)

        if record.get("schema") == "online_skill_review_outcome_v1":
            return validate_legacy_outcome(job, path)

        raise RuntimeError("Unknown Skill Review schema")

    scheduler = OnlineReviewScheduler(
        jobs,
        outcome_path=outcome_path,
        validate_outcome=validate,
        handlers=(
            {"SKILL_REVIEW": handler.process}
            if args.stage == "run"
            else {}
        ),
    )

    results = scheduler.run(
        max_new_jobs=(
            args.max_new_jobs if args.stage == "run"
            else 0
        )
    )

    counts = Counter(item["state"] for item in results)

    print("========== UNIFIED SKILL HANDLER ==========")
    print("Active memory:", caps.memory_info.version)
    print("Active skill:", caps.skill_id)
    print("Skill jobs:", len(results))

    for item in results:
        print(
            item["job_id"],
            item["state"],
            item["outcome_status"],
        )

    print("Cached:", counts["CACHED"])
    print("Pending:", counts["PENDING"])
    print("Executed:", counts["EXECUTED"])
    print("New Skill promotions: 0")
    print("LLM calls: 0")


if __name__ == "__main__":
    main()
