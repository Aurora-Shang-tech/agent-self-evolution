import argparse
import re
from collections import Counter
from pathlib import Path

from scripts.review_unified_online import (
    PROTOCOL_PATH,
    prepare_plan,
)
from scripts.run_unified_pilot import ROOT, verify_protocol
from scripts.run_unified_workflow import (
    validate_outcome as validate_legacy_outcome,
)
from src.evolution.online_memory_handler import OnlineMemoryHandler
from src.evolution.online_scheduler import (
    OnlineReviewScheduler,
    sha256,
)
from src.evolution.online_skill_handler import OnlineSkillHandler
from src.evolution.unified_controller import read_json


ROUTES = {"MEMORY_REVIEW", "SKILL_REVIEW"}


def outcome_path(root, job):
    job_id = job["job_id"]

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
        raise ValueError("Invalid review job ID")

    if job["route"] == "MEMORY_REVIEW":
        name = "review_outcome.json"
    elif job["route"] == "SKILL_REVIEW":
        name = "skill_review.json"
    else:
        raise ValueError("Unknown review route")

    return Path(root) / "reviews" / job_id / name


def build_scheduler(
    jobs,
    capabilities,
    protocol,
    root,
    *,
    enable_handlers=False,
    max_continuation_decisions=12,
    max_llm_calls=16,
):
    memory = OnlineMemoryHandler(
        capabilities,
        model=protocol["model"],
        max_steps=protocol["max_steps"],
        top_k=protocol["top_k"],
        max_continuation_decisions=max_continuation_decisions,
        max_llm_calls=max_llm_calls,
    )

    skill = OnlineSkillHandler(capabilities)

    def resolve(job):
        return outcome_path(root, job)

    def validate(job, path):
        record = read_json(path)
        schema = record.get("schema")

        if job["route"] == "MEMORY_REVIEW":
            if schema == "online_memory_outcome_v2":
                return memory.validate_outcome(job, path)

            if schema == "unified_memory_review_outcome_v1":
                return validate_legacy_outcome(job, path)

        elif job["route"] == "SKILL_REVIEW":
            if schema == "online_skill_diagnostic_v1":
                return skill.validate_outcome(job, path)

            if schema == "online_skill_review_outcome_v1":
                return validate_legacy_outcome(job, path)

        raise RuntimeError(
            f"Unsupported outcome schema: {job['job_id']}"
        )

    handlers = {}

    if enable_handlers:
        handlers = {
            "MEMORY_REVIEW": memory.process,
            "SKILL_REVIEW": skill.process,
        }

    return OnlineReviewScheduler(
        jobs,
        outcome_path=resolve,
        validate_outcome=validate,
        handlers=handlers,
    )


def execute(
    stage,
    max_new_jobs=0,
    max_continuation_decisions=12,
    max_llm_calls=16,
):
    if stage not in ("report", "run", "audit"):
        raise ValueError("Unknown workflow stage")

    if stage != "run" and max_new_jobs != 0:
        raise ValueError(
            "Only run stage may execute new review jobs"
        )

    protocol, capabilities = verify_protocol()
    plan = prepare_plan()

    if (
        plan["scope"] != "ONLINE_TRAIN_DEVELOPMENT"
        or plan["protocol_sha256"] != sha256(PROTOCOL_PATH)
        or plan["model"] != protocol["model"]
        or plan["memory_version"]
        != capabilities.memory_info.version
        or plan["memory_sha256"]
        != capabilities.memory_info.memory_hash
        or plan["active_skill_id"] != capabilities.skill_id
    ):
        raise RuntimeError("Frozen workflow protocol mismatch")

    jobs = plan["jobs"]

    if any(job["route"] not in ROUTES for job in jobs):
        raise RuntimeError("Unexpected review route")

    scheduler = build_scheduler(
        jobs,
        capabilities,
        protocol,
        ROOT,
        enable_handlers=(stage == "run"),
        max_continuation_decisions=max_continuation_decisions,
        max_llm_calls=max_llm_calls,
    )

    results = scheduler.run(
        max_new_jobs=max_new_jobs if stage == "run" else 0
    )

    counts = Counter(row["state"] for row in results)

    print("========== UNIFIED ONLINE WORKFLOW ==========")
    print("Stage:", stage)
    print("Active memory:", capabilities.memory_info.version)
    print("Active skill:", capabilities.skill_id)
    print("Episodes:", len(protocol["games"]))
    print("Review jobs:", len(results))

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

    if counts["EXECUTED"] == 0:
        print("New LLM calls: 0")
    else:
        print(
            "LLM usage: see individual Memory Review outcomes"
        )

    print("Automatic capability promotion: disabled")

    if stage == "audit":
        if (
            len(results) != 3
            or counts["CACHED"] != 3
            or counts["PENDING"] != 0
            or counts["EXECUTED"] != 0
        ):
            raise RuntimeError(
                "Frozen online Pilot has incomplete reviews"
            )

        summary = read_json(ROOT / "pilot_summary.json")

        if (
            summary["total_episodes"] != len(protocol["games"])
            or summary["successes"] != 5
            or summary["routes"]
            != dict(Counter(plan["routes"].values()))
            or summary["memory_promotions"] != 0
            or summary["skill_promotions"] != 0
            or summary["active_memory_version"]
            != capabilities.memory_info.version
            or summary["active_skill_id"]
            != capabilities.skill_id
            or summary["final_holdout_evaluated"] is not False
        ):
            raise RuntimeError("Frozen Pilot summary mismatch")

        print("Final audit: PASS")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        choices=("report", "run", "audit"),
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

    execute(
        stage=args.stage,
        max_new_jobs=args.max_new_jobs,
        max_continuation_decisions=args.max_continuation_decisions,
        max_llm_calls=args.max_llm_calls,
    )


if __name__ == "__main__":
    main()
