import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.review_unified_online import (
    PLAN_PATH,
    prepare_plan,
)
from scripts.run_unified_pilot import ROOT, verify_protocol
from src.core.trajectory import Trajectory
from src.evolution.online_scheduler import (
    OnlineReviewScheduler,
    sha256,
)


MEMORY_STATUS = "STOPPED_WITH_UNTESTED_CANDIDATE"
SKILL_STATUS = "NO_CANDIDATE_IN_PILOT"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def outcome_path(job):
    name = (
        "review_outcome.json"
        if job["route"] == "MEMORY_REVIEW"
        else "skill_review.json"
    )
    return ROOT / "reviews" / job["job_id"] / name


def validate_skill(job, record):
    check(
        record.get("schema")
        == "online_skill_review_outcome_v1",
        "Invalid Skill Review schema",
    )

    check(
        record.get("status") == SKILL_STATUS,
        "Unexpected Skill Review status",
    )

    check(
        record.get("trajectory_sha256")
        == job["trajectory_sha256"]
        and record.get("event_sha256")
        == job["event_sha256"],
        "Skill Review evidence mismatch",
    )

    check(
        record.get("success") is True
        and record.get("candidate_generated") is False
        and record.get("counterfactual_verified") is False
        and record.get("skill_promoted") is False
        and record.get("active_skill_changed") is False,
        "Invalid Skill Review decision",
    )

    trace = Trajectory.load_json(job["trajectory_path"])

    count = sum(
        bool(step.valid_format)
        for step in trace.steps
    )

    check(
        trace.success
        and count == record["environment_actions"],
        "Skill Review trajectory mismatch",
    )

    sighting = record.get("first_target_sighting")

    if sighting is not None:
        index = sighting["step_index"]

        check(
            type(index) is int
            and 0 <= index < len(trace.steps),
            "Invalid target sighting index",
        )

        step = trace.steps[index]

        check(
            step.action == sighting["action"]
            and sighting["target"] in step.next_observation,
            "Invalid target sighting evidence",
        )

    return record["status"]


def validate_memory(job, record, folder):
    check(
        record.get("schema")
        == "unified_memory_review_outcome_v1",
        "Invalid Memory Review schema",
    )

    check(
        record.get("status") == MEMORY_STATUS
        and record.get("memory_operation") == "NO_CHANGE"
        and record.get("verified_successes") == 0
        and record.get("lesson_extracted") is False
        and record.get("active_skill_id") is None,
        "Unexpected Memory Review decision",
    )

    cf_path = folder / "counterfactual_plan.json"
    analysis_path = folder / "failure_analysis.json"

    check(
        record.get("counterfactual_plan_sha256")
        == sha256(cf_path),
        "Counterfactual plan hash mismatch",
    )

    cf = read_json(cf_path)

    check(
        cf.get("job_id") == job["job_id"]
        and cf.get("status") == "FROZEN"
        and cf.get("trajectory_sha256")
        == job["trajectory_sha256"]
        and cf.get("event_sha256")
        == job["event_sha256"]
        and cf.get("review_plan_sha256")
        == sha256(PLAN_PATH)
        and cf.get("analysis_sha256")
        == sha256(analysis_path)
        and record.get("memory_version_after")
        == cf.get("memory_version")
        and cf.get("active_skill_id") is None,
        "Memory Review provenance mismatch",
    )

    for path, expected_hash in cf["code_sha256"].items():
        check(
            sha256(path) == expected_hash,
            f"Counterfactual code changed: {path}",
        )

    verified = record["verified_branches"]
    untested = record["untested_candidates"]
    candidates = cf["candidate_actions"]
    seen = set()

    for item in verified:
        index = item["candidate_index"]

        check(
            type(index) is int
            and 0 <= index < len(candidates)
            and index not in seen,
            "Invalid verified candidate index",
        )
        seen.add(index)

        path = (
            folder / "counterfactuals"
            / f"branch_{index:02d}.json"
        )

        check(
            item["evidence_sha256"] == sha256(path),
            "Counterfactual branch hash mismatch",
        )

        branch = read_json(path)

        check(
            branch["candidate_index"] == index
            and branch["candidate_action"]
            == candidates[index]
            and branch["status"]
            == "VERIFIED_IN_ALFWORLD"
            and branch["verification_plan_sha256"]
            == sha256(cf_path)
            and branch["success"] is False,
            "Invalid counterfactual branch evidence",
        )

    for item in untested:
        index = item["candidate_index"]

        check(
            type(index) is int
            and 0 <= index < len(candidates)
            and index not in seen
            and item["action"] == candidates[index],
            "Invalid untested candidate",
        )
        seen.add(index)

        path = (
            folder / "counterfactuals"
            / f"branch_{index:02d}.json"
        )

        check(
            not path.exists(),
            "Untested candidate already has a branch",
        )

    check(
        seen == set(range(len(candidates)))
        and len(verified) > 0,
        "Incomplete Memory Review accounting",
    )

    return record["status"]


def validate_outcome(job, path):
    record = read_json(path)

    check(
        record.get("job_id") == job["job_id"],
        "Review outcome job mismatch",
    )

    if job["route"] == "MEMORY_REVIEW":
        return validate_memory(
            job, record, path.parent
        )

    if job["route"] == "SKILL_REVIEW":
        return validate_skill(job, record)

    raise ValueError("Unknown Review route")


def run(stage):
    protocol, capabilities = verify_protocol()
    plan = prepare_plan()

    check(
        plan["protocol_sha256"]
        == sha256(ROOT / "protocol.json"),
        "Frozen protocol mismatch",
    )

    check(
        plan["memory_version"]
        == capabilities.memory_info.version
        and plan["memory_sha256"]
        == capabilities.memory_info.memory_hash
        and plan["active_skill_id"]
        == capabilities.skill_id,
        "Active capabilities mismatch",
    )

    scheduler = OnlineReviewScheduler(
        plan["jobs"],
        outcome_path=outcome_path,
        validate_outcome=validate_outcome,
    )

    results = scheduler.run(max_new_jobs=0)

    counts = Counter(
        item["state"] for item in results
    )

    print("========== UNIFIED WORKFLOW ==========")
    print("Memory:", capabilities.memory_info.version)
    print("Active skill:", capabilities.skill_id)
    print("Episodes:", len(protocol["games"]))
    print("Review jobs:", len(results))

    for item in results:
        print(
            item["job_id"],
            item["route"],
            item["state"],
            item["outcome_status"],
        )

    print("Cached:", counts["CACHED"])
    print("Pending:", counts["PENDING"])
    print("New jobs executed: 0")
    print("LLM calls: 0")

    if stage == "audit":
        check(
            counts["CACHED"] == len(results)
            and len(results) == 3,
            "Online Pilot is not fully processed",
        )

        summary = read_json(
            ROOT / "pilot_summary.json"
        )

        check(
            summary["total_episodes"] == len(protocol["games"])
            and summary["successes"] == 5
            and summary["routes"]
            == dict(Counter(plan["routes"].values()))
            and summary["memory_promotions"] == 0
            and summary["skill_promotions"] == 0
            and summary["active_memory_version"]
            == capabilities.memory_info.version
            and summary["active_skill_id"]
            == capabilities.skill_id
            and summary["final_holdout_evaluated"] is False,
            "Pilot summary mismatch",
        )

        print("Final audit: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("report", "audit"),
        required=True,
    )
    args = parser.parse_args()

    run(args.stage)


if __name__ == "__main__":
    main()
