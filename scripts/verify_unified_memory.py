import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from scripts.review_unified_online import (
    PLAN_PATH as REVIEW_PLAN_PATH,
    prepare_plan,
)
from scripts.run_unified_pilot import ROOT, verify_protocol
from src.core.trajectory import Trajectory
from src.counterfactual.verifier import verify_counterfactual
from src.evolution.capabilities import build_active_agent
from src.evolution.review_consumer import write_immutable


JOB_ID = "fresh_04"
FOLDER = ROOT / "reviews" / JOB_ID
ANALYSIS_PATH = FOLDER / "failure_analysis.json"
CF_PLAN_PATH = FOLDER / "counterfactual_plan.json"
BRANCH_DIR = FOLDER / "counterfactuals"

CODE_FILES = (
    "scripts/verify_unified_memory.py",
    "scripts/review_unified_online.py",
    "src/agents/react_agent.py",
    "src/core/trajectory.py",
    "src/core/protocol.py",
    "src/counterfactual/verifier.py",
    "src/counterfactual/replay.py",
    "src/envs/environment.py",
    "src/memory/state_tracker.py",
    "src/evolution/capabilities.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def validate_case(trace, analysis):
    if trace.success or not trace.steps:
        raise ValueError("Expected a nonempty failed trajectory")

    if trace.steps[-1].won:
        raise ValueError("Failure trajectory contains a final win")

    index = analysis.get("critical_step")

    if type(index) is not int or not 0 <= index < len(trace.steps):
        raise ValueError("Invalid critical step")

    original = trace.steps[index]

    if analysis.get("original_action") != original.action:
        raise ValueError("Original action mismatch")

    actions = analysis.get("counterfactual_actions")

    if not isinstance(actions, list) or not 1 <= len(actions) <= 3:
        raise ValueError("Invalid candidate list")

    if len(set(actions)) != len(actions):
        raise ValueError("Duplicate candidate action")

    for action in actions:
        if (
            not isinstance(action, str)
            or action == original.action
            or action not in original.admissible_actions
        ):
            raise ValueError(
                f"Invalid counterfactual action: {action}"
            )

    if len(trace.messages) < 2 + 2 * index:
        raise ValueError("Missing original LLM context")

    return index, actions


def current_plan():
    protocol, capabilities = verify_protocol()
    review = prepare_plan()

    jobs = [
        job for job in review["jobs"]
        if job["job_id"] == JOB_ID
    ]

    if len(jobs) != 1 or jobs[0]["route"] != "MEMORY_REVIEW":
        raise ValueError("Expected one online memory review job")

    job = jobs[0]

    if (
        review["memory_version"] != capabilities.memory_info.version
        or review["memory_sha256"] != capabilities.memory_info.memory_hash
        or capabilities.skill_id is not None
    ):
        raise ValueError("Active capabilities mismatch")

    if sha(job["trajectory_path"]) != job["trajectory_sha256"]:
        raise ValueError("Trajectory hash mismatch")

    if sha(job["event_path"]) != job["event_sha256"]:
        raise ValueError("Event hash mismatch")

    record = read_json(ANALYSIS_PATH)

    if (
        record.get("status") != "ANALYZED_NOT_VERIFIED"
        or record.get("job_id") != JOB_ID
        or record.get("trajectory_sha256")
        != job["trajectory_sha256"]
        or record.get("event_sha256")
        != job["event_sha256"]
        or record.get("protocol_sha256")
        != review["protocol_sha256"]
        or record.get("model") != protocol["model"]
        or record.get("memory_version")
        != capabilities.memory_info.version
    ):
        raise ValueError("Failure analysis provenance mismatch")

    trace = Trajectory.load_json(job["trajectory_path"])
    index, actions = validate_case(trace, record["analysis"])

    expected_game = next(
        (
            game for game in protocol["games"]
            if game["id"] == job["game_id"]
        ),
        None,
    )

    if (
        expected_game is None
        or Path(trace.gamefile).resolve()
        != Path(expected_game["game_file"]).resolve()
    ):
        raise ValueError("Counterfactual game mismatch")

    plan = {
        "schema": "online_counterfactual_plan_v1",
        "status": "FROZEN",
        "job_id": JOB_ID,
        "model": protocol["model"],
        "max_steps": protocol["max_steps"],
        "top_k": protocol["top_k"],
        "memory_version": capabilities.memory_info.version,
        "memory_sha256": capabilities.memory_info.memory_hash,
        "active_skill_id": None,
        "review_plan_sha256": sha(REVIEW_PLAN_PATH),
        "analysis_sha256": sha(ANALYSIS_PATH),
        "trajectory_sha256": job["trajectory_sha256"],
        "event_sha256": job["event_sha256"],
        "critical_step": index,
        "original_action": trace.steps[index].action,
        "candidate_actions": actions,
        "code_sha256": {
            path: sha(path) for path in CODE_FILES
        },
    }

    return plan, trace, capabilities


def prepare():
    plan, trace, capabilities = current_plan()
    write_immutable(CF_PLAN_PATH, plan)

    if read_json(CF_PLAN_PATH) != plan:
        raise RuntimeError("Frozen verification plan differs")

    return plan, trace, capabilities


def branch_path(index):
    return BRANCH_DIR / f"branch_{index:02d}.json"


def validate_saved_branch(record, plan, index):
    if (
        record.get("schema") != "online_counterfactual_branch_v1"
        or record.get("verification_plan_sha256")
        != sha(CF_PLAN_PATH)
        or record.get("candidate_index") != index
        or record.get("candidate_action")
        != plan["candidate_actions"][index]
        or record.get("critical_step") != plan["critical_step"]
        or record.get("original_action") != plan["original_action"]
        or record.get("status") != "VERIFIED_IN_ALFWORLD"
    ):
        raise RuntimeError("Cached branch provenance mismatch")


def run_candidate(index):
    if not CF_PLAN_PATH.is_file():
        raise RuntimeError("Run --stage plan first")

    plan, trace, capabilities = prepare()

    if not 0 <= index < len(plan["candidate_actions"]):
        raise ValueError("Candidate index out of range")

    output = branch_path(index)

    if output.exists():
        saved = read_json(output)
        validate_saved_branch(saved, plan, index)
        print("[cached]", plan["candidate_actions"][index])
        print(
            "Success:", saved["success"],
            "Continuation steps:", saved["continuation_steps"],
        )
        return

    if os.getenv("OPENAI_MODEL") != plan["model"]:
        raise RuntimeError("Incorrect LLM model")

    from src.llm.client import LLMClient

    agent = build_active_agent(
        LLMClient(),
        capabilities,
        max_steps=plan["max_steps"],
        top_k=plan["top_k"],
    )

    action = plan["candidate_actions"][index]

    result = verify_counterfactual(
        trajectory=trace,
        critical_step=plan["critical_step"],
        counterfactual_action=action,
        agent=agent,
        max_steps=plan["max_steps"],
    )

    if (
        result.critical_step != plan["critical_step"]
        or result.original_action != plan["original_action"]
        or result.counterfactual_action != action
    ):
        raise RuntimeError("Counterfactual result mismatch")

    record = {
        "schema": "online_counterfactual_branch_v1",
        "status": "VERIFIED_IN_ALFWORLD",
        "verification_plan_sha256": sha(CF_PLAN_PATH),
        "candidate_index": index,
        "critical_step": result.critical_step,
        "original_action": result.original_action,
        "candidate_action": result.counterfactual_action,
        "success": result.success,
        "continuation_steps": result.continuation_steps,
        "agent_llm_calls": agent.llm_calls,
        "trajectory": (
            asdict(result.trajectory)
            if result.trajectory is not None
            else None
        ),
    }

    write_immutable(output, record)

    print("\n========== COUNTERFACTUAL VERIFICATION ==========")
    print("Job:", JOB_ID)
    print("Candidate:", index, action)
    print("Critical step:", result.critical_step)
    print("Original action:", result.original_action)
    print("Success:", result.success)
    print("Continuation steps:", result.continuation_steps)
    print("Agent LLM calls:", agent.llm_calls)
    print("Saved:", output)
    print("Active memory: unchanged")


def report():
    if not CF_PLAN_PATH.is_file():
        raise RuntimeError("Run --stage plan first")

    plan, _, _ = prepare()

    print("\n========== COUNTERFACTUAL STATUS ==========")
    print("Job:", plan["job_id"])
    print("Memory:", plan["memory_version"])
    print("Critical step:", plan["critical_step"])

    for index, action in enumerate(plan["candidate_actions"]):
        path = branch_path(index)

        if not path.is_file():
            print(index, action, "PENDING")
            continue

        record = read_json(path)
        validate_saved_branch(record, plan, index)

        print(
            index,
            action,
            "success=", record["success"],
            "continuation=", record["continuation_steps"],
            "llm=", record["agent_llm_calls"],
        )

    print("Memory updated: False")
    print("Active skill:", plan["active_skill_id"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("plan", "run", "report"),
    )
    parser.add_argument(
        "--candidate-index",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    if args.stage == "plan":
        plan, trace, _ = prepare()
        step = trace.steps[plan["critical_step"]]

        print("\n========== COUNTERFACTUAL PLAN ==========")
        print("Task:", trace.task)
        print("Critical step:", plan["critical_step"])
        print("Original action:", plan["original_action"])
        print(
            "Observation:",
            " ".join(step.observation.split())[:400],
        )
        print("Candidates:", plan["candidate_actions"])
        print("Frozen plan:", CF_PLAN_PATH)
        print("LLM calls: 0")

    elif args.stage == "run":
        run_candidate(args.candidate_index)

    else:
        report()


if __name__ == "__main__":
    main()
