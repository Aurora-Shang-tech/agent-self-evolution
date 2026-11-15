import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from scripts.run_unified_pilot import (
    ROOT,
    PROTOCOL_PATH,
    verify_protocol,
)
from src.analysis.failure_analyzer import FailureAnalyzer
from src.core.trajectory import Trajectory
from src.data.alfworld_data import GameExample
from src.evolution.feedback_router import FeedbackRouter
from src.evolution.unified_controller import (
    UnifiedFeedbackController,
)
from src.evolution.review_consumer import write_immutable


PLAN_PATH = ROOT / "online_review_plan.json"

HANDLERS = {
    "MEMORY_REVIEW": "failure_analyzer",
    "SKILL_REVIEW": "skill_inefficiency_analysis",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare_plan():
    protocol, capabilities = verify_protocol()

    controller = UnifiedFeedbackController(
        capabilities, ROOT
    )

    router = FeedbackRouter(
        protocol["reference_actions_by_type"]
    )

    jobs = []
    routes = {}
    seen_games = set()

    for index, raw in enumerate(protocol["games"]):
        if raw["id"] in seen_games:
            raise ValueError("Duplicate game in protocol")

        seen_games.add(raw["id"])

        game = GameExample(
            id=raw["id"],
            split=raw["split"],
            game_file=Path(raw["game_file"]),
            task_type=raw["task_type"],
        )

        episode_id = f"fresh_{index:02d}"
        trace_path = (
            ROOT / "episodes"
            / f"episode_{index:02d}.json"
        )
        event_path = (
            ROOT / "events"
            / f"{episode_id}.json"
        )

        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)

        if not event_path.is_file():
            raise FileNotFoundError(event_path)

        event = controller.process_saved_episode(
            episode_id=episode_id,
            game=game,
            trajectory_path=trace_path,
            router=router,
            phase="learning",
        )

        trace = Trajectory.load_json(str(trace_path))

        if (
            not trace.steps
            or trace.task != event["decision"]["task"]
            or trace.success != event["decision"]["success"]
            or trace.success != bool(trace.steps[-1].won)
        ):
            raise ValueError(
                f"Episode outcome mismatch: {episode_id}"
            )

        if (
            event["memory_version"]
            != capabilities.memory_info.version
            or event["memory_sha256"]
            != capabilities.memory_info.memory_hash
            or event["active_skill_id"]
            != capabilities.skill_id
        ):
            raise ValueError("Active capability mismatch")

        route = event["decision"]["route"]
        routes[episode_id] = route

        if route == "ARCHIVE":
            continue

        if route not in HANDLERS:
            raise ValueError(f"Unsupported route: {route}")

        if (
            route == "MEMORY_REVIEW" and trace.success
        ) or (
            route == "SKILL_REVIEW" and not trace.success
        ):
            raise ValueError("Route/outcome mismatch")

        jobs.append({
            "job_id": episode_id,
            "game_id": game.id,
            "route": route,
            "handler": HANDLERS[route],
            "source_scope": "ONLINE_TRAIN_DEVELOPMENT",
            "status": "PENDING",
            "trajectory_path": str(trace_path),
            "trajectory_sha256": sha(trace_path),
            "event_path": str(event_path),
            "event_sha256": sha(event_path),
        })

    if len(routes) != 6:
        raise ValueError("Incomplete online pilot")

    plan = {
        "schema": "online_review_plan_v1",
        "scope": "ONLINE_TRAIN_DEVELOPMENT",
        "protocol_sha256": sha(PROTOCOL_PATH),
        "memory_version": capabilities.memory_info.version,
        "memory_sha256": capabilities.memory_info.memory_hash,
        "active_skill_id": capabilities.skill_id,
        "model": protocol["model"],
        "failure_analyzer_sha256": sha(
            "src/analysis/failure_analyzer.py"
        ),
        "routes": routes,
        "jobs": jobs,
    }

    write_immutable(PLAN_PATH, plan)
    return plan


def analyze_memory():
    if not PLAN_PATH.is_file():
        raise RuntimeError("Run --stage plan first")

    plan = prepare_plan()

    if sha("src/analysis/failure_analyzer.py") != (
        plan["failure_analyzer_sha256"]
    ):
        raise RuntimeError("Failure analyzer code changed")

    jobs = [
        job for job in plan["jobs"]
        if job["route"] == "MEMORY_REVIEW"
    ]

    if len(jobs) != 1:
        raise RuntimeError(
            f"Expected one memory review, got {len(jobs)}"
        )

    job = jobs[0]
    output = (
        ROOT / "reviews" / job["job_id"]
        / "failure_analysis.json"
    )

    if output.exists():
        saved = read_json(output)

        if (
            saved["trajectory_sha256"]
            != job["trajectory_sha256"]
            or saved["event_sha256"]
            != job["event_sha256"]
            or saved["protocol_sha256"]
            != plan["protocol_sha256"]
        ):
            raise RuntimeError("Cached analysis source mismatch")

        print("[cached]", job["job_id"])
        print(json.dumps(
            saved["analysis"],
            ensure_ascii=False,
            indent=2,
        ))
        return

    if os.getenv("OPENAI_MODEL") != plan["model"]:
        raise RuntimeError(
            f"OPENAI_MODEL must be {plan['model']}"
        )

    trace = Trajectory.load_json(
        job["trajectory_path"]
    )

    if trace.success:
        raise ValueError("Memory review must be a failure")

    from src.llm.client import LLMClient

    analyzer = FailureAnalyzer(LLMClient())

    analysis = analyzer.analyze(
        trace,
        max_retries=2,
    )

    record = {
        "schema": "online_failure_analysis_v1",
        "job_id": job["job_id"],
        "status": "ANALYZED_NOT_VERIFIED",
        "protocol_sha256": plan["protocol_sha256"],
        "trajectory_sha256": job["trajectory_sha256"],
        "event_sha256": job["event_sha256"],
        "memory_version": plan["memory_version"],
        "model": plan["model"],
        "analysis": asdict(analysis),
        "counterfactual_verified": False,
        "memory_updated": False,
    }

    write_immutable(output, record)

    print("\n========== ONLINE MEMORY REVIEW ==========")
    print("Job:", job["job_id"])
    print("Critical step:", analysis.critical_step)
    print("Failure type:", analysis.failure_type)
    print("Original action:", analysis.original_action)
    print(
        "Counterfactual candidates:",
        list(analysis.counterfactual_actions),
    )
    print("Status: ANALYZED_NOT_VERIFIED")
    print("Active memory unchanged:", plan["memory_version"])
    print("Saved:", output)


def report():
    plan = prepare_plan()

    memory_jobs = sum(
        j["route"] == "MEMORY_REVIEW"
        for j in plan["jobs"]
    )
    skill_jobs = sum(
        j["route"] == "SKILL_REVIEW"
        for j in plan["jobs"]
    )

    print("\n========== ONLINE REVIEW STATUS ==========")
    print("Memory version:", plan["memory_version"])
    print("Active skill:", plan["active_skill_id"])
    print("Memory jobs:", memory_jobs)
    print("Skill jobs:", skill_jobs)

    for job in plan["jobs"]:
        analysis_path = (
            ROOT / "reviews" / job["job_id"]
            / "failure_analysis.json"
        )

        status = (
            "ANALYZED_NOT_VERIFIED"
            if job["route"] == "MEMORY_REVIEW"
            and analysis_path.is_file()
            else "PENDING"
        )

        print(job["job_id"], job["route"], status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("plan", "analyze-memory", "report"),
        required=True,
    )

    args = parser.parse_args()

    if args.stage == "plan":
        plan = prepare_plan()

        print("\n========== ONLINE REVIEW PLAN ==========")
        print("Scope:", plan["scope"])
        print("Review jobs:", len(plan["jobs"]))
        for job in plan["jobs"]:
            print(
                job["job_id"],
                job["route"],
                job["handler"],
            )
        print("LLM calls: 0")

    elif args.stage == "analyze-memory":
        analyze_memory()

    else:
        report()


if __name__ == "__main__":
    main()
