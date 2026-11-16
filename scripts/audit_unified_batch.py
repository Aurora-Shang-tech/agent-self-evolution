from collections import Counter
from pathlib import Path

from scripts.run_unified_batch import (
    ROOT,
    PROTOCOL_PATH,
    verify_batch,
    collect_jobs,
)
from scripts.run_unified_online import build_scheduler
from src.core.trajectory import Trajectory
from src.evolution.feedback_router import FeedbackRouter
from src.evolution.online_scheduler import sha256
from src.evolution.review_consumer import write_immutable
from src.evolution.unified_controller import read_json


def main():
    protocol, caps = verify_batch()
    jobs, event_count = collect_jobs(protocol, caps)

    router = FeedbackRouter(
        protocol["reference_actions_by_type"]
    )

    episodes = []
    routes = Counter()
    successes = 0
    total_actions = 0

    assert event_count == len(protocol["games"]) == 6

    for index, game in enumerate(protocol["games"]):
        episode_id = f"batch_{index:02d}"

        trace_path = (
            ROOT / "episodes"
            / f"episode_{index:02d}.json"
        )
        event_path = (
            ROOT / "events"
            / f"{episode_id}.json"
        )

        trace = Trajectory.load_json(str(trace_path))
        event = read_json(event_path)

        assert trace.steps
        assert trace.success == bool(trace.steps[-1].won)

        assert event["schema"] == "unified_feedback_event_v1"
        assert event["episode_id"] == episode_id
        assert event["game_id"] == game["id"]
        assert event["phase"] == "learning"
        assert event["trajectory_sha256"] == sha256(trace_path)
        assert event["memory_version"] == caps.memory_info.version
        assert event["memory_sha256"] == caps.memory_info.memory_hash
        assert event["active_skill_id"] == caps.skill_id

        decision = router.route(
            trace, game["task_type"], phase="learning"
        )

        assert event["decision"] == decision.to_dict()

        actions = sum(
            bool(step.valid_format)
            for step in trace.steps
        )

        successes += int(trace.success)
        total_actions += actions
        routes[decision.route] += 1

        episodes.append({
            "episode_id": episode_id,
            "game_id": game["id"],
            "task_type": game["task_type"],
            "success": trace.success,
            "environment_actions": actions,
            "route": decision.route,
            "trajectory_sha256": sha256(trace_path),
            "event_sha256": sha256(event_path),
        })

    scheduler = build_scheduler(
        jobs,
        caps,
        protocol,
        ROOT,
        enable_handlers=False,
    )

    results = scheduler.run(max_new_jobs=0)

    assert successes == 6
    assert total_actions == 88
    assert routes == {
        "ARCHIVE": 4,
        "SKILL_REVIEW": 2,
    }
    assert len(results) == 2

    reviews = []

    for item in results:
        assert item["state"] == "CACHED"
        assert item["route"] == "SKILL_REVIEW"
        assert item["outcome_status"] == "DIAGNOSTIC_ONLY"

        path = (
            ROOT / "reviews"
            / item["job_id"]
            / "skill_review.json"
        )

        record = read_json(path)

        assert record["candidate_generated"] is False
        assert record["skill_promoted"] is False

        reviews.append({
            "job_id": item["job_id"],
            "status": item["outcome_status"],
            "outcome_sha256": sha256(path),
        })

    assert {r["job_id"] for r in reviews} == {
        "batch_01", "batch_03"
    }

    summary = {
        "schema": "unified_online_batch_audit_v1",
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "source_split": "train",
        "total_episodes": len(episodes),
        "successes": successes,
        "total_environment_actions": total_actions,
        "mean_environment_actions": (
            total_actions / len(episodes)
        ),
        "routes": dict(sorted(routes.items())),
        "memory_version": caps.memory_info.version,
        "memory_sha256": caps.memory_info.memory_hash,
        "active_skill_id": caps.skill_id,
        "memory_promotions": 0,
        "skill_promotions": 0,
        "new_memory_review_executed": False,
        "final_holdout_evaluated": False,
        "episodes": episodes,
        "reviews": reviews,
    }

    output = ROOT / "batch_audit.json"
    write_immutable(output, summary)

    print("========== UNIFIED BATCH FINAL AUDIT ==========")
    print("Episodes:", len(episodes))
    print("Successes:", successes)
    print("Average actions:", total_actions / len(episodes))
    print("Routes:", dict(routes))
    print("Reviews:", len(reviews))
    print("Review states:", [r["status"] for r in reviews])
    print("Memory:", caps.memory_info.version)
    print("Active skill:", caps.skill_id)
    print("Promotions: 0")
    print("Final Holdout: untouched")
    print("Audit: PASS")
    print("Saved:", output)
    print("New LLM calls: 0")


if __name__ == "__main__":
    main()
