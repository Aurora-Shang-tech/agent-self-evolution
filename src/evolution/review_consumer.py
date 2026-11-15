import hashlib
import json
import os
from collections import Counter
from pathlib import Path


HANDLERS = {
    "MEMORY_REVIEW": "memory_failure_analysis",
    "SKILL_REVIEW": "skill_inefficiency_analysis",
}


def fingerprint(data):
    content = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_immutable(path, data):
    path = Path(path)
    content = json.dumps(
        data, ensure_ascii=False, indent=2
    ) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(
                f"Existing review plan differs: {path}"
            )
        return

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


class ReviewQueueConsumer:
    def __init__(self, controller):
        self.controller = controller
        self.capabilities = controller.capabilities

    def prepare_historical(self):
        events = self.controller.review_queue()
        jobs = []
        seen = set()

        for event in sorted(
            events,
            key=lambda x: x["episode_id"],
        ):
            episode_id = event["episode_id"]
            route = event["decision"]["route"]

            if episode_id in seen:
                raise ValueError("Duplicate review episode")

            seen.add(episode_id)

            if route not in HANDLERS:
                raise ValueError("Unknown review route")

            if event["phase"] != "learning":
                raise ValueError(
                    "Evaluation episode in learning queue"
                )

            if not event["game_id"].startswith("train:"):
                raise ValueError("Non-Train review episode")

            if (
                event["memory_version"]
                != self.capabilities.memory_info.version
                or event["memory_sha256"]
                != self.capabilities.memory_info.memory_hash
            ):
                raise ValueError("Memory version mismatch")

            if (
                event["active_skill_id"]
                != self.capabilities.skill_id
                or event["skill_mode"]
                != self.capabilities.skill_mode
            ):
                raise ValueError("Active skill mismatch")

            success = event["decision"]["success"]

            if type(success) is not bool:
                raise ValueError("Invalid episode success")

            if (
                route == "MEMORY_REVIEW" and success
            ) or (
                route == "SKILL_REVIEW" and not success
            ):
                raise ValueError(
                    "Review route contradicts episode outcome"
                )

            jobs.append({
                "job_id": episode_id,
                "game_id": event["game_id"],
                "task_type": event["task_type"],
                "route": route,
                "handler": HANDLERS[route],
                "source_scope": "HISTORICAL_DIAGNOSTIC",
                "status": "REFERENCE_ONLY",
                "eligible_for_learning": False,
                "memory_version": event["memory_version"],
                "memory_sha256": event["memory_sha256"],
                "active_skill_id": event["active_skill_id"],
                "trajectory_path": event["trajectory_path"],
                "trajectory_sha256": event["trajectory_sha256"],
                "source_event_sha256": fingerprint(event),
            })

        counts = Counter(job["route"] for job in jobs)

        return {
            "schema": "unified_review_plan_v1",
            "scope": "HISTORICAL_DIAGNOSTIC",
            "active_memory_version": (
                self.capabilities.memory_info.version
            ),
            "active_memory_sha256": (
                self.capabilities.memory_info.memory_hash
            ),
            "active_skill_id": self.capabilities.skill_id,
            "summary": {
                "review_jobs": len(jobs),
                "memory_reviews": counts["MEMORY_REVIEW"],
                "skill_reviews": counts["SKILL_REVIEW"],
                "executable_jobs": 0,
            },
            "jobs": jobs,
        }
