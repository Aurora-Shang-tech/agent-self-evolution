import hashlib
import json
from pathlib import Path


ROUTES = {"MEMORY_REVIEW", "SKILL_REVIEW"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class OnlineReviewScheduler:
    def __init__(
        self,
        jobs,
        *,
        outcome_path,
        validate_outcome,
        handlers=None,
    ):
        self.jobs = tuple(jobs)
        self.outcome_path = outcome_path
        self.validate_outcome = validate_outcome
        self.handlers = handlers or {}

    @staticmethod
    def verify_source(job):
        job_id = job["job_id"]
        route = job["route"]

        if (
            not isinstance(job_id, str)
            or not job_id
            or route not in ROUTES
            or job["source_scope"] != "ONLINE_TRAIN_DEVELOPMENT"
            or not job["game_id"].startswith("train:")
        ):
            raise ValueError(
                "Review job is not an authorized Train job"
            )

        trace_path = Path(job["trajectory_path"])
        event_path = Path(job["event_path"])

        if (
            sha256(trace_path) != job["trajectory_sha256"]
            or sha256(event_path) != job["event_sha256"]
        ):
            raise RuntimeError(
                f"Review evidence changed: {job_id}"
            )

        event = load_json(event_path)
        decision = event["decision"]

        if (
            event.get("schema") != "unified_feedback_event_v1"
            or event.get("episode_id") != job_id
            or event.get("game_id") != job["game_id"]
            or event.get("phase") != "learning"
            or Path(event["trajectory_path"]).resolve()
            != trace_path.resolve()
            or event["trajectory_sha256"]
            != job["trajectory_sha256"]
            or decision["route"] != route
            or type(decision["success"]) is not bool
            or (
                route == "MEMORY_REVIEW"
                and decision["success"]
            )
            or (
                route == "SKILL_REVIEW"
                and not decision["success"]
            )
        ):
            raise ValueError(
                f"Review event mismatch: {job_id}"
            )

    def inspect(self):
        seen = set()
        entries = []

        for job in self.jobs:
            job_id = job["job_id"]

            if job_id in seen:
                raise ValueError(
                    f"Duplicate review job: {job_id}"
                )

            seen.add(job_id)
            self.verify_source(job)

            path = Path(self.outcome_path(job))

            if path.exists():
                if not path.is_file():
                    raise RuntimeError(
                        f"Invalid outcome path: {path}"
                    )

                status = self.validate_outcome(job, path)

                if not isinstance(status, str) or not status:
                    raise ValueError(
                        f"Invalid outcome status: {job_id}"
                    )

                entries.append((job, path, status))
            else:
                entries.append((job, path, None))

        return entries

    def run(self, *, max_new_jobs=0):
        if (
            type(max_new_jobs) is not int
            or max_new_jobs < 0
        ):
            raise ValueError(
                "max_new_jobs must be a nonnegative integer"
            )

        entries = self.inspect()
        results = []
        new_jobs = 0

        for job, path, status in entries:
            if status is not None:
                state = "CACHED"

            elif new_jobs >= max_new_jobs:
                state = "PENDING"

            else:
                handler = self.handlers.get(job["route"])

                if handler is None:
                    raise RuntimeError(
                        "No authorized handler for "
                        f"{job['route']}"
                    )

                handler(job, path)

                if not path.is_file():
                    raise RuntimeError(
                        "Handler did not save its outcome: "
                        f"{job['job_id']}"
                    )

                status = self.validate_outcome(
                    job, path
                )

                if (
                    not isinstance(status, str)
                    or not status
                ):
                    raise ValueError(
                        "Handler saved an invalid outcome"
                    )

                new_jobs += 1
                state = "EXECUTED"

            results.append({
                "job_id": job["job_id"],
                "route": job["route"],
                "state": state,
                "outcome_status": status,
            })

        return results
