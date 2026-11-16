import re
from pathlib import Path

from src.core.trajectory import Trajectory
from src.evolution.online_scheduler import OnlineReviewScheduler
from src.evolution.review_consumer import write_immutable
from src.evolution.unified_controller import read_json


class OnlineSkillHandler:
    STATUS = "DIAGNOSTIC_ONLY"

    def __init__(self, capabilities):
        self.capabilities = capabilities

    def build_record(self, job):
        OnlineReviewScheduler.verify_source(job)

        if job["route"] != "SKILL_REVIEW":
            raise ValueError("Expected SKILL_REVIEW")

        event = read_json(job["event_path"])
        decision = event["decision"]
        trace = Trajectory.load_json(job["trajectory_path"])

        if (
            not trace.success
            or not trace.steps
            or not trace.steps[-1].won
            or trace.task != decision["task"]
        ):
            raise ValueError("Invalid successful trajectory")

        if (
            event["memory_version"]
            != self.capabilities.memory_info.version
            or event["memory_sha256"]
            != self.capabilities.memory_info.memory_hash
            or event["active_skill_id"]
            != self.capabilities.skill_id
        ):
            raise RuntimeError("Active capability mismatch")

        actions = sum(
            bool(step.valid_format)
            for step in trace.steps
        )

        if actions != decision["environment_actions"]:
            raise RuntimeError("Environment action count mismatch")

        first_take = None
        first_object = None

        for index, step in enumerate(trace.steps):
            if not step.valid_format or not step.admissible:
                continue

            match = re.fullmatch(
                r"take ([a-z][a-z0-9]* \d+) from (.+)",
                step.action.strip().lower(),
            )

            if match:
                first_take = index
                first_object = match.group(1)
                break

        search_end = (
            first_take if first_take is not None
            else len(trace.steps)
        )

        locations = []
        seen_locations = set()

        for step in trace.steps[:search_end]:
            if not step.valid_format or not step.admissible:
                continue

            match = re.fullmatch(
                r"(?:go to|open) ([a-z][a-z0-9]* \d+)",
                step.action.strip().lower(),
            )

            if match:
                location = match.group(1)

                if location not in seen_locations:
                    seen_locations.add(location)
                    locations.append(location)

        first_seen = None

        if first_object is not None:
            pattern = re.compile(
                r"(?<![a-z0-9_])"
                + re.escape(first_object)
                + r"(?![a-z0-9_])",
                re.IGNORECASE,
            )

            for index, step in enumerate(
                trace.steps[:first_take]
            ):
                if pattern.search(step.next_observation):
                    first_seen = index
                    break

        sighting = None

        if first_seen is not None:
            step = trace.steps[first_seen]
            sighting = {
                "step_index": first_seen,
                "action": step.action,
                "object": first_object,
            }

        return {
            "schema": "online_skill_diagnostic_v1",
            "job_id": job["job_id"],
            "game_id": job["game_id"],
            "status": self.STATUS,
            "trajectory_sha256": job["trajectory_sha256"],
            "event_sha256": job["event_sha256"],
            "memory_version":
                self.capabilities.memory_info.version,
            "memory_sha256":
                self.capabilities.memory_info.memory_hash,
            "active_skill_id": self.capabilities.skill_id,
            "task": trace.task,
            "success": True,
            "environment_actions": actions,
            "reference_median": decision["reference_median"],
            "reference_samples": decision["reference_samples"],
            "pre_first_take_steps": search_end,
            "first_taken_object": first_object,
            "first_take_step_index": first_take,
            "first_object_sighting": sighting,
            "searched_locations": locations,
            "distinct_searched_locations": len(locations),
            "diagnosis": (
                "Search overhead observed; no verified "
                "alternative policy in this review."
            ),
            "candidate_generated": False,
            "skill_promoted": False,
            "active_skill_changed": False,
        }

    def validate_outcome(self, job, path):
        actual = read_json(path)
        expected = self.build_record(job)

        if actual != expected:
            raise RuntimeError(
                f"Skill diagnostic mismatch: {job['job_id']}"
            )

        return actual["status"]

    def process(self, job, output_path):
        output_path = Path(output_path)

        if output_path.exists():
            return self.validate_outcome(job, output_path)

        record = self.build_record(job)
        write_immutable(output_path, record)

        return self.validate_outcome(job, output_path)
