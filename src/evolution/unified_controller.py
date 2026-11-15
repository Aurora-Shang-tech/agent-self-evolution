import hashlib
import json
import os
import re
from pathlib import Path

from src.core.trajectory import Trajectory
from src.evolution.feedback_router import FeedbackRouter


REVIEW_ROUTES = {"MEMORY_REVIEW", "SKILL_REVIEW"}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class UnifiedFeedbackController:
    def __init__(self, capabilities, output_dir):
        self.capabilities = capabilities
        self.output_dir = Path(output_dir)
        self.event_dir = self.output_dir / "events"

    def process_saved_episode(
        self,
        *,
        episode_id,
        game,
        trajectory_path,
        router: FeedbackRouter,
        phase="learning",
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", episode_id):
            raise ValueError("Invalid episode ID")

        if phase == "learning" and game.split != "train":
            raise ValueError(
                "Only Train tasks can enter learning"
            )

        path = Path(trajectory_path)
        trace = Trajectory.load_json(str(path))

        if not trace.steps:
            raise ValueError("Empty episode trajectory")

        if (
            Path(trace.gamefile).resolve()
            != Path(game.game_file).resolve()
        ):
            raise ValueError("Episode game mismatch")

        decision = router.route(
            trace,
            game.task_type,
            phase=phase,
        )

        reference_config = {
            "references": router.references,
            "min_reference_samples": router.min_reference_samples,
            "min_extra_actions": router.min_extra_actions,
            "min_action_ratio": router.min_action_ratio,
        }

        event = {
            "schema": "unified_feedback_event_v1",
            "episode_id": episode_id,
            "game_id": game.id,
            "game_file": str(Path(game.game_file).resolve()),
            "task_type": game.task_type,
            "phase": phase,
            "trajectory_path": str(path),
            "trajectory_sha256": file_hash(path),
            "memory_version": (
                self.capabilities.memory_info.version
            ),
            "memory_sha256": (
                self.capabilities.memory_info.memory_hash
            ),
            "active_skill_id": self.capabilities.skill_id,
            "skill_mode": self.capabilities.skill_mode,
            "reference_config_sha256": fingerprint(
                reference_config
            ),
            "decision": decision.to_dict(),
        }

        output = self.event_dir / f"{episode_id}.json"

        if output.exists():
            if read_json(output) != event:
                raise RuntimeError(
                    f"Conflicting saved event: {episode_id}"
                )
            return event

        self.event_dir.mkdir(parents=True, exist_ok=True)

        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(event, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)

        return event

    def review_queue(self):
        results = []

        for path in sorted(self.event_dir.glob("*.json")):
            event = read_json(path)

            if event.get("schema") != "unified_feedback_event_v1":
                raise ValueError("Invalid feedback event schema")

            if path.stem != event.get("episode_id"):
                raise ValueError("Feedback event ID mismatch")

            if (
                file_hash(event["trajectory_path"])
                != event["trajectory_sha256"]
            ):
                raise ValueError(
                    "Feedback source trajectory changed"
                )

            if event["decision"]["route"] in REVIEW_ROUTES:
                results.append(event)

        return results
