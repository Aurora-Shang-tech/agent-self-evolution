"""检查从LLM轨迹中总结出来的经验，是否与实际环境反馈相矛盾"""


import re
from dataclasses import dataclass

from src.core.trajectory import Trajectory
from src.memory.state_tracker import TaskStateTracker


@dataclass(frozen=True)
class EvidenceConflict:
    source_action: str
    winning_action: str
    reason: str


class EvidenceConsistencyChecker:

    @staticmethod
    def _is_equivalence_claim(lesson: str) -> bool:
        text = lesson.lower()

        positive = (
            "synonymous",
            "valid candidate",
            "valid substitute",
            "interchangeable",
            "equivalent",
            "do not reject",
            "do not dismiss",
        )

        negative = (
            "not interchangeable",
            "not equivalent",
            "not synonymous",
            "do not substitute",
            "not a valid substitute",
        )

        return (
            any(term in text for term in positive)
            and not any(term in text for term in negative)
        )

    @staticmethod
    def check(
        lesson: str,
        task: str,
        trajectory: Trajectory,
    ) -> list[EvidenceConflict]:
        if not EvidenceConsistencyChecker._is_equivalence_claim(lesson):
            return []

        tracker = TaskStateTracker(task)

        if tracker.operation != "clean":
            return []

        quoted_types = set(
            re.findall(
                r'"([a-z][a-z0-9]*)"',
                lesson.lower(),
            )
        )

        if tracker.target not in quoted_types:
            return []

        cleaned = set()
        unsuccessful_moves = []
        conflicts = []

        for step in trajectory.steps:
            if not step.valid_format or not step.admissible:
                continue

            action = step.action.lower().strip()

            clean_match = re.fullmatch(
                r"clean ([a-z][a-z0-9]* \d+) with "
                r"[a-z][a-z0-9]* \d+",
                action,
            )

            if clean_match:
                obj = clean_match.group(1)

                confirmation = (
                    rf"\byou clean (?:the )?"
                    rf"{re.escape(obj)} using\b"
                )

                if re.search(
                    confirmation,
                    step.next_observation,
                    re.I,
                ):
                    cleaned.add(obj)

                continue

            move_match = re.fullmatch(
                r"move ([a-z][a-z0-9]* \d+) to "
                r"([a-z][a-z0-9]* \d+)",
                action,
            )

            if move_match is None:
                continue

            obj, destination = move_match.groups()
            obj_type = obj.rsplit(" ", 1)[0]

            confirmation = (
                rf"\byou move (?:the )?"
                rf"{re.escape(obj)} to (?:the )?"
                rf"{re.escape(destination)}\b"
            )

            if not re.search(
                confirmation,
                step.next_observation,
                re.I,
            ):
                continue

            if obj not in cleaned:
                continue

            if step.won and obj_type == tracker.target:
                for previous_obj, previous_dest, previous_action in unsuccessful_moves:
                    previous_type = previous_obj.rsplit(" ", 1)[0]

                    if previous_dest != destination:
                        continue

                    if previous_type == tracker.target:
                        continue

                    if previous_type not in quoted_types:
                        continue

                    conflicts.append(
                        EvidenceConflict(
                            source_action=previous_action,
                            winning_action=action,
                            reason=(
                                "A cleaned substitute object did not "
                                "complete the task, while a cleaned "
                                "object of the required type did."
                            ),
                        )
                    )

            elif not step.won and obj_type in quoted_types:
                unsuccessful_moves.append(
                    (obj, destination, action)
                )

        return conflicts



