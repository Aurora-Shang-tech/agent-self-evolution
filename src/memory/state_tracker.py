
"""具体物体的状态追踪"""

import re
from dataclasses import dataclass, field

from src.core.trajectory import TrajectoryStep


@dataclass
class TaskStateTracker:
    task: str
    operation: str | None = field(init=False, default=None)
    target: str | None = field(init=False, default=None)
    action_observed: bool = False
    object_states: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self):
        task = self.task.lower().strip()

        match = re.search(
            r"\b(clean|cool|heat)\s+(?:a|an|some|the)\s+([a-z][a-z0-9]*)\b",
            task,
        )

        if match:
            self.operation, self.target = match.groups()
            return

        match = re.search(
            r"\bput\s+(?:a|an|some|the)\s+"
            r"(clean|cool|cold|hot|heated)\s+([a-z][a-z0-9]*)\b",
            task,
        )

        if match:
            attribute, self.target = match.groups()
            self.operation = {
                "cold": "cool",
                "hot": "heat",
                "heated": "heat",
            }.get(attribute, attribute)

    def observe(self, step: TrajectoryStep) -> None:
        if self.operation is None or self.target is None:
            return

        if not step.valid_format or not step.admissible:
            return

        action = " ".join(step.action.lower().split())

        match = re.fullmatch(
            r"(clean|cool|heat) ([a-z][a-z0-9]*) (\d+) "
            r"with [a-z][a-z0-9]* \d+",
            action,
        )

        if match is None:
            return

        operation, object_type, object_id = match.groups()

        if object_type != self.target:
            return

        if operation == self.operation:
            self.action_observed = True

        verb = {
            "clean": r"clean(?:ed)?",
            "cool": r"cool(?:ed)?",
            "heat": r"heat(?:ed)?",
        }[operation]

        confirmation = (
            rf"\byou\s+{verb}\s+(?:the\s+)?"
            rf"{re.escape(object_type)}\s+{object_id}\s+using\b"
        )

        if not re.search(confirmation, step.next_observation, re.I):
            return

        key = f"{object_type} {object_id}"
        states = self.object_states.setdefault(key, set())

        if operation == "clean":
            states.add("clean")

        elif operation == "cool":
            states.discard("heat")
            states.add("cool")

        elif operation == "heat":
            states.discard("cool")
            states.add("heat")

    def is_confirmed(self, object_id: str) -> bool:
        return self.operation in self.object_states.get(object_id, set())

    @property
    def confirmed(self) -> bool:
        if self.operation is None:
            return False

        return any(
            self.operation in states
            for states in self.object_states.values()
        )

    @property
    def requirement_met(self) -> bool:
        return self.operation is None or self.confirmed

