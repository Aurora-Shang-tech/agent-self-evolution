import re

from src.memory.state_tracker import TaskStateTracker


class ConditionalSearchSkill:
    def __init__(self, proposal: dict, task: str):
        if proposal["decision"] != "CANDIDATE":
            raise ValueError("Skill is not a candidate")

        trigger = proposal["trigger"]

        if (
            trigger["require_target_not_found"] is not True
            or trigger["require_unvisited_location"] is not True
        ):
            raise ValueError("Unsupported trigger")

        self.min_locations = trigger[
            "min_unproductive_search_locations"
        ]

        if (
            type(self.min_locations) is not int
            or not 1 <= self.min_locations <= 8
        ):
            raise ValueError("Invalid trigger threshold")

        self.locations = proposal["preferred_location_types"]

        if self.locations != ["countertop"]:
            raise ValueError(
                "Only the evaluated countertop candidate is supported"
            )

        if proposal["action_template"] != (
            "go to <selected unvisited countertop>"
        ):
            raise ValueError("Unsupported skill template")

        self.target = TaskStateTracker(task).target
        self.visited = set()
        self.unproductive = set()
        self.target_found = False
        self.used = False

    def _has_target(self, text: str) -> bool:
        if not self.target:
            return False

        return bool(
            re.search(
                rf"(?<![a-z0-9]){re.escape(self.target)}"
                r"\s+\d+\b",
                text.lower(),
            )
        )

    def observe(self, step):
        if not step.valid_format or not step.admissible:
            return

        action = " ".join(step.action.lower().split())
        observation = step.next_observation.lower()

        match = re.fullmatch(
            r"(?:go to|open|examine) ([a-z][a-z0-9]* \d+)",
            action,
        )

        location = match.group(1) if match else None

        if action.startswith("go to ") and location:
            self.visited.add(location)

        if self._has_target(observation):
            self.target_found = True

        if self.target and re.fullmatch(
            rf"take {re.escape(self.target)} \d+ from .+",
            action,
        ):
            self.target_found = True

        if (
            location
            and "you see" in observation
            and not self.target_found
        ):
            self.unproductive.add(location)

    def suggest(self, admissible_actions):
        if (
            not self.target
            or self.target_found
            or self.used
            or len(self.unproductive) < self.min_locations
        ):
            return None

        actions = [
            " ".join(a.lower().split())
            for a in admissible_actions
        ]

        if any(
            re.fullmatch(
                rf"take {re.escape(self.target)} \d+ from .+",
                action,
            )
            for action in actions
        ):
            return None

        for kind in self.locations:
            for action in sorted(actions):
                if not re.fullmatch(
                    rf"go to {re.escape(kind)} \d+",
                    action,
                ):
                    continue

                location = action.removeprefix("go to ")

                if location not in self.visited:
                    return action

        return None

    def consume(self):
        self.used = True
