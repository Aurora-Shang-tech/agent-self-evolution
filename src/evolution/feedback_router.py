from dataclasses import asdict, dataclass
from statistics import median

from src.core.trajectory import Trajectory


@dataclass(frozen=True)
class FeedbackDecision:
    route: str
    reason: str
    task: str
    task_type: str
    success: bool
    environment_actions: int
    reference_median: float | None
    reference_samples: int

    def to_dict(self):
        return asdict(self)


class FeedbackRouter:
    def __init__(
        self,
        reference_actions_by_type: dict[str, list[int]],
        *,
        min_reference_samples: int = 5,
        min_extra_actions: int = 5,
        min_action_ratio: float = 1.5,
    ):
        if min_reference_samples < 1:
            raise ValueError("Invalid minimum sample count")

        if min_extra_actions < 0 or min_action_ratio < 1:
            raise ValueError("Invalid search screening thresholds")

        self.min_reference_samples = min_reference_samples
        self.min_extra_actions = min_extra_actions
        self.min_action_ratio = min_action_ratio

        self.references = {}

        for task_type, actions in reference_actions_by_type.items():
            if not isinstance(task_type, str) or not task_type:
                raise ValueError("Invalid task type")

            if any(
                type(n) is not int or n <= 0
                for n in actions
            ):
                raise ValueError("Invalid reference action count")

            self.references[task_type] = {
                "samples": len(actions),
                "median": (
                    float(median(actions))
                    if actions else None
                ),
            }

    def route(
        self,
        trajectory: Trajectory,
        task_type: str,
        *,
        phase: str = "learning",
    ) -> FeedbackDecision:
        if phase not in ("learning", "validation", "holdout"):
            raise ValueError("Unknown experiment phase")

        env_actions = sum(
            bool(step.valid_format)
            for step in trajectory.steps
        )

        reference = self.references.get(
            task_type,
            {"samples": 0, "median": None},
        )

        n = reference["samples"]
        reference_median = reference["median"]

        def decision(route, reason):
            return FeedbackDecision(
                route=route,
                reason=reason,
                task=trajectory.task,
                task_type=task_type,
                success=bool(trajectory.success),
                environment_actions=env_actions,
                reference_median=reference_median,
                reference_samples=n,
            )

        if phase != "learning":
            return decision(
                "IGNORE",
                "Evaluation data must not enter learning",
            )

        if not trajectory.success:
            return decision(
                "MEMORY_REVIEW",
                "Unsuccessful episode requires failure analysis",
            )

        if (
            reference_median is None
            or n < self.min_reference_samples
        ):
            return decision(
                "ARCHIVE",
                "Insufficient successful reference episodes",
            )

        extra_actions = env_actions - reference_median

        if (
            extra_actions >= self.min_extra_actions
            and env_actions >= (
                reference_median * self.min_action_ratio
            )
        ):
            return decision(
                "SKILL_REVIEW",
                "Successful episode with unusually high action cost",
            )

        return decision(
            "ARCHIVE",
            "Successful episode without strong inefficiency signal",
        )
