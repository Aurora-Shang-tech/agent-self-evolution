
"""Counterfactual ALFWorld verification with original memory/state context."""

from dataclasses import dataclass

from src.core.protocol import (
    environment_feedback,
    trajectory_prefix_messages,
)
from src.core.trajectory import Trajectory, TrajectoryStep
from src.counterfactual.replay import replay_to_step
from src.memory.state_tracker import TaskStateTracker


@dataclass(frozen=True)
class CounterfactualResult:
    critical_step: int
    original_action: str
    counterfactual_action: str
    success: bool
    continuation_steps: int
    trajectory: Trajectory | None = None


@dataclass(frozen=True)
class CounterfactualBatchResult:
    critical_step: int
    original_action: str
    results: tuple[CounterfactualResult, ...]


def verify_counterfactual(
    trajectory: Trajectory,
    critical_step: int,
    counterfactual_action: str,
    agent,
    *,
    max_steps: int = 50,
) -> CounterfactualResult:
    if not 0 <= critical_step < len(trajectory.steps):
        raise IndexError(
            f"critical_step out of range: {critical_step}"
        )

    original_step = trajectory.steps[critical_step]

    env, replayed = replay_to_step(
        trajectory=trajectory,
        step_index=critical_step,
        max_steps=max_steps,
    )

    try:
        if counterfactual_action not in replayed.admissible_actions:
            raise ValueError(
                "Counterfactual action is not admissible: "
                f"{counterfactual_action}"
            )

        prefix_length = 2 + 2 * critical_step

        if trajectory.messages:
            if len(trajectory.messages) < prefix_length:
                raise RuntimeError(
                    "Stored LLM context is incomplete"
                )

            messages = [
                dict(row)
                for row in trajectory.messages[:prefix_length]
            ]

        elif (
            getattr(agent, "use_state_tracking", False)
            or getattr(agent, "experience_store", None) is not None
        ):
            raise RuntimeError(
                "Cannot restore memory/state context "
                "from legacy trajectory"
            )

        else:
            messages = trajectory_prefix_messages(
                trajectory,
                critical_step,
            )

        tracker = None

        if agent.use_state_tracking:
            tracker = TaskStateTracker(trajectory.task)

            for step in trajectory.steps[:critical_step]:
                tracker.observe(step)

        messages.append({
            "role": "assistant",
            "content": (
                "Thought: Try an alternative action "
                "from the critical state.\n"
                f"Action: {counterfactual_action}"
            ),
        })

        cf = env.step(counterfactual_action)

        if cf.won or cf.done:
            return CounterfactualResult(
                critical_step,
                original_step.action,
                counterfactual_action,
                bool(cf.won),
                0,
                None,
            )

        if tracker is not None:
            tracker.observe(
                TrajectoryStep(
                    observation=replayed.observation,
                    admissible_actions=list(
                        replayed.admissible_actions
                    ),
                    thought=(
                        "Try an alternative action "
                        "from the critical state."
                    ),
                    action=counterfactual_action,
                    valid_format=True,
                    admissible=cf.admissible,
                    next_observation=cf.observation,
                    score=cf.score,
                    done=cf.done,
                    won=cf.won,
                )
            )

        feedback = environment_feedback(
            step=critical_step + 1,
            observation=cf.observation,
            admissible_actions=cf.admissible_actions,
            valid_format=True,
            admissible=cf.admissible,
            done=cf.done,
            won=cf.won,
        )

        feedback += agent._state_reminder(tracker)

        messages.append({
            "role": "user",
            "content": feedback,
        })

        remaining_steps = max_steps - (critical_step + 1)

        if remaining_steps <= 0:
            return CounterfactualResult(
                critical_step,
                original_step.action,
                counterfactual_action,
                False,
                0,
                None,
            )

        continuation = agent.continue_from(
            env,
            task=trajectory.task,
            gamefile=trajectory.gamefile,
            observation=cf.observation,
            admissible_actions=cf.admissible_actions,
            messages=messages,
            max_decisions=remaining_steps,
            tracker=tracker,
            step_offset=critical_step + 1,
        )

        return CounterfactualResult(
            critical_step,
            original_step.action,
            counterfactual_action,
            continuation.success,
            len(continuation),
            continuation,
        )

    finally:
        env.close()


def verify_counterfactual_candidates(
    trajectory: Trajectory,
    critical_step: int,
    counterfactual_actions: tuple[str, ...],
    agent,
    *,
    max_steps: int = 50,
) -> CounterfactualBatchResult:
    if not 0 <= critical_step < len(trajectory.steps):
        raise IndexError(
            f"Invalid critical_step: {critical_step}"
        )

    results = tuple(
        verify_counterfactual(
            trajectory,
            critical_step,
            action,
            agent,
            max_steps=max_steps,
        )
        for action in counterfactual_actions
    )

    return CounterfactualBatchResult(
        critical_step,
        trajectory.steps[critical_step].action,
        results,
    )

