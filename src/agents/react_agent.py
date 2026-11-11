
"""ReAct Agent for ALFWorld with replayable execution context."""

from src.core.protocol import (
    environment_feedback,
    initial_messages,
    parse_assistant,
)
from src.core.trajectory import Trajectory, TrajectoryStep
from src.envs.environment import ALFWorldEnvironment
from src.memory.state_tracker import TaskStateTracker


class ReActAgent:
    def __init__(
        self,
        llm,
        max_steps: int = 50,
        experience_store=None,
        experience_top_k: int = 3,
        use_state_tracking: bool = False,
    ):
        self.llm = llm
        self.max_steps = max_steps
        self.experience_store = experience_store
        self.experience_top_k = experience_top_k
        self.use_state_tracking = use_state_tracking

    def run(self, env: ALFWorldEnvironment) -> Trajectory:
        state = env.reset()
        trajectory = Trajectory(
            task=state.task,
            gamefile=state.gamefile,
        )

        experiences = []

        if self.experience_store is not None:
            experiences = [
                item.lesson
                for item in self.experience_store.retrieve(
                    task=state.task,
                    top_k=self.experience_top_k,
                )
            ]

        tracker = (
            TaskStateTracker(state.task)
            if self.use_state_tracking
            else None
        )

        messages = initial_messages(
            task=state.task,
            observation=state.observation,
            admissible_actions=state.admissible_actions,
            experiences=experiences,
        )

        reminder = self._state_reminder(tracker)

        if reminder:
            messages[-1]["content"] += reminder

        return self._run_loop(
            env,
            trajectory=trajectory,
            messages=messages,
            observation=state.observation,
            admissible_actions=state.admissible_actions,
            max_decisions=self.max_steps,
            tracker=tracker,
        )

    def continue_from(
        self,
        env: ALFWorldEnvironment,
        *,
        task: str,
        gamefile: str,
        observation: str,
        admissible_actions: tuple[str, ...],
        messages: list[dict],
        max_decisions: int,
        tracker: TaskStateTracker | None = None,
        step_offset: int = 0,
    ) -> Trajectory:
        if self.use_state_tracking and tracker is None:
            raise ValueError(
                "State-aware continuation requires "
                "restored TaskStateTracker"
            )

        trajectory = Trajectory(
            task=task,
            gamefile=gamefile,
        )

        return self._run_loop(
            env,
            trajectory=trajectory,
            messages=[dict(message) for message in messages],
            observation=observation,
            admissible_actions=admissible_actions,
            max_decisions=max_decisions,
            tracker=tracker,
            step_offset=step_offset,
        )

    def _run_loop(
        self,
        env: ALFWorldEnvironment,
        *,
        trajectory: Trajectory,
        messages: list[dict],
        observation: str,
        admissible_actions: tuple[str, ...],
        max_decisions: int,
        tracker: TaskStateTracker | None = None,
        step_offset: int = 0,
    ) -> Trajectory:
        for local_step in range(max_decisions):
            response = self.llm.chat(messages)
            parsed = parse_assistant(response)

            messages.append({
                "role": "assistant",
                "content": response,
            })

            absolute_step = step_offset + local_step + 1

            if not parsed.valid_format:
                trajectory.add_step(
                    TrajectoryStep(
                        observation=observation,
                        admissible_actions=list(admissible_actions),
                        thought=parsed.reasoning,
                        action=parsed.action or "",
                        valid_format=False,
                        admissible=False,
                        next_observation=observation,
                        score=0.0,
                        done=False,
                        won=False,
                    )
                )

                feedback = environment_feedback(
                    step=absolute_step,
                    observation=observation,
                    admissible_actions=admissible_actions,
                    valid_format=False,
                    admissible=False,
                    done=False,
                    won=False,
                )

                messages.append({
                    "role": "user",
                    "content": feedback,
                })
                continue

            assert parsed.action is not None

            env_step = env.step(parsed.action)

            step = TrajectoryStep(
                observation=observation,
                admissible_actions=list(admissible_actions),
                thought=parsed.reasoning,
                action=parsed.action,
                valid_format=True,
                admissible=env_step.admissible,
                next_observation=env_step.observation,
                score=env_step.score,
                done=env_step.done,
                won=env_step.won,
            )

            trajectory.add_step(step)

            if tracker is not None:
                tracker.observe(step)

            if env_step.won:
                trajectory.success = True
                break

            if env_step.done:
                break

            observation = env_step.observation
            admissible_actions = env_step.admissible_actions

            feedback = environment_feedback(
                step=absolute_step,
                observation=observation,
                admissible_actions=admissible_actions,
                valid_format=True,
                admissible=env_step.admissible,
                done=env_step.done,
                won=env_step.won,
            )

            feedback += self._state_reminder(tracker)

            messages.append({
                "role": "user",
                "content": feedback,
            })

        else:
            trajectory.truncated = True

        trajectory.messages = [
            dict(message)
            for message in messages
        ]

        return trajectory

    @staticmethod
    def _state_reminder(
        tracker: TaskStateTracker | None,
    ) -> str:
        if tracker is None or tracker.operation is None:
            return ""

        confirmed = sorted(
            object_id
            for object_id, states in tracker.object_states.items()
            if tracker.operation in states
        )

        confirmed_text = (
            ", ".join(confirmed)
            if confirmed
            else "none confirmed by environment"
        )

        return (
            "\n\nTask-state evidence (latest state):\n"
            f"Required operation: {tracker.operation}\n"
            f"Target object type: {tracker.target}\n"
            f"Confirmed instances: {confirmed_text}\n"
            "Do not infer the required state from the storage location "
            "or from a successful move action.\n"
            "Before final placement, ensure the same object instance "
            "has the required confirmed state.\n"
            "If this state is not confirmed, plan how to perform the "
            "required operation using available actions.\n"
            "A later heating or cooling action can invalidate "
            "an earlier temperature state."
        )

