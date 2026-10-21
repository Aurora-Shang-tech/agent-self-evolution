"""测试新版ReAct Agent是否能跑通一个ALFWorld episode"""

from pathlib import Path

from src.agents.react_agent import ReActAgent
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient

GAMEFILE = Path("/home/sxw/.cache/alfworld/json_2.1.1/train/pick_clean_then_place_in_recep-Bowl-None-Cabinet-13/trial_T20190909_022941_058533/game.tw-pddl")

MAX_STEPS = 50

def main() -> None:
    if not GAMEFILE.exists():
        raise FileNotFoundError("Gamefile does not exist")

    llm = LLMClient()

    env = ALFWorldEnvironment(
        gamefile=str(GAMEFILE),
        max_steps=MAX_STEPS,
    )

    agent = ReActAgent(
        llm=llm,
        max_steps=MAX_STEPS,
    )

    try:
        trajectory = agent.run(env)

    finally:
        env.close()

    print()
    print("=" * 60)
    print("ReAct V2 Test")
    print("=" * 60)

    print(f"Task: {trajectory.task}")
    print(f"Gamefile: {trajectory.gamefile}")
    print(f"Success: {trajectory.success}")
    print(f"Environment steps: {len(trajectory)}")

    invalid_actions = sum(
        not step.admissible
        for step in trajectory.steps
    )

    print(f"Invalid actions: {invalid_actions}")
    
    # 打印完整轨迹
    for i, step in enumerate(
        trajectory.steps,
        start=1,
    ):
        print()
        print("-" * 60)
        print(f"Step {i}")
        print("-" * 60)

        print("Observation:")
        print(step.observation)

        print()
        print("Thought:")
        print(step.thought)

        print()
        print("Action:")
        print(step.action)

        print()
        print(
            f"Admissible: {step.admissible}"
        )

        print()
        print("Next observation:")
        print(step.next_observation)

        print()
        print(
            f"Done: {step.done} | "
            f"Won: {step.won} | "
            f"Score: {step.score}"
        )

    output_path = (
        "outputs/react_v2/episode_000.json"
    )

    trajectory.save_json(output_path)

    print()
    print("=" * 60)
    print(
        f"Trajectory saved to: {output_path}"
    )


if __name__ == "__main__":
    main()


