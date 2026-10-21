"""在真实 ALFWorld 环境中测试 Reflexion。"""

import argparse
import json
from pathlib import Path

from src.agents.reflexion_agent import ReflexionAgent
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamefile", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()

    llm = LLMClient()

    agent = ReflexionAgent(
        llm=llm,
        max_steps=args.max_steps,
        max_attempts=args.max_attempts,
    )

    # 环境步数上限与每次尝试的决策上限保持一致
    env = ALFWorldEnvironment(
        gamefile=args.gamefile,
        max_steps=args.max_steps,
    )

    try:
        result = agent.run(env)
    finally:
        env.close()

    print(f"\nTask: {result.task}")
    print(f"Success: {result.success}")
    print(f"Attempts: {len(result.attempts)}")
    print(f"Total steps: {result.total_steps}")

    for i, trajectory in enumerate(result.attempts, start=1):
        print(f"\nAttempt {i}:")
        print(f"  Success: {trajectory.success}")
        print(f"  Steps: {len(trajectory.steps)}")
        print(f"  Truncated: {trajectory.truncated}")

        if i <= len(result.reflections):
            print(f"  Reflection: {result.reflections[i - 1]}")

    # 保存完整尝试轨迹，便于后续分析
    output = Path("outputs/reflexion_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "task": result.task,
        "gamefile": result.gamefile,
        "success": result.success,
        "total_steps": result.total_steps,
        "reflections": result.reflections,
        "attempts": [
            {
                "success": trajectory.success,
                "truncated": trajectory.truncated,
                "steps": [
                    {
                        "thought": step.thought,
                        "action": step.action,
                        "admissible": step.admissible,
                        "observation": step.observation,
                        "next_observation": step.next_observation,
                        "won": step.won,
                    }
                    for step in trajectory.steps
                ],
            }
            for trajectory in result.attempts
        ],
    }

    output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
