"""评估Experience-Augmented ReAct"""

import json
import os
from pathlib import Path

from src.agents.react_agent import ReActAgent
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient
from src.data.manifest import load_manifest
from src.core.trajectory import Trajectory
from src.memory.store import ExperienceStore

MAX_STEPS = 50
MANIFEST_PATH = Path("outputs/manifests/valid_seen_seed42_140.json")
MEMORY_PATH = Path("outputs/memory/ordinary_experience_store_qwen3.8-chat_train_seed42_1000.json")
EXPERIENCE_TOP_K = 3

MODEL = os.getenv("OPENAI_MODEL")
MODEL_DIR_NAME = MODEL.replace("/", "_")

OUTPUT_DIR = Path(f"outputs/ordinary_memory_{MODEL_DIR_NAME}_valid_seen_seed42_140")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    llm = LLMClient()

    experience_store = ExperienceStore()
    experience_store.load(MEMORY_PATH)
    
    # 从固定manifest加载实验任务
    games = load_manifest(MANIFEST_PATH)
    num_episodes = len(games)

    successes = 0
    truncated_episodes = 0

    total_decision_steps = 0
    total_environment_steps = 0

    total_format_errors = 0
    total_inadmissible_actions = 0
    total_valid_actions = 0

    results = []

    for episode_index, game in enumerate(games):
        print()
        print("=" * 60)
        print(f"Episode {episode_index + 1}/{num_episodes}")
        print("=" * 60)

        output_path = OUTPUT_DIR / f"episode_{episode_index:03d}.json"

        # 已完成的episode直接加载，避免重复调用LLM
        if output_path.exists():
            print(f"Resume: loading existing trajectory: {output_path}")
            trajectory = Trajectory.load_json(str(output_path))

        else:
            env = ALFWorldEnvironment(gamefile=str(game.game_file), max_steps=MAX_STEPS)
            agent = ReActAgent(
                llm=llm, 
                max_steps=MAX_STEPS,
                experience_store=experience_store,
                experience_top_k=EXPERIENCE_TOP_K,
            )

            try:
                trajectory = agent.run(env)

            finally:
                env.close()
        
            # 保存trajectory
            trajectory.save_json(str(output_path))
        

        decision_steps = len(trajectory) # Agent总决策数
        environment_steps = sum(step.valid_format for step in trajectory.steps) # 格式正确，环境动作数量
        format_errors = sum(not step.valid_format for step in trajectory.steps) 

        inadmissible_actions = sum(step.valid_format and not step.admissible for step in trajectory.steps)
        valid_actions = sum(step.valid_format and step.admissible for step in trajectory.steps)
        invalid_actions = format_errors + inadmissible_actions # 总无效决策
        
        # 检查指标之间是否自洽。
        assert decision_steps == format_errors + environment_steps
        assert environment_steps == inadmissible_actions + valid_actions

        successes += int(trajectory.success)
        truncated_episodes += int(trajectory.truncated)

        total_decision_steps += decision_steps
        total_environment_steps += environment_steps

        total_format_errors += format_errors
        total_inadmissible_actions += inadmissible_actions
        total_valid_actions += valid_actions

        result = {
            "episode": episode_index,
            "game_id": game.id,
            "split": game.split,
            "task_type": game.task_type,
            "task": trajectory.task,
            "gamefile": trajectory.gamefile,

            "success": trajectory.success,
            "truncated": trajectory.truncated,

            "decision_steps": decision_steps,
            "environment_steps": environment_steps,

            "format_errors": format_errors,
            "inadmissible_actions": inadmissible_actions,
            "invalid_actions": invalid_actions,
            "valid_actions": valid_actions,
        }

        results.append(result)

        print(f"Task: {trajectory.task}")
        print(f"Task type: {game.task_type}")
        print(f"Success: {trajectory.success}")
        print(f"Truncated: {trajectory.truncated}")
        print(f"Decision steps: {decision_steps}")
        print(f"Environment steps: {environment_steps}")
        print(f"Format errors: {format_errors}")
        print(f"Inadmissible actions: {inadmissible_actions}")    

    task_type_stats = {}
    for result in results:
        task_type = result["task_type"]
        if task_type not in task_type_stats:
            task_type_stats[task_type] = {
                "episodes": 0,
                "successes": 0,
            }
        stats = task_type_stats[task_type]
        stats["episodes"] += 1
        stats["successes"] += int(result["success"])

    for stats in task_type_stats.values():
        stats["success_rate"] = stats["successes"] / stats["episodes"]

    # Summary
    success_rate = successes / num_episodes
    truncated_rate = truncated_episodes / num_episodes
    average_decision_steps = total_decision_steps / num_episodes
    average_environment_steps = total_environment_steps / num_episodes
        
    if total_decision_steps:
        format_valid_rate = (total_decision_steps - total_format_errors) / total_decision_steps
        valid_action_rate = total_valid_actions / total_decision_steps
    else:
        format_valid_rate = 0.0
        valid_action_rate = 0.0

    if total_environment_steps:
        admissible_action_rate = total_valid_actions / total_environment_steps
    else:
        admissible_action_rate = 0.0

    summary = {
        "model": MODEL,
        "manifest": str(MANIFEST_PATH),
        "max_steps": MAX_STEPS,

        "num_episodes": num_episodes,
        "successes": successes,
        "success_rate": success_rate,

        "truncated_episodes": truncated_episodes,
        "truncated_rate": truncated_rate,

        "average_decision_steps": average_decision_steps,
        "average_environment_steps": average_environment_steps,

        "total_decision_steps": total_decision_steps,
        "total_environment_steps": total_environment_steps,

        "format_errors": total_format_errors,
        "inadmissible_actions": total_inadmissible_actions,

        "format_valid_rate": format_valid_rate,
        "admissible_action_rate": admissible_action_rate,
        "valid_action_rate": valid_action_rate,

        "episodes": results,
        "task_type_stats": task_type_stats,
    }
    summary_path = OUTPUT_DIR / "summary.json"

    with summary_path.open("w", encoding="utf-8",) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("Baseline Summary")
    print("=" * 60)
    
    print(f"Model: {MODEL}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Episodes: {num_episodes}")

    print(
        f"Successes: "
        f"{successes}/{num_episodes}"
    )
    print(
        f"Success rate: "
        f"{success_rate:.2%}"
    )

    print(
        f"Truncated rate: "
        f"{truncated_rate:.2%}"
    )

    print(
        f"Average decision steps: "
        f"{average_decision_steps:.2f}"
    )
    print(
        f"Average environment steps: "
        f"{average_environment_steps:.2f}"
    )

    print(
        f"Format errors: "
        f"{total_format_errors}"
    )
    print(
        f"Inadmissible actions: "
        f"{total_inadmissible_actions}"
    )

    print(
        f"Format valid rate: "
        f"{format_valid_rate:.2%}"
    )
    print(
        f"Admissible action rate: "
        f"{admissible_action_rate:.2%}"
    )
    print(
        f"Valid action rate: "
        f"{valid_action_rate:.2%}"
    )

    print()
    print("Task type results:")

    for task_type, stats in sorted(
        task_type_stats.items()
        ):
        print(
            f"  {task_type}: "
            f"{stats['successes']}/"
            f"{stats['episodes']} "
            f"({stats['success_rate']:.2%})"
        )

if __name__ == "__main__":
    main()






