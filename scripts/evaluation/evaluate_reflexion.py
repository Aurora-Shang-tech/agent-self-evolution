
"""评估 Reflexion Agent，支持 seen/unseen 和断点续跑。"""

import argparse
import json
import os
from pathlib import Path

from src.agents.reflexion_agent import ReflexionAgent
from src.data.manifest import load_manifest
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient


SPLIT_SIZES = {
    "valid_seen": 140,
    "valid_unseen": 134,
}


def save_json(path: Path, data: dict) -> None:
    """保存 JSON 结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def trajectory_to_dict(trajectory) -> dict:
    """序列化单次尝试的完整轨迹。"""
    return {
        "task": trajectory.task,
        "gamefile": trajectory.gamefile,
        "success": trajectory.success,
        "truncated": trajectory.truncated,
        "steps": [
            {
                "observation": step.observation,
                "admissible_actions": step.admissible_actions,
                "thought": step.thought,
                "action": step.action,
                "valid_format": step.valid_format,
                "admissible": step.admissible,
                "next_observation": step.next_observation,
                "score": step.score,
                "done": step.done,
                "won": step.won,
            }
            for step in trajectory.steps
        ],
    }


def attempt_stats(trajectory) -> dict:
    """计算单次尝试的指标。"""
    decision_steps = len(trajectory.steps)

    environment_steps = sum(
        step.valid_format for step in trajectory.steps
    )
    format_errors = sum(
        not step.valid_format for step in trajectory.steps
    )
    inadmissible_actions = sum(
        step.valid_format and not step.admissible
        for step in trajectory.steps
    )
    valid_actions = sum(
        step.valid_format and step.admissible
        for step in trajectory.steps
    )

    assert decision_steps == format_errors + environment_steps
    assert environment_steps == (
        inadmissible_actions + valid_actions
    )

    return {
        "success": trajectory.success,
        "truncated": trajectory.truncated,
        "decision_steps": decision_steps,
        "environment_steps": environment_steps,
        "format_errors": format_errors,
        "inadmissible_actions": inadmissible_actions,
        "valid_actions": valid_actions,
    }


def build_episode_record(game, episode_index, result) -> dict:
    """将 Reflexion 多次尝试合并成一个任务结果。"""
    stats = [attempt_stats(t) for t in result.attempts]

    return {
        "episode": episode_index,
        "game_id": game.id,
        "split": game.split,
        "task_type": game.task_type,
        "task": result.task,
        "gamefile": result.gamefile,
        "success": result.success,
        "attempt_count": len(result.attempts),
        "reflection_count": len(result.reflections),
        "reflections": result.reflections,
        "decision_steps": sum(s["decision_steps"] for s in stats),
        "environment_steps": sum(
            s["environment_steps"] for s in stats
        ),
        "format_errors": sum(s["format_errors"] for s in stats),
        "inadmissible_actions": sum(
            s["inadmissible_actions"] for s in stats
        ),
        "valid_actions": sum(s["valid_actions"] for s in stats),
        "truncated_attempts": sum(
            int(s["truncated"]) for s in stats
        ),
        "attempt_stats": stats,
        "attempts": [
            trajectory_to_dict(t) for t in result.attempts
        ],
    }


def build_summary(records, *, model, manifest, max_steps,
                  max_attempts) -> dict:
    """汇总成功率、总步数及任务类型表现。"""
    n = len(records)
    successes = sum(int(r["success"]) for r in records)

    task_type_stats = {}
    for r in records:
        task_type = r["task_type"]
        stats = task_type_stats.setdefault(
            task_type, {"episodes": 0, "successes": 0}
        )
        stats["episodes"] += 1
        stats["successes"] += int(r["success"])

    for stats in task_type_stats.values():
        stats["success_rate"] = (
            stats["successes"] / stats["episodes"]
        )

    totals = {
        key: sum(r[key] for r in records)
        for key in (
            "decision_steps",
            "environment_steps",
            "format_errors",
            "inadmissible_actions",
            "valid_actions",
            "attempt_count",
            "reflection_count",
            "truncated_attempts",
        )
    }

    return {
        "model": model,
        "manifest": str(manifest),
        "max_steps_per_attempt": max_steps,
        "max_attempts": max_attempts,
        "num_episodes": n,
        "successes": successes,
        "success_rate": successes / n if n else 0.0,
        "average_decision_steps": (
            totals["decision_steps"] / n if n else 0.0
        ),
        "average_environment_steps": (
            totals["environment_steps"] / n if n else 0.0
        ),
        "average_attempts": (
            totals["attempt_count"] / n if n else 0.0
        ),
        "total_decision_steps": totals["decision_steps"],
        "total_environment_steps": totals["environment_steps"],
        "total_attempts": totals["attempt_count"],
        "total_reflections": totals["reflection_count"],
        "format_errors": totals["format_errors"],
        "inadmissible_actions": totals["inadmissible_actions"],
        "valid_actions": totals["valid_actions"],
        "truncated_attempts": totals["truncated_attempts"],
        "task_type_stats": task_type_stats,
        "episodes": [
            {k: v for k, v in r.items() if k != "attempts"}
            for r in records
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=list(SPLIT_SIZES),
        required=True,
    )
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()

    if args.max_steps < 1 or args.max_attempts < 1:
        parser.error("max-steps and max-attempts must be >= 1")

    model = os.getenv("OPENAI_MODEL")
    if not model:
        parser.error("OPENAI_MODEL is not set")

    size = SPLIT_SIZES[args.split]
    manifest = Path(
        f"outputs/manifests/{args.split}_seed42_{size}.json"
    )
    output_dir = Path(
        f"outputs/reflexion_{model.replace('/', '_')}_"
        f"{args.split}_seed42_{size}_"
        f"steps{args.max_steps}_attempts{args.max_attempts}"
    )

    games = load_manifest(manifest)
    llm = LLMClient()
    records = []

    for index, game in enumerate(games):
        output_path = output_dir / f"episode_{index:03d}.json"

        if output_path.exists():
            print(f"[{index + 1}/{len(games)}] Resume: {output_path}")
            record = json.loads(
                output_path.read_text(encoding="utf-8")
            )

            # 避免错误复用不同任务的结果
            if (
                record["episode"] != index
                or record["game_id"] != game.id
                or record["gamefile"] != str(game.game_file)
                or record["split"] != game.split
            ):
                raise RuntimeError(f"Episode mismatch: {output_path}")

            if not 1 <= record["attempt_count"] <= args.max_attempts:
                raise RuntimeError(f"Invalid attempt count: {output_path}")
        else:
            env = ALFWorldEnvironment(
                gamefile=str(game.game_file),
                max_steps=args.max_steps,
            )
            agent = ReflexionAgent(
                llm=llm,
                max_steps=args.max_steps,
                max_attempts=args.max_attempts,
            )

            try:
                result = agent.run(env)
            finally:
                env.close()

            record = build_episode_record(game, index, result)
            save_json(output_path, record)

        records.append(record)

        print(
            f"[{index + 1}/{len(games)}] "
            f"success={record['success']} "
            f"attempts={record['attempt_count']} "
            f"steps={record['decision_steps']}"
        )

        # 每个 episode 完成后更新汇总，支持中断后恢复
        summary = build_summary(
            records,
            model=model,
            manifest=manifest,
            max_steps=args.max_steps,
            max_attempts=args.max_attempts,
        )
        save_json(output_dir / "summary.json", summary)

    print("\n" + "=" * 60)
    print("Reflexion Summary")
    print("=" * 60)
    print(f"Episodes: {summary['num_episodes']}")
    print(
        f"Successes: {summary['successes']}/"
        f"{summary['num_episodes']}"
    )
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(
        "Average decision steps: "
        f"{summary['average_decision_steps']:.2f}"
    )
    print(f"Average attempts: {summary['average_attempts']:.2f}")
    print(f"Total reflections: {summary['total_reflections']}")


if __name__ == "__main__":
    main()

