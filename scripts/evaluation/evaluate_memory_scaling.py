
import argparse
import hashlib
import json
import os
from pathlib import Path

from src.agents.react_agent import ReActAgent
from src.core.trajectory import Trajectory
from src.data.manifest import load_manifest
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient
from src.memory.store import ExperienceStore


SPLIT_SIZES = {
    "valid_seen": 140,
    "valid_unseen": 134,
}

MAX_STEPS = 50
TOP_K = 3
PERCENTAGES = (25, 50, 75)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_trajectory(path, trajectory):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    try:
        trajectory.save_json(str(temporary))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_trajectory(path, game):
    if not path.exists():
        return None

    try:
        trajectory = Trajectory.load_json(str(path))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError) as e:
        print(f"Invalid trajectory cache: {path} ({e})")
        path.unlink()
        return None

    if Path(trajectory.gamefile).resolve() != Path(game.game_file).resolve():
        raise RuntimeError(
            f"Game mismatch: {path}"
        )

    return trajectory


def build_record(index, game, trajectory):
    decision_steps = len(trajectory)

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

    assert decision_steps == environment_steps + format_errors
    assert environment_steps == inadmissible_actions + valid_actions

    return {
        "episode": index,
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
        "invalid_actions": format_errors + inadmissible_actions,
        "valid_actions": valid_actions,
    }


def build_summary(
    records,
    model,
    manifest_path,
    percentage,
    memory_size,
    memory_hash,
):
    n = len(records)
    successes = sum(int(r["success"]) for r in records)
    truncated = sum(int(r["truncated"]) for r in records)

    decision_steps = sum(r["decision_steps"] for r in records)
    environment_steps = sum(r["environment_steps"] for r in records)
    format_errors = sum(r["format_errors"] for r in records)
    inadmissible = sum(r["inadmissible_actions"] for r in records)
    valid_actions = sum(r["valid_actions"] for r in records)

    successful_steps = sum(
        r["decision_steps"]
        for r in records
        if r["success"]
    )

    task_type_stats = {}

    for record in records:
        task_type = record["task_type"]

        stats = task_type_stats.setdefault(
            task_type,
            {"episodes": 0, "successes": 0},
        )

        stats["episodes"] += 1
        stats["successes"] += int(record["success"])

    for stats in task_type_stats.values():
        stats["success_rate"] = (
            stats["successes"] / stats["episodes"]
        )

    return {
        "model": model,
        "manifest": str(manifest_path),
        "max_steps": MAX_STEPS,
        "experience_top_k": TOP_K,
        "memory_percentage": percentage,
        "memory_size": memory_size,
        "memory_hash": memory_hash,
        "num_episodes": n,
        "successes": successes,
        "success_rate": successes / n if n else 0.0,
        "truncated_episodes": truncated,
        "truncated_rate": truncated / n if n else 0.0,
        "average_decision_steps": (
            decision_steps / n if n else 0.0
        ),
        "average_success_steps": (
            successful_steps / successes if successes else 0.0
        ),
        "average_environment_steps": (
            environment_steps / n if n else 0.0
        ),
        "total_decision_steps": decision_steps,
        "total_environment_steps": environment_steps,
        "format_errors": format_errors,
        "inadmissible_actions": inadmissible,
        "format_valid_rate": (
            environment_steps / decision_steps
            if decision_steps else 0.0
        ),
        "admissible_action_rate": (
            valid_actions / environment_steps
            if environment_steps else 0.0
        ),
        "valid_action_rate": (
            valid_actions / decision_steps
            if decision_steps else 0.0
        ),
        "task_type_stats": task_type_stats,
        "episodes": records,
    }


def validate_experiment(output_dir, config):
    config_path = output_dir / "experiment_config.json"

    if config_path.exists():
        with config_path.open(encoding="utf-8") as f:
            old_config = json.load(f)

        if old_config != config:
            raise RuntimeError(
                f"Experiment configuration changed: {config_path}"
            )

        return

    existing_episodes = list(output_dir.glob("episode_*.json"))

    if existing_episodes:
        raise RuntimeError(
            "Existing trajectory caches have no experiment_config.json. "
            "Verify their configuration before resuming."
        )

    save_json(config_path, config)


def evaluate(split, percentage, llm, model):
    size = SPLIT_SIZES[split]

    manifest_path = Path(
        f"outputs/manifests/{split}_seed42_{size}.json"
    )

    memory_path = Path(
        f"outputs/memory_scaling/stores/memory_{percentage}.json"
    )

    output_dir = Path(
        "outputs/memory_scaling/evaluations/"
        f"{model.replace('/', '_')}_{split}_"
        f"seed42_{size}_memory{percentage}_"
        f"steps{MAX_STEPS}_topk{TOP_K}"
    )

    if not memory_path.exists():
        raise FileNotFoundError(memory_path)

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    store = ExperienceStore()
    store.load(memory_path)

    games = load_manifest(manifest_path)
    memory_hash = file_hash(memory_path)

    config = {
        "model": model,
        "split": split,
        "percentage": percentage,
        "max_steps": MAX_STEPS,
        "top_k": TOP_K,
        "memory_hash": memory_hash,
        "manifest_hash": file_hash(manifest_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    validate_experiment(output_dir, config)

    records = []

    print(f"\nSplit: {split}")
    print(f"Memory: M{percentage} ({len(store)} experiences)")
    print(f"Tasks: {len(games)}")
    print(f"Output: {output_dir}")

    for index, game in enumerate(games):
        episode_path = output_dir / f"episode_{index:03d}.json"

        trajectory = load_trajectory(episode_path, game)

        if trajectory is None:
            env = ALFWorldEnvironment(
                gamefile=str(game.game_file),
                max_steps=MAX_STEPS,
            )

            agent = ReActAgent(
                llm=llm,
                max_steps=MAX_STEPS,
                experience_store=store,
                experience_top_k=TOP_K,
            )

            try:
                trajectory = agent.run(env)
            finally:
                env.close()

            save_trajectory(episode_path, trajectory)

        record = build_record(index, game, trajectory)
        records.append(record)

        summary = build_summary(
            records,
            model,
            manifest_path,
            percentage,
            len(store),
            memory_hash,
        )

        save_json(output_dir / "summary.json", summary)

        print(
            f"[M{percentage}][{split}] "
            f"{index + 1}/{len(games)} "
            f"success={trajectory.success} "
            f"steps={record['decision_steps']} "
            f"rate={summary['success_rate']:.2%}",
            flush=True,
        )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=("valid_seen", "valid_unseen"),
        required=True,
    )

    parser.add_argument(
        "--percentages",
        type=int,
        nargs="+",
        default=list(PERCENTAGES),
    )

    args = parser.parse_args()

    model = os.getenv("OPENAI_MODEL")

    if not model:
        raise RuntimeError("OPENAI_MODEL is not set")

    if any(p not in (0, 25, 50, 75, 100) for p in args.percentages):
        raise ValueError("Invalid memory percentage")

    llm = LLMClient()

    for percentage in args.percentages:
        summary = evaluate(
            args.split,
            percentage,
            llm,
            model,
        )

        print(
            f"\nM{percentage} {args.split}: "
            f"{summary['successes']}/{summary['num_episodes']} "
            f"({summary['success_rate']:.2%}), "
            f"avg_steps={summary['average_decision_steps']:.2f}, "
            f"avg_success_steps={summary['average_success_steps']:.2f}\n"
        )


if __name__ == "__main__":
    main()

