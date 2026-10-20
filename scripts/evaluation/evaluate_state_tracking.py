
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


MAX_STEPS = 50
TOP_K = 3

METHODS = {
    "react": (False, False),
    "verified": (True, False),
    "react_state": (False, True),
    "verified_state": (True, True),
}

SPLIT_SIZES = {
    "valid_seen": 140,
    "valid_unseen": 134,
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    try:
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_trajectory(path: Path, trajectory: Trajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    try:
        trajectory.save_json(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=SPLIT_SIZES,
        required=True,
    )

    parser.add_argument(
        "--method",
        choices=METHODS,
        required=True,
    )

    parser.add_argument(
        "--max-new",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.max_new is not None and args.max_new < 1:
        parser.error("--max-new must be >= 1")

    model = os.getenv("OPENAI_MODEL")

    if not model:
        raise RuntimeError("OPENAI_MODEL is not set")

    use_memory, use_state_tracking = METHODS[args.method]

    size = SPLIT_SIZES[args.split]

    manifest_path = Path(
        f"outputs/manifests/{args.split}_seed42_{size}.json"
    )

    games = load_manifest(manifest_path)

    if len(games) != size:
        raise RuntimeError(
            f"Expected {size} games, got {len(games)}"
        )

    store = None
    memory_hash = None

    if use_memory:
        memory_path = Path(
            f"outputs/memory/experience_store_"
            f"{model.replace('/', '_')}_train_seed42_1000.json"
        )

        store = ExperienceStore()
        store.load(memory_path)
        memory_hash = file_hash(memory_path)

    output_dir = (
        Path("outputs/ablation")
        / model.replace("/", "_")
        / args.split
        / args.method
    )

    source_paths = [
        Path("src/agents/react_agent.py"),
        Path("src/memory/state_tracker.py"),
        Path("src/core/protocol.py"),
        Path("src/memory/store.py"),
        Path("src/llm/client.py"),
        Path("src/envs/environment.py"),
    ]

    config = {
        "method": args.method,
        "split": args.split,
        "model": model,
        "max_steps": MAX_STEPS,
        "top_k": TOP_K,
        "use_memory": use_memory,
        "use_state_tracking": use_state_tracking,
        "manifest_hash": file_hash(manifest_path),
        "memory_hash": memory_hash,
        "source_hashes": {
            str(path): file_hash(path)
            for path in source_paths
        },
    }

    config_path = output_dir / "experiment_config.json"

    if config_path.exists():
        old_config = json.loads(
            config_path.read_text(encoding="utf-8")
        )

        if old_config != config:
            raise RuntimeError(
                "Experiment configuration changed. "
                "Do not mix trajectories from different configurations."
            )
    else:
        if list(output_dir.glob("episode_*.json")):
            raise RuntimeError(
                "Existing trajectories have no experiment configuration."
            )

        save_json(config_path, config)

    llm = LLMClient()
    new_runs = 0

    for index, game in enumerate(games):
        episode_path = output_dir / f"episode_{index:03d}.json"

        if episode_path.exists():
            trajectory = Trajectory.load_json(str(episode_path))

            if (
                Path(trajectory.gamefile).resolve()
                != Path(game.game_file).resolve()
            ):
                raise RuntimeError(
                    f"Game mismatch: {episode_path}"
                )

        else:
            if args.max_new is not None and new_runs >= args.max_new:
                break

            env = ALFWorldEnvironment(
                gamefile=str(game.game_file),
                max_steps=MAX_STEPS,
            )

            agent = ReActAgent(
                llm=llm,
                max_steps=MAX_STEPS,
                experience_store=store,
                experience_top_k=TOP_K,
                use_state_tracking=use_state_tracking,
            )

            try:
                trajectory = agent.run(env)
            finally:
                env.close()

            save_trajectory(episode_path, trajectory)
            new_runs += 1

        print(
            f"[{index + 1}/{len(games)}] "
            f"{args.method} "
            f"success={trajectory.success} "
            f"steps={len(trajectory.steps)}",
            flush=True,
        )

    records = []

    for index, game in enumerate(games):
        episode_path = output_dir / f"episode_{index:03d}.json"

        if not episode_path.exists():
            continue

        trajectory = Trajectory.load_json(str(episode_path))

        records.append({
            "episode": index,
            "game_id": game.id,
            "task_type": game.task_type,
            "task": trajectory.task,
            "success": trajectory.success,
            "truncated": trajectory.truncated,
            "decision_steps": len(trajectory.steps),
        })

    successes = sum(record["success"] for record in records)
    total_steps = sum(record["decision_steps"] for record in records)

    summary = {
        "method": args.method,
        "split": args.split,
        "model": model,
        "completed": len(records),
        "total": len(games),
        "successes": successes,
        "success_rate_completed": (
            successes / len(records) if records else 0
        ),
        "average_decision_steps": (
            total_steps / len(records) if records else 0
        ),
        "finished": len(records) == len(games),
        "episodes": records,
    }

    save_json(output_dir / "summary.json", summary)

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print("Method:", args.method)
    print("Split:", args.split)
    print("Completed:", f"{len(records)}/{len(games)}")
    print("Successes:", successes)
    print("Success rate (completed only):",
          f"{summary['success_rate_completed']:.2%}")
    print("Average steps:",
          f"{summary['average_decision_steps']:.2f}")
    print("Finished:", summary["finished"])
    print("Output:", output_dir)


if __name__ == "__main__":
    main()

