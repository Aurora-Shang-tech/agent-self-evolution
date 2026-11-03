
import argparse
import hashlib
import json
import os
from itertools import combinations
from math import comb
from pathlib import Path

from src.agents.react_agent import ReActAgent
from src.core.trajectory import Trajectory
from src.data.manifest import load_manifest
from src.envs.environment import ALFWorldEnvironment
from src.llm.client import LLMClient
from src.memory.experience import Experience
from src.memory.store import ExperienceStore


SPLIT_SIZES = {
    "valid_seen": 140,
    "valid_unseen": 134,
}

METHODS = (
    "original_130",
    "direct_update",
    "rule_strength_voting",
    "source_aware_revision",
)

MAX_STEPS = 50
TOP_K = 3

OUTPUT_ROOT = Path("outputs/memory_evolution/evaluations")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_name(path.name + ".tmp")

    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    os.replace(temp, path)


def save_trajectory(path: Path, trajectory: Trajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_name(path.name + ".tmp")

    try:
        trajectory.save_json(str(temp))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def memory_file(method: str, model: str) -> Path:
    if method == "original_130":
        return Path(
            "outputs/memory/"
            f"experience_store_{model.replace('/', '_')}"
            "_train_seed42_1000.json"
        )

    if method == "direct_update":
        return Path(
            "outputs/memory_evolution/"
            "train_full_v1/working_memory.json"
        )

    return (
        Path("outputs/memory_update_comparison")
        / "train_full_comparison"
        / method
        / "snapshot.json"
    )


def load_store(method: str, path: Path) -> ExperienceStore:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("memory")
    else:
        raise ValueError(f"Invalid memory format: {path}")

    if not isinstance(records, list) or not records:
        raise ValueError(f"Empty or invalid memory list: {path}")

    if isinstance(data, dict):
        declared_size = data.get("memory_size")

        if (
            declared_size is not None
            and declared_size != len(records)
        ):
            raise ValueError(f"Memory size mismatch: {path}")

    store = ExperienceStore()

    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid memory item {index}: {path}"
            )

        lesson = item.get("lesson")
        source_task = item.get("source_task")

        if not isinstance(lesson, str) or not lesson.strip():
            raise ValueError(
                f"Missing lesson at index {index}: {path}"
            )

        if (
            not isinstance(source_task, str)
            or not source_task.strip()
        ):
            raise ValueError(
                f"Missing source_task at index {index}: {path}"
            )

        store.add(
            Experience(
                failure_type=item.get(
                    "failure_type", "memory_evolution"
                ),
                lesson=lesson,
                source_task=source_task,
            )
        )

    print(f"Loaded {method}: {len(store)} memories")

    return store


def get_output_dir(
    model: str,
    split: str,
    method: str,
    use_state_tracking: bool,
) -> Path:
    root = (
        OUTPUT_ROOT
        / model.replace("/", "_")
        / split
    )

    # 保持之前有 State 实验的路径不变
    if use_state_tracking:
        return root / method

    # 无 State 实验独立保存
    return root / "no_state" / method


def source_hashes() -> dict:
    paths = [
        Path(__file__),
        Path("src/agents/react_agent.py"),
        Path("src/core/trajectory.py"),
        Path("src/core/protocol.py"),
        Path("src/envs/environment.py"),
        Path("src/memory/experience.py"),
        Path("src/memory/store.py"),
        Path("src/memory/state_tracker.py"),
        Path("src/llm/client.py"),
    ]

    return {
        str(path): file_hash(path)
        for path in paths
    }


def config_matches(old_config: dict, config: dict) -> bool:
    if old_config == config:
        return True

    # 兼容此前已运行的评估：
    # 只允许评估入口脚本自身的哈希发生变化
    old = dict(old_config)
    new = dict(config)

    if (
        old.get("use_state_tracking")
        != new.get("use_state_tracking")
    ):
        return False

    old_hashes = dict(old.pop("source_hashes", {}))
    new_hashes = dict(new.pop("source_hashes", {}))

    old_hashes.pop(str(Path(__file__)), None)
    new_hashes.pop(str(Path(__file__)), None)

    return old == new and old_hashes == new_hashes


def validate_experiment(
    output_dir: Path,
    config: dict,
) -> None:
    config_path = output_dir / "experiment_config.json"

    if config_path.exists():
        old_config = json.loads(
            config_path.read_text(encoding="utf-8")
        )

        if not config_matches(old_config, config):
            raise RuntimeError(
                "Experiment configuration changed. "
                f"Do not reuse old trajectories: {output_dir}"
            )

        return

    if list(output_dir.glob("episode_*.json")):
        raise RuntimeError(
            "Existing trajectories without experiment config: "
            f"{output_dir}"
        )

    save_json(config_path, config)


def build_record(index, game, trajectory: Trajectory) -> dict:
    if (
        Path(trajectory.gamefile).resolve()
        != Path(game.game_file).resolve()
    ):
        raise RuntimeError(
            f"Game mismatch at episode {index}"
        )

    steps = trajectory.steps

    decision_steps = len(steps)

    environment_steps = sum(
        bool(step.valid_format)
        for step in steps
    )

    format_errors = sum(
        not step.valid_format
        for step in steps
    )

    inadmissible_actions = sum(
        step.valid_format and not step.admissible
        for step in steps
    )

    return {
        "episode": index,
        "game_id": game.id,
        "task_type": game.task_type,
        "task": trajectory.task,
        "success": bool(trajectory.success),
        "truncated": bool(trajectory.truncated),
        "decision_steps": decision_steps,
        "environment_steps": environment_steps,
        "format_errors": format_errors,
        "inadmissible_actions": inadmissible_actions,
    }


def build_summary(records: list, total: int) -> dict:
    successes = sum(
        int(record["success"])
        for record in records
    )

    total_steps = sum(
        record["decision_steps"]
        for record in records
    )

    successful_steps = sum(
        record["decision_steps"]
        for record in records
        if record["success"]
    )

    return {
        "finished": len(records) == total,
        "completed": len(records),
        "total": total,
        "successes": successes,
        "success_rate": (
            successes / len(records)
            if records else None
        ),
        "average_decision_steps": (
            total_steps / len(records)
            if records else None
        ),
        "average_success_steps": (
            successful_steps / successes
            if successes else None
        ),
        "total_format_errors": sum(
            r["format_errors"] for r in records
        ),
        "total_inadmissible_actions": sum(
            r["inadmissible_actions"] for r in records
        ),
        "episodes": records,
    }


def evaluate(
    method: str,
    split: str,
    model: str,
    games: list,
    max_new: int | None,
    llm: LLMClient,
    use_state_tracking: bool,
) -> dict:

    memory_path = memory_file(method, model)
    store = load_store(method, memory_path)

    manifest_path = Path(
        f"outputs/manifests/"
        f"{split}_seed42_{len(games)}.json"
    )

    output_dir = get_output_dir(
        model,
        split,
        method,
        use_state_tracking,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "method": method,
        "model": model,
        "split": split,
        "manifest_hash": file_hash(manifest_path),
        "memory_path": str(memory_path),
        "memory_hash": file_hash(memory_path),
        "memory_size": len(store),
        "max_steps": MAX_STEPS,
        "top_k": TOP_K,
        "use_state_tracking": use_state_tracking,
        "source_hashes": source_hashes(),
    }

    validate_experiment(output_dir, config)

    records = []
    new_count = 0

    state_name = "state_on" if use_state_tracking else "state_off"

    print("\n" + "=" * 65)
    print(f"Method: {method}")
    print(f"Split: {split}")
    print(f"State Tracking: {state_name}")
    print(f"Memory size: {len(store)}")
    print(f"Tasks: {len(games)}")
    print(f"Output: {output_dir}")
    print("=" * 65)

    for index, game in enumerate(games):
        episode_path = (
            output_dir / f"episode_{index:03d}.json"
        )

        if episode_path.exists():
            trajectory = Trajectory.load_json(
                str(episode_path)
            )

        else:
            if max_new is not None and new_count >= max_new:
                continue

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

            new_count += 1

        record = build_record(
            index,
            game,
            trajectory,
        )

        records.append(record)

        summary = build_summary(
            records,
            len(games),
        )

        summary.update({
            "method": method,
            "split": split,
            "model": model,
            "state_tracking": use_state_tracking,
            "memory_size": len(store),
        })

        save_json(
            output_dir / "summary.json",
            summary,
        )

        print(
            f"[{method}][{state_name}][{split}] "
            f"{index + 1}/{len(games)} "
            f"success={record['success']} "
            f"steps={record['decision_steps']} "
            f"rate={summary['success_rate']:.2%}",
            flush=True,
        )

    summary = build_summary(
        records,
        len(games),
    )

    summary.update({
        "method": method,
        "split": split,
        "model": model,
        "state_tracking": use_state_tracking,
        "memory_size": len(store),
    })

    save_json(
        output_dir / "summary.json",
        summary,
    )

    print(f"\nSUMMARY: {method} ({state_name})")
    print(
        f"Completed: "
        f"{summary['completed']}/{summary['total']}"
    )
    print(f"Successes: {summary['successes']}")

    if summary["success_rate"] is not None:
        print(
            f"Success rate: "
            f"{summary['success_rate']:.2%}"
        )
        print(
            f"Average steps: "
            f"{summary['average_decision_steps']:.2f}"
        )

    print(
        "Average successful steps:",
        summary["average_success_steps"],
    )

    return summary


def mcnemar_exact(improved: int, regressed: int) -> float:
    n = improved + regressed

    if n == 0:
        return 1.0

    smaller = min(improved, regressed)

    return min(
        1.0,
        2.0 * sum(
            comb(n, i)
            for i in range(smaller + 1)
        ) / (2 ** n),
    )


def paired_comparison(
    a_name: str,
    a: dict,
    b_name: str,
    b: dict,
) -> dict:

    rows_a = {
        record["game_id"]: record
        for record in a["episodes"]
    }

    rows_b = {
        record["game_id"]: record
        for record in b["episodes"]
    }

    if (
        rows_a.keys() != rows_b.keys()
        or len(rows_a) != len(a["episodes"])
        or len(rows_b) != len(b["episodes"])
    ):
        raise RuntimeError(
            "Paired evaluation tasks do not match"
        )

    improved = sum(
        not rows_a[key]["success"]
        and rows_b[key]["success"]
        for key in rows_a
    )

    regressed = sum(
        rows_a[key]["success"]
        and not rows_b[key]["success"]
        for key in rows_a
    )

    return {
        "from": a_name,
        "to": b_name,
        "improved": improved,
        "regressed": regressed,
        "net": improved - regressed,
        "mcnemar_exact_p": mcnemar_exact(
            improved,
            regressed,
        ),
        "success_rate_change": (
            b["success_rate"] - a["success_rate"]
        ),
        "average_steps_change": (
            b["average_decision_steps"]
            - a["average_decision_steps"]
        ),
    }


def print_comparison(row: dict) -> None:
    print(
        f"{row['from']} -> {row['to']}: "
        f"+{row['improved']}/-{row['regressed']}, "
        f"net={row['net']:+d}, "
        f"steps={row['average_steps_change']:+.2f}, "
        f"McNemar p={row['mcnemar_exact_p']:.6f}"
    )


def compare_methods(
    summaries: dict,
    model: str,
    split: str,
    use_state_tracking: bool,
) -> None:

    finished = {
        method: summary
        for method, summary in summaries.items()
        if summary["finished"]
    }

    comparisons = []

    for a, b in combinations(finished, 2):
        comparisons.append(
            paired_comparison(
                a,
                finished[a],
                b,
                finished[b],
            )
        )

    root = (
        OUTPUT_ROOT
        / model.replace("/", "_")
        / split
    )

    if not use_state_tracking:
        root = root / "no_state"

    save_json(
        root / "comparison.json",
        {
            "state_tracking": use_state_tracking,
            "summaries": {
                method: {
                    key: value
                    for key, value in summary.items()
                    if key != "episodes"
                }
                for method, summary in summaries.items()
            },
            "paired_comparisons": comparisons,
        },
    )

    state_name = "ON" if use_state_tracking else "OFF"

    print("\n" + "=" * 65)
    print(f"PAIRED MEMORY COMPARISONS - STATE {state_name}")
    print("=" * 65)

    for row in comparisons:
        print_comparison(row)


def compare_states(
    all_summaries: dict,
    model: str,
    split: str,
) -> None:

    if "on" not in all_summaries or "off" not in all_summaries:
        return

    on = all_summaries["on"]
    off = all_summaries["off"]

    comparisons = []

    for method in METHODS:
        if method not in on or method not in off:
            continue

        if not on[method]["finished"] or not off[method]["finished"]:
            continue

        comparisons.append(
            paired_comparison(
                f"{method}:state_off",
                off[method],
                f"{method}:state_on",
                on[method],
            )
        )

    root = (
        OUTPUT_ROOT
        / model.replace("/", "_")
        / split
    )

    save_json(
        root / "state_comparison.json",
        {
            "paired_comparisons": comparisons,
        },
    )

    print("\n" + "=" * 65)
    print("PAIRED STATE COMPARISONS")
    print("=" * 65)

    for row in comparisons:
        print_comparison(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=SPLIT_SIZES,
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )

    parser.add_argument(
        "--state",
        choices=("on", "off", "both"),
        default="on",
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
        raise RuntimeError(
            "OPENAI_MODEL environment variable is not set"
        )

    size = SPLIT_SIZES[args.split]

    manifest_path = Path(
        f"outputs/manifests/"
        f"{args.split}_seed42_{size}.json"
    )

    games = load_manifest(manifest_path)

    if len(games) != size:
        raise RuntimeError(
            f"Unexpected task count: {len(games)}"
        )

    methods = list(dict.fromkeys(args.methods))

    for method in methods:
        path = memory_file(method, model)

        if not path.is_file():
            raise FileNotFoundError(path)

        load_store(method, path)

    llm = LLMClient()

    states = (
        ("off", "on")
        if args.state == "both"
        else (args.state,)
    )

    all_summaries = {}

    for state in states:
        use_state_tracking = state == "on"
        summaries = {}

        for method in methods:
            summaries[method] = evaluate(
                method=method,
                split=args.split,
                model=model,
                games=games,
                max_new=args.max_new,
                llm=llm,
                use_state_tracking=use_state_tracking,
            )

        all_summaries[state] = summaries

        compare_methods(
            summaries=summaries,
            model=model,
            split=args.split,
            use_state_tracking=use_state_tracking,
        )

    compare_states(
        all_summaries,
        model,
        args.split,
    )


if __name__ == "__main__":
    main()

